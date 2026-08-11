"""Structured logging for provenance, not for debugging.

A run manifest records the *inputs* a number depended on -- data, config, build.
This module records the *decisions taken while producing it*: which backend the
process selected, which reduction policy was in force, which degenerate cases
the data actually hit, which warning-worthy diagnostics fired, and the digests
of what came out. Between the two, a surprising number can be accounted for
without rerunning anything, which is the whole reason this repository logs at
all. General-purpose tracing is explicitly not the goal.

Four decisions follow from that purpose.

**A closed vocabulary, not free text.** :class:`EventKind` names every kind of
thing worth recording, and :func:`log_event` is the only way to emit one. Lines
render as ``kind key=value ...`` with the keys sorted, so two runs that took the
same decisions produce the same lines regardless of the order a call site
happened to pass its fields -- the same canonicalisation rule
:func:`~tfidf_stability.utils.hashing.hash_json` applies to configs.

**Volatile state never reaches the reproducibility surface.** A
:class:`logging.LogRecord` carries ``created``, ``msecs``, ``relativeCreated``,
``process`` and ``thread``, and every one of them differs between two identical
runs. None is ever read here. :data:`DETERMINISTIC_FORMAT` names only
``levelname``, ``name`` and ``message``; :class:`EventRecorder` stores only the
structured payload the call site supplied and never the record it arrived on.
So :meth:`EventRecorder.digest` covers the decisions and not when, where or by
which process they were taken -- the same separation
:func:`~tfidf_stability.utils.io.strip_volatile` makes for manifests. Wall-clock
timestamps exist (``configure(timestamps=True)``) but are opt-in, and turning
them on is what takes the output out of the reproducible set.

**Warnings are not routed through here.** ``TauExceedsScoreRangeWarning`` and
``ChainInflationWarning`` stay :func:`warnings.warn` calls, and this module
never calls :func:`logging.captureWarnings`. ``pyproject.toml`` sets
``filterwarnings = ["error"]``, which makes those diagnostics load-bearing test
failures rather than notices; capturing them into the logging system would
demote a hard failure into a line nobody reads. An ``EventKind.DIAGNOSTIC``
event records *that* a diagnostic fired and never stands in for the warning.

**No handler is installed on import.** Whether and how to log is the
application's decision, so importing this package leaves the logging system
exactly as it found it, root logger included; :func:`configure` is the opt-in
the CLI calls. Not even a :class:`~logging.NullHandler` is installed: with no
handler anywhere on the chain, records at ``WARNING`` and above still reach
stderr through :data:`logging.lastResort`, and in a project where a degenerate
configuration is a genuine finding, silencing that default would be the wrong
favour.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import IO, Any

from tfidf_stability.utils.hashing import hash_json

__all__ = [
    "DETERMINISTIC_FORMAT",
    "ROOT_NAME",
    "TIMESTAMPED_FORMAT",
    "Event",
    "EventKind",
    "EventRecorder",
    "capture",
    "configure",
    "get_logger",
    "log_event",
    "reset",
]

#: Every logger in this package is this one or a child of it, so a single
#: :func:`configure` call governs the whole package.
ROOT_NAME = "tfidf_stability"

#: Reproducible: no ``asctime``, no ``process``, no ``thread``. Two identical
#: runs emit byte-identical output under this format.
DETERMINISTIC_FORMAT = "%(levelname)s %(name)s %(message)s"

#: For watching a long sweep from a terminal. Not for anything that gets hashed.
TIMESTAMPED_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

#: ``LogRecord`` attribute carrying the structured payload. Prefixed because
#: ``extra=`` writes straight onto the record and a collision with a stdlib
#: attribute such as ``name`` or ``msg`` raises at emit time.
_EVENT_ATTR = "tfidf_event"

#: Marks the handler :func:`configure` installed, so a second call can replace
#: its own handler without disturbing one the host application added.
_INSTALLED_MARK = "_tfidf_stability_installed"


class EventKind(str, Enum):
    """The closed set of decisions worth recording.

    Closed on purpose. An open-ended log becomes debugging chatter that nobody
    reads; this vocabulary is short enough that a reader can be told what a run
    record contains, and every member answers a question that has actually been
    asked of a published number.
    """

    #: Which backend this process selected, and why the other one was not used.
    BACKEND_SELECTED = "backend_selected"
    #: Which summation policy a stage ran under. Changes the low bits, and so
    #: changes which near-ties resolve which way.
    REDUCTION_POLICY = "reduction_policy"
    #: A defined-but-degenerate case was reached: a zero-norm document, a query
    #: that embeds to the zero vector, an undefined margin.
    DEGENERATE = "degenerate"
    #: A warning-worthy diagnostic fired. Records the occurrence alongside the
    #: :func:`warnings.warn` call; never instead of it.
    DIAGNOSTIC = "diagnostic"
    #: The digest of something produced, so an artefact can be tied to the run
    #: that wrote it without opening either.
    DIGEST = "digest"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def level(self) -> int:
        """Severity, fixed per kind rather than chosen per call site.

        Fixed because the level is part of the vocabulary: a reader filtering at
        ``WARNING`` must get exactly the diagnostics and nothing else, which is
        not true if each call site picks for itself.
        """
        return logging.WARNING if self is EventKind.DIAGNOSTIC else logging.INFO


_PLAIN = re.compile(r"[A-Za-z0-9_.:/@+-]+")


def _render_value(value: object) -> str:
    """Render one field value unambiguously, on a single line."""
    if isinstance(value, float):
        # repr, not a fixed-precision format: a margin of 5e-324 and a margin of
        # 0.0 are the difference between a near-tie and an exact tie, and any
        # rounding in the rendering erases exactly that distinction.
        return repr(value)
    if value is None or isinstance(value, (bool, int)):
        return json.dumps(value)
    # Collapsed to one line and quoted when not plainly parseable, so a reason
    # string containing spaces or newlines cannot forge extra key=value pairs.
    text = " ".join(str(value).split())
    return text if _PLAIN.fullmatch(text) else json.dumps(text, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Event:
    """One recorded decision: a kind, and the fields that qualify it.

    Fields are held as a sorted tuple rather than a mapping so that an event is
    hashable and compares equal regardless of the order its call site listed
    them -- two runs that decided the same thing produce equal events.
    """

    kind: EventKind
    fields: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def build(cls, kind: EventKind, fields: Mapping[str, Any]) -> Event:
        """Canonicalise a mapping of fields into an event."""
        return cls(kind, tuple(sorted(fields.items())))

    def render(self) -> str:
        """The single-line human form, ``kind key=value ...``."""
        return str(self.kind) + "".join(f" {k}={_render_value(v)}" for k, v in self.fields)

    def to_dict(self) -> dict[str, Any]:
        """The structured form, for a run record."""
        return {"kind": self.kind.value, "fields": dict(self.fields)}


def get_logger(name: str | None = None) -> logging.Logger:
    """The package logger, or the child of it named for a module.

    Call sites pass ``__name__``. Names already inside the package are used
    as-is; anything else is reparented under :data:`ROOT_NAME`, so no caller can
    accidentally emit outside the subtree :func:`configure` controls.
    """
    if name is None or name == ROOT_NAME:
        return logging.getLogger(ROOT_NAME)
    if name.startswith(f"{ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def log_event(logger: logging.Logger, kind: EventKind, /, **fields: Any) -> Event:
    """Emit one event and return it, for callers that also want to store it.

    ``kind`` is positional-only so that a field may legitimately be called
    ``kind`` without shadowing the parameter.
    """
    event = Event.build(kind, fields)
    # The rendered line is passed as the message with no args, so a literal '%'
    # inside a value is never treated as a format specifier.
    logger.log(kind.level, event.render(), extra={_EVENT_ATTR: event})
    return event


class EventRecorder(logging.Handler):
    """Collects events into a run record that can be digested.

    Only structured events are kept. Free-text records from third-party
    libraries are for a human to read and would put whatever they happened to
    say -- paths, addresses, timings -- into the digest.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.events: list[Event] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the payload the call site built, never the record itself."""
        event = getattr(record, _EVENT_ATTR, None)
        if isinstance(event, Event):
            self.events.append(event)

    def of_kind(self, kind: EventKind) -> list[Event]:
        """Every recorded event of one kind, in the order they were emitted."""
        return [event for event in self.events if event.kind is kind]

    def to_list(self) -> list[dict[str, Any]]:
        """The run record: structured events in emission order."""
        return [event.to_dict() for event in self.events]

    def digest(self) -> str:
        """SHA-256 over the recorded decisions.

        Order-sensitive, because the order decisions were taken in is itself
        part of what happened -- selecting the native backend after a fallback
        is not the same run as selecting it first.
        """
        return hash_json(self.to_list())


@contextmanager
def capture(level: int = logging.DEBUG) -> Iterator[EventRecorder]:
    """Collect this package's events for the duration of the block.

    Restores the package logger's previous level and handler set on exit,
    including when the body raises, so a captured section cannot leave the
    logging system altered for whatever runs next.
    """
    logger = logging.getLogger(ROOT_NAME)
    recorder = EventRecorder()
    previous_level = logger.level
    logger.addHandler(recorder)
    logger.setLevel(level)
    try:
        yield recorder
    finally:
        logger.removeHandler(recorder)
        logger.setLevel(previous_level)


def configure(
    *,
    level: int | str = logging.INFO,
    stream: IO[str] | None = None,
    timestamps: bool = False,
) -> logging.Handler:
    """Install this package's handler. For applications to call, never imports.

    Only the ``tfidf_stability`` logger is touched -- never the root logger, and
    never another library's -- so calling this cannot redirect or silence
    anything a host application had already set up.

    Idempotent: a second call replaces the handler the first installed rather
    than stacking a duplicate, so an entry point invoked twice in one process
    does not emit every line twice.

    Args:
        level: Threshold for the package logger.
        stream: Destination; defaults to ``sys.stderr``.
        timestamps: Prefix each line with wall-clock time. Off by default
            because it is the one thing here that makes output differ between
            two identical runs, and reproducible output is the point.

    Returns:
        The installed handler, for a caller that wants to adjust it further.
    """
    logger = logging.getLogger(ROOT_NAME)
    reset()
    handler = logging.StreamHandler() if stream is None else logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter(TIMESTAMPED_FORMAT if timestamps else DETERMINISTIC_FORMAT)
    )
    setattr(handler, _INSTALLED_MARK, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler


def reset() -> None:
    """Undo :func:`configure`, returning the package logger to its import state.

    Removes only handlers this module installed. A handler the host application
    attached to our logger is its business, and silently detaching it would be
    the same overreach as configuring the root logger in the first place.
    """
    logger = logging.getLogger(ROOT_NAME)
    for handler in [h for h in logger.handlers if getattr(h, _INSTALLED_MARK, False)]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
