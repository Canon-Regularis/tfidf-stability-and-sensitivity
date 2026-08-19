"""Structured logging for provenance.

A run manifest records the inputs a number depended on: data, config, build.
This module records the decisions taken while producing it: backend selected,
reduction policy in force, degenerate cases hit, diagnostics fired, digests of
what came out. Between the two, a surprising number can be accounted for without
rerunning anything. General-purpose tracing is out of scope.

Closed vocabulary. :class:`EventKind` names every recordable kind and
:func:`log_event` is the only emitter. Lines render as ``kind key=value ...``
with keys sorted, so two runs that took the same decisions produce the same
lines whatever order a call site passed its fields, matching the
canonicalisation :func:`~tfidf_stability.utils.hashing.hash_json` applies to
configs.

Volatile state stays off the reproducibility surface. A
:class:`logging.LogRecord` carries ``created``, ``msecs``, ``relativeCreated``,
``process`` and ``thread``, all of which differ between two identical runs; none
is read here. :data:`DETERMINISTIC_FORMAT` names only ``levelname``, ``name``
and ``message``, and :class:`EventRecorder` stores the structured payload the
call site supplied rather than the record it arrived on, so
:meth:`EventRecorder.digest` covers the decisions alone. Wall-clock timestamps
are opt-in (``configure(timestamps=True)``) and leave the reproducible set.

Warnings do not route through here. ``pyproject.toml`` sets
``filterwarnings = ["error"]``, so ``TauExceedsScoreRangeWarning`` and
``ChainInflationWarning`` are test failures; :func:`logging.captureWarnings`
would demote them to a line nobody reads. An ``EventKind.DIAGNOSTIC`` event
records that a diagnostic fired alongside the :func:`warnings.warn` call.

No handler is installed on import, not even a :class:`~logging.NullHandler`:
with nothing on the chain, records at ``WARNING`` and above still reach stderr
through :data:`logging.lastResort`, and a degenerate configuration is a genuine
finding here. :func:`configure` is the opt-in the CLI calls.
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

    Every member answers a question that has been asked of a published number,
    which keeps the set short enough to state in full.
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

    def __str__(self) -> str:
        return self.value

    @property
    def level(self) -> int:
        """Severity, fixed per kind rather than chosen per call site.

        Part of the vocabulary: a reader filtering at ``WARNING`` gets the
        diagnostics and nothing else, which fails once call sites choose.
        """
        return logging.WARNING if self is EventKind.DIAGNOSTIC else logging.INFO


_PLAIN = re.compile(r"[A-Za-z0-9_.:/@+-]+")


def _render_value(value: object) -> str:
    """Render one field value unambiguously, on a single line."""
    if isinstance(value, float):
        # repr keeps every bit: a margin of 5e-324 and a margin of 0.0 separate a
        # near-tie from an exact tie, and fixed precision would round that
        # distinction away.
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

    Fields are a sorted tuple rather than a mapping, so an event is hashable and
    compares equal whatever order the call site listed them in.
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
    as-is; anything else is reparented under :data:`ROOT_NAME`, so nothing emits
    outside the subtree :func:`configure` controls.
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

    Only structured events are kept: free-text records from other libraries would
    put whatever they happened to say (paths, addresses, timings) into the digest.
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

        Order-sensitive: selecting the native backend after a fallback is a
        different run from selecting it outright.
        """
        return hash_json(self.to_list())


@contextmanager
def capture(level: int = logging.DEBUG) -> Iterator[EventRecorder]:
    """Collect this package's events for the duration of the block.

    Restores the package logger's previous level and handler set on exit, the
    raising case included, so a captured section leaves the logging system as it
    found it.
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
    """Install this package's handler. For applications to call; imports must not.

    Touches only the ``tfidf_stability`` logger, so it cannot redirect or silence
    anything a host application had already set up. Idempotent: a second call
    replaces the handler the first installed rather than stacking a duplicate, so
    an entry point invoked twice in one process does not double every line.

    Args:
        level: Threshold for the package logger.
        stream: Destination; defaults to ``sys.stderr``.
        timestamps: Prefix each line with wall-clock time. Off by default: it is
            the one thing here that makes two identical runs differ.

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

    Removes only handlers this module installed; one the host application
    attached to our logger is its business.
    """
    logger = logging.getLogger(ROOT_NAME)
    for handler in [h for h in logger.handlers if getattr(h, _INSTALLED_MARK, False)]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
