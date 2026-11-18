"""Structured logging of the decisions taken while producing a number.

:class:`EventKind` is the closed vocabulary and :func:`log_event` the only emitter. Keys
sort as in :func:`~tfidf_stability.utils.io.canonical_json`. No handler is installed on
import; :func:`configure` is the opt-in. Rationale: docs/engineering_notes.md#logging-provenance
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

from tfidf_stability.utils.hashing import hash_text
from tfidf_stability.utils.io import canonical_json

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

#: Names only ``levelname``, ``name`` and ``message``. The volatile
#: :class:`logging.LogRecord` fields (``created``, ``process``, ``thread``) are left out,
#: so two identical runs emit byte-identical output under this format.
DETERMINISTIC_FORMAT = "%(levelname)s %(name)s %(message)s"

#: :data:`DETERMINISTIC_FORMAT` with wall-clock time, for watching a long sweep from a
#: terminal. Output under it is not for hashing.
TIMESTAMPED_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

#: ``LogRecord`` attribute carrying the structured payload. Prefixed because
#: ``extra=`` writes straight onto the record and a collision with a stdlib
#: attribute such as ``name`` or ``msg`` raises at emit time.
_EVENT_ATTR = "tfidf_event"

#: Marks the handler :func:`configure` installed, so a second call can replace
#: its own handler without disturbing one the host application added.
_INSTALLED_MARK = "_tfidf_stability_installed"


class EventKind(str, Enum):
    """The closed set of decisions a run records.

    Each member answers a question about how a published number was produced, and
    the set is short enough to state in full. ``DIAGNOSTIC`` accompanies :func:`warnings.warn`
    and never replaces it: :func:`logging.captureWarnings` would stop warnings failing tests.
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
    #: :func:`warnings.warn` call; never instead of it. The pytest ``"error"`` filter
    #: makes each warning a test failure, so :func:`logging.captureWarnings` is not used.
    #: Rationale: docs/engineering_notes.md#logging-provenance
    DIAGNOSTIC = "diagnostic"
    #: The digest of something produced, so an artefact can be tied to the run
    #: that wrote it without opening either.
    DIGEST = "digest"

    def __str__(self) -> str:
        return self.value

    @property
    def level(self) -> int:
        """Severity, fixed per kind rather than chosen per call site.

        A reader filtering at ``WARNING`` gets the diagnostics and nothing else;
        a level chosen per call site would break that.
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

    def __post_init__(self) -> None:
        """Normalise ``kind`` to its member.

        ``EventRecorder.of_kind`` selects on the member, so a recorded string
        would be invisible to a query for the kind it names.
        """
        object.__setattr__(self, "kind", EventKind(self.kind))

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

    Only structured events are kept, so :meth:`EventRecorder.digest` covers decisions
    alone; free-text records from other libraries would add paths, addresses and timings.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.events: list[Event] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the payload the call site built, never the :class:`logging.LogRecord`.

        The record's ``created``, ``msecs``, ``process`` and ``thread`` differ between
        two identical runs.
        """
        event = getattr(record, _EVENT_ATTR, None)
        if isinstance(event, Event):
            self.events.append(event)

    def of_kind(self, kind: EventKind) -> list[Event]:
        """Every recorded event of one kind, in the order they were emitted.

        The argument is coerced because ``EventKind`` is a ``str`` enum: a spelling
        equals a member without being it, and the filter below compares with ``is``.
        An uncoerced string selects nothing and reports a kind as unrecorded.
        """
        wanted = EventKind(kind)
        return [event for event in self.events if event.kind is wanted]

    def to_list(self) -> list[dict[str, Any]]:
        """The run record: structured events in emission order."""
        return [event.to_dict() for event in self.events]

    def digest(self) -> str:
        """SHA-256 over the recorded decisions, in emission order.

        Selecting the native backend after a fallback is a different run from selecting
        it outright. ``canonical_json`` renders a non-finite field as ``null``.
        """
        return hash_text(canonical_json(self.to_list(), indent=None))


@contextmanager
def capture(level: int = logging.DEBUG) -> Iterator[EventRecorder]:
    """Collect this package's events for the duration of the block.

    Yields an :class:`EventRecorder` and restores the package logger's previous level
    and handler set on exit, the raising case included.
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
    """Install and return the package handler; applications call it, imports do not.

    Touches only the ``tfidf_stability`` logger and replaces the handler an earlier call
    installed. ``level`` is the threshold, ``stream`` defaults to ``sys.stderr``, and
    ``timestamps`` replaces :data:`DETERMINISTIC_FORMAT` with wall-clock lines that differ.
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

    Removes only handlers this module installed. No :class:`~logging.NullHandler` is added:
    with no handler on the chain, ``WARNING`` records reach stderr via :data:`logging.lastResort`.
    """
    logger = logging.getLogger(ROOT_NAME)
    for handler in [h for h in logger.handlers if getattr(h, _INSTALLED_MARK, False)]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
