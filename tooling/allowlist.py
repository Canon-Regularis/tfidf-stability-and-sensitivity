"""Argued equivalences: the format both mutation campaigns record them in.

An entry names the site a mutation applies to, stamps the source line it was
written about, and gives the argument for why the mutant cannot be observed. The
two keys differ -- the Python one carries the operator kind, the C++ one a
column -- but the stamp, the rule that an entry without an argument is ignored,
and the parse are one thing, held here rather than in each campaign.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

__all__ = [
    "STAMP_PREFIX",
    "CppKey",
    "Entry",
    "PythonKey",
    "Verdict",
    "claims_cpp",
    "claims_python",
    "fingerprint",
    "parse_cpp",
    "parse_python",
    "verdict",
    "without_stamp",
]

#: ``(line, kind, before, after)``. The kind names the operator the Python
#: mutator applied, which a line alone does not determine.
PythonKey = tuple[int, str, str, str]

#: ``(line, column, before, after)``. The column is part of the key because a
#: line can carry two candidates that read the same and are equivalent for
#: different reasons.
CppKey = tuple[int, int, str, str]

STAMP_PREFIX = "src="
_STAMP_WIDTH = len(STAMP_PREFIX) + 8

_Key = TypeVar("_Key", PythonKey, CppKey)


@dataclass(frozen=True, slots=True)
class Verdict(Generic[_Key]):
    """Survivors carrying no claim, and claims matching no survivor.

    Both directions fail a campaign: an unexplained survivor is the gap a
    campaign exists to find, and a claim nobody re-reads is how an allowlist
    turns into a blanket suppression.
    """

    unclaimed: tuple[_Key, ...]
    stale: tuple[_Key, ...]
    covered: int


def verdict(survivors: Sequence[_Key], claims: Mapping[_Key, str]) -> Verdict[_Key]:
    """Split survivors and claims against each other.

    Counted rather than collected: two mutable sites on one line can share a
    key, and reporting one claim against two survivors would read as though
    one were unaccounted for.
    """
    counts = Counter(survivors)
    return Verdict(
        unclaimed=tuple(sorted(key for key in counts if key not in claims)),
        stale=tuple(sorted(key for key in claims if key not in counts)),
        covered=sum(n for key, n in counts.items() if key in claims),
    )


def fingerprint(line: str) -> str:
    """The stamp recorded beside an allowlist entry.

    Whitespace-normalised, so reindenting a line does not invalidate an argument
    about it, while changing the expression does.
    """
    return hashlib.sha256(" ".join(line.split()).encode("utf-8")).hexdigest()[:8]


def without_stamp(reason: str) -> str:
    """Drop a leading ``src=<8 hex>`` marker from an allowlist reason."""
    text = reason.strip()
    head, _, rest = text.partition(" ")
    if head.startswith(STAMP_PREFIX) and len(head) == _STAMP_WIDTH:
        return rest.strip()
    return text


@dataclass(frozen=True, slots=True)
class Entry:
    """One allowlist line, parsed, with what a rewrite needs to put it back.

    ``index`` is the entry's position in the file and ``raw`` its original text,
    so changing the line number rewrites one field and leaves every other byte
    of the file alone.
    """

    path: str
    line: int
    column: int | None
    kind: str | None
    before: str
    after: str
    stamp: str
    reason: str
    index: int
    raw: str

    @property
    def argued(self) -> bool:
        """Whether the entry carries an argument rather than only a stamp."""
        return bool(self.reason)


def _split(raw: str) -> tuple[list[str], str]:
    """Fields before the first ``#``, and the comment after it."""
    statement, _, comment = raw.partition("#")
    return statement.split(), comment.strip()


def _stamp_of(comment: str) -> str:
    head, _, _ = comment.strip().partition(" ")
    return head if head.startswith(STAMP_PREFIX) and len(head) == _STAMP_WIDTH else ""


def parse_python(text: str) -> list[Entry]:
    """Every well-formed entry in the Python allowlist, in file order.

    Malformed lines, blanks and comments are skipped rather than reported: the
    file is prose as well as data, and its header is not an entry.
    """
    entries: list[Entry] = []
    for index, raw in enumerate(text.splitlines()):
        fields, comment = _split(raw)
        if len(fields) < 6 or fields[4] != "->":
            continue
        entries.append(
            Entry(
                path=fields[0],
                line=int(fields[1]),
                column=None,
                kind=fields[2],
                before=fields[3],
                after=fields[5],
                stamp=_stamp_of(comment),
                reason=without_stamp(comment),
                index=index,
                raw=raw,
            )
        )
    return entries


def parse_cpp(text: str) -> list[Entry]:
    """Every well-formed entry in the C++ allowlist, in file order.

    An entry whose position carries no column is skipped: it cannot say which
    token on the line it means.
    """
    entries: list[Entry] = []
    for index, raw in enumerate(text.splitlines()):
        fields, comment = _split(raw)
        if len(fields) < 5 or fields[3] != "->":
            continue
        where, _, column = fields[1].partition(":")
        if not column:
            continue
        entries.append(
            Entry(
                path=fields[0],
                line=int(where),
                column=int(column),
                kind=None,
                before=fields[2],
                after=fields[4],
                stamp=_stamp_of(comment),
                reason=without_stamp(comment),
                index=index,
                raw=raw,
            )
        )
    return entries


def claims_python(entries: list[Entry], module: Path) -> dict[PythonKey, str]:
    """Arguments for one module, keyed the way the Python campaign keys sites.

    An entry with no argument is dropped: an entry that does not say why the
    mutation cannot be observed is a suppression, and the file holds arguments.
    """
    wanted = module.as_posix()
    return {
        (e.line, str(e.kind), e.before, e.after): e.reason
        for e in entries
        if e.path == wanted and e.argued
    }


def claims_cpp(entries: list[Entry], module: Path) -> dict[CppKey, str]:
    """Arguments for one header, keyed the way the C++ campaign keys sites."""
    wanted = module.as_posix()
    return {
        (e.line, int(e.column or 0), e.before, e.after): e.reason
        for e in entries
        if e.path == wanted and e.argued
    }
