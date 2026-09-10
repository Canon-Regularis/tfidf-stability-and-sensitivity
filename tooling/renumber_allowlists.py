"""Move allowlist entries onto the lines their stamps name.

Editing a covered file shifts the lines below the edit, and both mutation gates
then fail because an entry names a line that no longer holds what it describes.
The stamp records the source line the argument was written about, so relocating
an entry is a search for that fingerprint rather than a guess at the nearest
line, which is what the gates refuse.

An entry whose stamp matches no line is left alone and reported: the expression
it argues about has changed, and only a reader can say whether the argument
still holds. An entry whose stamp matches two lines is reported the same way.
There is no mode that rewrites a stamp; that would re-bless an argument nobody
has re-read.

Usage::

    python -m tooling.renumber_allowlists --check
    python -m tooling.renumber_allowlists --write
    python -m tooling.renumber_allowlists --write --only src/tfidf_stability/x.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from tooling.allowlist import Entry, fingerprint, parse_cpp, parse_python

REPO = Path(__file__).resolve().parents[1]

#: allowlist file -> the parser for its format.
ALLOWLISTS = {
    REPO / "configs" / "equivalent_mutants.txt": parse_python,
    REPO / "configs" / "equivalent_mutants_cpp.txt": parse_cpp,
}


@dataclass(frozen=True, slots=True)
class Move:
    """One entry that should name a different line."""

    entry: Entry
    to_line: int


@dataclass(frozen=True, slots=True)
class Refusal:
    """One entry the tool declines to move, with what a reader needs."""

    entry: Entry
    detail: str


def _position(entry: Entry, line: int) -> str:
    """The position field as the entry's format spells it."""
    return f"{line}:{entry.column}" if entry.column is not None else str(line)


def _rewritten(entry: Entry, line: int) -> str:
    """``entry.raw`` with only the position field changed.

    Located by walking past the path rather than by re-joining fields, so every
    other byte of the line, including its column alignment, survives.
    """
    start = entry.raw.index(entry.path) + len(entry.path)
    while start < len(entry.raw) and entry.raw[start].isspace():
        start += 1
    end = start
    while end < len(entry.raw) and not entry.raw[end].isspace():
        end += 1
    return entry.raw[:start] + _position(entry, line) + entry.raw[end:]


def plan_file(
    entries: list[Entry], only: str | None = None, repo: Path = REPO
) -> tuple[list[Move], list[Refusal]]:
    """Decide what to do with every entry of one allowlist.

    ``repo`` is a parameter so a test can point the whole decision at a crafted
    tree rather than at the repository it is running in.
    """
    moves: list[Move] = []
    refusals: list[Refusal] = []
    bodies: dict[str, list[str]] = {}

    for entry in entries:
        if only is not None and entry.path != only:
            continue
        if not entry.stamp:
            refusals.append(Refusal(entry, "carries no source stamp"))
            continue
        source = repo / entry.path
        if not source.exists():
            refusals.append(Refusal(entry, "names a file that is gone"))
            continue
        body = bodies.setdefault(entry.path, source.read_text(encoding="utf-8").split("\n"))

        wanted = entry.stamp.removeprefix("src=")
        if 1 <= entry.line <= len(body) and fingerprint(body[entry.line - 1]) == wanted:
            continue

        candidates = [n for n, text in enumerate(body, 1) if fingerprint(text) == wanted]
        if entry.column is not None:
            # A C++ entry's `before` is the source token the mutation replaces,
            # so it must still be on the line. A Python entry's is the name of
            # an AST operator (`GtE`), which never appears in the source.
            candidates = [n for n in candidates if entry.before in body[n - 1]]

        if not candidates:
            inside = 1 <= entry.line <= len(body)
            here = body[entry.line - 1].strip() if inside else "(past the end)"
            refusals.append(
                Refusal(
                    entry,
                    f"no line carries its stamp; line {entry.line} now reads: {here[:70]}",
                )
            )
        elif len(candidates) > 1:
            refusals.append(
                Refusal(entry, f"its stamp matches lines {candidates}, so the move is ambiguous")
            )
        else:
            moves.append(Move(entry, candidates[0]))
    return moves, refusals


def apply_moves(text: str, moves: list[Move]) -> str:
    """Rewrite the allowlist, changing only the position field of each move."""
    lines = text.split("\n")
    for move in moves:
        lines[move.entry.index] = _rewritten(move.entry, move.to_line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="report, change nothing")
    action.add_argument("--write", action="store_true", help="apply every unambiguous move")
    parser.add_argument("--only", default=None, help="restrict to one source path")
    args = parser.parse_args(argv)
    if not args.check and not args.write:
        args.check = True

    moved = 0
    refused = 0
    for path, parse in ALLOWLISTS.items():
        text = path.read_text(encoding="utf-8")
        moves, refusals = plan_file(parse(text), args.only, REPO)
        for move in moves:
            print(f"  {move.entry.path}:{move.entry.line} -> {move.to_line}")
        for refusal in refusals:
            print(
                f"REFUSED {refusal.entry.path}:{refusal.entry.line} "
                f"{refusal.entry.before} -> {refusal.entry.after}\n"
                f"  {refusal.detail}\n"
                f"  the argument reads: {refusal.entry.reason[:100]}",
                file=sys.stderr,
            )
        moved += len(moves)
        refused += len(refusals)
        if args.write and moves:
            path.write_text(apply_moves(text, moves), encoding="utf-8", newline="")

    verb = "moved" if args.write else "pending"
    print(f"\n{moved} {verb}, {refused} refused")
    if refused:
        return 1
    return 1 if (args.check and moved) else 0


if __name__ == "__main__":
    raise SystemExit(main())
