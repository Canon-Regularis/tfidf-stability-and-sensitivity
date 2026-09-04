#!/usr/bin/env python3
"""Mutation testing for the C++ core.

``run_mutation_tests.py`` parses Python with ``ast``. The C++ half of this
project -- the half asserted to be bit-identical to the reference -- had never
been asked whether its assertions can fail, and when it was, the answer was that
several could not. The pairwise summation tree was pinned only by "error no
worse than naive", which any tree satisfies; ``is_canonical`` was probed at an
index far enough out of range that an off-by-one bound still rejected it; and
the whole attribute-rank loop could be deleted from ``build_keys`` with the
entire native suite still passing.

There is no C++ AST available here, so mutation is lexical. A scanner walks the
file tracking whether it is inside a line comment, a block comment, a string or
a character literal, and skips preprocessor lines, so operators inside those are
never touched. Every candidate is a single-token replacement.

A mutant that fails to compile is ``stillborn`` and counted neither way: that is
a defect of lexical mutation, not evidence about the tests. A mutant that
compiles and still passes ctest is a ``survivor``, and every survivor must be
argued in ``configs/equivalent_mutants_cpp.txt`` or this exits non-zero.

Usage::

    python scripts/run_cpp_mutation_tests.py cpp/include/tfidf/core/reduction.hpp
    python scripts/run_cpp_mutation_tests.py <header> --build build-tests --json out.json

The build directory must already be configured and its baseline must be green:
a tree whose tests already fail would score every mutant as killed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EQUIVALENTS = REPO / "configs" / "equivalent_mutants_cpp.txt"

#: Ordered longest-first, so `<=` is matched before `<`.
OPERATORS: list[tuple[str, str]] = [
    ("<=", "<"),
    (">=", ">"),
    ("==", "!="),
    ("!=", "=="),
    ("&&", "||"),
    ("||", "&&"),
    ("<", "<="),
    (">", ">="),
    ("+", "-"),
    ("-", "+"),
    ("*", "/"),
    ("/", "*"),
]

WORDS: list[tuple[str, str]] = [("true", "false"), ("false", "true")]

#: Only these literals. Mutating an arbitrary constant mostly yields stillborn
#: or uninteresting mutants.
NUMBERS: list[tuple[str, str]] = [("0", "1"), ("1", "0"), ("2", "3")]

_IDENT = re.compile(r"[A-Za-z_0-9]")


def fingerprint(line: str) -> str:
    """The stamp recorded beside an allowlist entry.

    Whitespace-normalised, so reindenting a line does not invalidate an argument
    about it, while changing the expression does.
    """
    return hashlib.sha256(" ".join(line.split()).encode("utf-8")).hexdigest()[:8]


def scan(text: str) -> list[tuple[int, str, str]]:
    """Candidate ``(offset, before, after)`` triples.

    Comments, string and character literals and preprocessor lines are skipped,
    so a mutation never lands somewhere the compiler does not read as code.
    """
    out: list[tuple[int, str, str]] = []
    i, n = 0, len(text)
    line_start = True

    while i < n:
        ch = text[i]

        if ch == "\n":
            line_start = True
            i += 1
            continue
        if line_start and ch in " \t":
            i += 1
            continue
        if line_start and ch == "#":  # preprocessor
            while i < n and text[i] != "\n":
                if text[i] == "\\" and i + 1 < n and text[i + 1] == "\n":
                    i += 1
                i += 1
            continue
        line_start = False

        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue

        # `->`, `<<` and `>>` are single tokens; splitting them is a syntax
        # error rather than a mutation.
        if text.startswith("->", i) or text.startswith("<<", i) or text.startswith(">>", i):
            i += 2
            continue
        if text[i] in "+-*/<>=!&|" and i + 1 < n and text[i + 1] == "=" and text[i] not in "<>=!":
            i += 2  # compound assignment
            continue

        for before, after in OPERATORS:
            if text.startswith(before, i):
                out.append((i, before, after))
                i += len(before)
                break
        else:
            if _IDENT.match(ch):
                j = i
                while j < n and _IDENT.match(text[j]):
                    j += 1
                word = text[i:j]
                prev = text[i - 1] if i else " "
                nxt = text[j] if j < n else " "
                if not _IDENT.match(prev) and prev != ".":
                    for before, after in WORDS + NUMBERS:
                        if word == before and not _IDENT.match(nxt) and nxt != ".":
                            out.append((i, before, after))
                            break
                i = j
            else:
                i += 1
    return out


def load_equivalents(module: Path) -> dict[tuple[int, int, str, str], str]:
    """Argued equivalences for one header, keyed by ``(line, column, before, after)``.

    The column is part of the key because a line can carry two candidates that
    read the same and are equivalent for different reasons: in the fsum
    correction, one `<` compares the remainder against zero and the other
    compares a partial, and the arguments are not interchangeable.
    """
    if not EQUIVALENTS.exists():
        return {}
    wanted = module.as_posix()
    claims: dict[tuple[int, int, str, str], str] = {}
    for raw in EQUIVALENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)
        body, reason = line[0].strip(), (line[1].strip() if len(line) > 1 else "")
        if not body:
            continue
        fields = body.split()
        if len(fields) < 5 or fields[0] != wanted or fields[3] != "->":
            continue
        where, _, column = fields[1].partition(":")
        if not column:
            continue  # an entry without a column cannot say which token it means
        # Emptiness is tested AFTER the stamp is stripped. Testing the raw text
        # would let an entry whose whole reason is its own fingerprint through,
        # carrying an empty argument -- the suppression this file refuses.
        argument = without_stamp(reason)
        if not argument:
            continue
        claims[(int(where), int(column), fields[2], fields[4])] = argument
    return claims


def without_stamp(reason: str) -> str:
    """Drop a leading ``src=<8 hex>`` marker from an allowlist reason."""
    head, _, rest = reason.partition(" ")
    if head.startswith("src=") and len(head) == len("src=") + 8:
        return rest.strip()
    return reason.strip()


def build_and_test(build: str, target: str, test_timeout: int) -> str:
    """``killed`` | ``survived`` | ``stillborn``.

    A mutant can make the suite loop forever -- dropping the increment that ends
    a merge, or relaxing a bound the loop counts against -- and three campaigns
    died on exactly that before this handled it. `ctest --timeout` makes ctest
    kill the test itself and report failure, which is both the right verdict (a
    suite that does not terminate has not passed) and the only way to avoid
    leaving the test binary running, holding a lock on the file the next build
    needs to relink.

    A build that times out is `stillborn` instead: nothing was learned about the
    tests, and treating it as a kill would inflate the score with infrastructure.
    """
    try:
        built = subprocess.run(
            ["cmake", "--build", build, "--target", target],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "stillborn"
    if built.returncode != 0:
        return "stillborn"
    try:
        tested = subprocess.run(
            [
                "ctest",
                "--test-dir",
                build,
                "--no-tests=error",
                "-Q",
                "--timeout",
                str(test_timeout),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=test_timeout * 4 + 120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # ctest itself did not return, so the mutant is detected either way.
        return "killed"
    return "killed" if tested.returncode != 0 else "survived"


def campaign(
    relative: str, build: str, target: str, limit: int | None, test_timeout: int
) -> dict[str, object]:
    path = REPO / relative
    original = path.read_text(encoding="utf-8", newline="")
    lines = original.split("\n")

    line_of: dict[int, int] = {}
    col_of: dict[int, int] = {}
    number, start = 1, 0
    for offset, ch in enumerate(original):
        line_of[offset] = number
        col_of[offset] = offset - start + 1
        if ch == "\n":
            number += 1
            start = offset + 1

    candidates = scan(original)
    if limit:
        candidates = candidates[:limit]

    claimed = load_equivalents(Path(relative))
    counts = {"killed": 0, "survived": 0, "stillborn": 0}
    survivors: list[dict[str, object]] = []
    started = time.monotonic()

    print(f"{relative}: {len(candidates)} candidates", flush=True)
    try:
        for k, (offset, before, after) in enumerate(candidates, 1):
            path.write_text(
                original[:offset] + after + original[offset + len(before) :],
                encoding="utf-8",
                newline="",
            )
            verdict = build_and_test(build, target, test_timeout)
            counts[verdict] += 1
            if verdict == "survived":
                where, column = line_of[offset], col_of[offset]
                source = lines[where - 1].strip()
                key = (where, column, before, after)
                survivors.append(
                    {
                        "line": where,
                        "column": column,
                        "before": before,
                        "after": after,
                        "source": source,
                        "stamp": fingerprint(lines[where - 1]),
                        "claimed": key in claimed,
                        "reason": claimed.get(key, ""),
                    }
                )
                mark = "claimed" if key in claimed else "SURVIVED"
                print(
                    f"  {mark} {relative}:{where}:{column}  {before} -> {after}   {source[:70]}",
                    flush=True,
                )
            if k % 25 == 0:
                print(f"  {k}/{len(candidates)}  {counts}", flush=True)
    finally:
        path.write_text(original, encoding="utf-8", newline="")
        subprocess.run(
            ["cmake", "--build", build, "--target", target],
            cwd=REPO,
            capture_output=True,
            timeout=1800,
            check=False,
        )

    tried = counts["killed"] + counts["survived"]
    return {
        "file": relative,
        "counts": counts,
        "score": round(100 * counts["killed"] / tried, 1) if tried else None,
        "survivors": survivors,
        "unclaimed": [s for s in survivors if not s["claimed"]],
        "stale": [
            {"line": line, "column": column, "before": before, "after": after}
            for (line, column, before, after) in claimed
            if not any(
                s["line"] == line
                and s["column"] == column
                and s["before"] == before
                and s["after"] == after
                for s in survivors
            )
        ],
        "seconds": round(time.monotonic() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header", help="path relative to the repository root")
    parser.add_argument("--build", default="build-tests", help="configured build directory")
    parser.add_argument("--target", default="tfidf_tests", help="test target to rebuild")
    parser.add_argument("--limit", type=int, default=None, help="stop after N mutants")
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=180,
        help="seconds before ctest kills a test; a mutant that loops forever "
        "is detected by this rather than by hanging the campaign",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full record here")
    args = parser.parse_args(argv)

    record = campaign(args.header, args.build, args.target, args.limit, args.test_timeout)
    if args.json:
        args.json.write_text(json.dumps(record, indent=2), encoding="utf-8")

    counts = record["counts"]
    print(
        f"\n{record['file']}: killed {counts['killed']}, survived {counts['survived']}, "
        f"stillborn {counts['stillborn']} ({record['seconds']}s)"
    )

    failed = False
    for survivor in record["unclaimed"]:
        print(
            f"UNCLAIMED SURVIVOR {record['file']}:{survivor['line']}  "
            f"{survivor['before']} -> {survivor['after']}  {survivor['source']}\n"
            f"  Kill it with a test, or argue it in {EQUIVALENTS.name} as:\n"
            f"  {record['file']}  {survivor['line']}  {survivor['before']} -> "
            f"{survivor['after']}  # src={survivor['stamp']} <why nothing can observe it>",
            file=sys.stderr,
        )
        failed = True
    # An entry that matched no survivor is the other failure mode: the mutant is
    # killed now, or the line moved, and either way the claim needs re-reading
    # rather than leaving to rot.
    for entry in record["stale"]:
        print(
            f"STALE CLAIM {record['file']}:{entry['line']}:{entry['column']}  "
            f"{entry['before']} -> {entry['after']} matched no survivor",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
