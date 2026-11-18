"""Lexical mutation testing for the C++ core, scored by ctest.

Candidates are single-token replacements outside comments, literals and
preprocessor lines. Rationale: docs/engineering_notes.md#cpp-mutation-score-scope
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

from tooling.allowlist import (
    claims_cpp,
    fingerprint,
    parse_cpp,
    without_stamp,
)
from tooling.allowlist import verdict as claim_verdict

__all__ = ["fingerprint", "without_stamp"]

REPO = Path(__file__).resolve().parents[1]

EQUIVALENTS = REPO / "configs" / "equivalent_mutants_cpp.txt"

#: Holds the pristine header while a campaign runs. The restore in `campaign`
#: runs in a `finally`, which covers an exception and Ctrl-C but not SIGKILL,
#: so a run that dies leaves this file and the next run repairs from it. One
#: unlink ends the recovery state.
INTERRUPTED = REPO / ".mutation-cpp-interrupted.json"

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
    """Argued equivalences for one header, keyed by position and token pair.

    ``EQUIVALENTS`` is read at call time, so a test that points the module at
    another file gets that file. The format lives in ``tooling.allowlist``.
    """
    if not EQUIVALENTS.exists():
        return {}
    return claims_cpp(parse_cpp(EQUIVALENTS.read_text(encoding="utf-8")), module)


def build_and_test(build: str, target: str, test_timeout: int) -> str:
    """Build the target, run ctest, and return killed, survived or stillborn.

    A build failure or timeout is ``stillborn``; a ctest that does not return
    within the backstop is ``killed``. Rationale: docs/engineering_notes.md#ctest-timeout-bounds
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
            # Above the project's 300s ctest property, so ctest does the
            # killing and no orphaned binary holds the exe lock.
            timeout=max(test_timeout, 300) * 2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # ctest itself did not return, so the mutant is detected either way.
        return "killed"
    return "killed" if tested.returncode != 0 else "survived"


def write_canonical_json(path: Path, payload: object) -> None:
    """Write JSON with sorted keys, LF endings and a trailing newline.

    `tooling` does not import `tfidf_stability.utils.io`, where the library's
    `canonical_json` lives, so the same form is spelled once here instead.
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=""
    )


def restore_interrupted() -> str | None:
    """Repair a header a previous run left mutated, returning its path.

    Runs before the baseline check, because a leftover mutant fails that check
    for a reason that reads as unrelated to the mutation.
    """
    if not INTERRUPTED.exists():
        return None
    saved = json.loads(INTERRUPTED.read_text(encoding="utf-8"))
    relative = str(saved["path"])
    (REPO / relative).write_text(saved["original"], encoding="utf-8", newline="")
    INTERRUPTED.unlink()
    return relative


def verify_baseline(build: str, target: str, test_timeout: int) -> None:
    """Refuse to score anything against a tree whose tests already fail.

    A red baseline scores every mutant as killed, because `killed` means only
    that ctest returned non-zero; an unmutated tree that passes is `survived`.
    """
    verdict = build_and_test(build, target, test_timeout)
    if verdict != "survived":
        raise SystemExit(
            f"baseline is not green: unmutated, {build} reports '{verdict}', so every "
            f"mutant would score as killed and the campaign would report a clean sweep.\n"
            f"A header left mutated by an interrupted run is repaired before this check, "
            f"so the cause is in the tree or the build rather than a stale mutant."
        )


class Survivor(TypedDict):
    """A mutant that compiled and that ctest did not catch.

    ``claimed`` and ``reason`` record whether the allowlist argues this one is
    equivalent, so the verdict below reads them rather than the file again.
    """

    line: int
    column: int
    before: str
    after: str
    source: str
    stamp: str
    claimed: bool
    reason: str


class StaleClaim(TypedDict):
    """An allowlist entry that matched no survivor in a complete run."""

    line: int
    column: int
    before: str
    after: str


class Record(TypedDict):
    """One campaign's full result, as written to ``--json``.

    ``score`` is ``None`` when nothing was scored, so a run that mutated only
    stillborn candidates cannot report a percentage it did not measure.
    """

    partial: bool
    file: str
    counts: dict[str, int]
    score: float | None
    survivors: list[Survivor]
    unclaimed: list[Survivor]
    stale: list[StaleClaim]
    seconds: float


def campaign(
    relative: str, build: str, target: str, limit: int | None, test_timeout: int
) -> Record:
    repaired = restore_interrupted()
    if repaired is not None:
        print(f"repaired {repaired}, left mutated by an interrupted run", flush=True)

    verify_baseline(build, target, test_timeout)
    path = REPO / relative
    with path.open(encoding="utf-8", newline="") as handle:
        original = handle.read()
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
    if limit is not None:
        # `limit == 0` is a cap of zero candidates, not "no limit", so the test
        # is against None. A negative cap would slice a candidate off the end
        # while still marking the run partial, which suppresses stale-claim
        # reporting, so it is refused.
        if limit < 0:
            raise SystemExit(f"--limit must be non-negative, got {limit}")
        candidates = candidates[:limit]

    claimed = load_equivalents(Path(relative))
    counts = {"killed": 0, "survived": 0, "stillborn": 0}
    survivors: list[Survivor] = []
    started = time.monotonic()

    print(f"{relative}: {len(candidates)} candidates", flush=True)
    write_canonical_json(INTERRUPTED, {"original": original, "path": relative})
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
        INTERRUPTED.unlink(missing_ok=True)
        subprocess.run(
            ["cmake", "--build", build, "--target", target],
            cwd=REPO,
            capture_output=True,
            timeout=1800,
            check=False,
        )

    tried = counts["killed"] + counts["survived"]
    # Shared with the Python campaign, which decides the same two directions.
    split = claim_verdict(
        [(s["line"], s["column"], s["before"], s["after"]) for s in survivors], claimed
    )
    return {
        # A truncated run has not seen every candidate, so a claim it did not
        # reach is unmatched rather than stale. Reporting those would fail a
        # smoke run for valid entries.
        "partial": limit is not None,
        "file": relative,
        "counts": counts,
        "score": round(100 * counts["killed"] / tried, 1) if tried else None,
        "survivors": survivors,
        "unclaimed": [s for s in survivors if not s["claimed"]],
        "stale": [
            {"line": line, "column": column, "before": before, "after": after}
            for line, column, before, after in split.stale
        ],
        "seconds": round(time.monotonic() - started, 1),
    }


def suggested_entry(relative: str, survivor: Survivor) -> str:
    """The allowlist line to paste if the survivor is equivalent.

    Built here so a test can paste it back through :func:`load_equivalents`.
    The column is required: an entry without one is ignored.
    """
    return (
        f"{relative}  {survivor['line']}:{survivor['column']}  "
        f"{survivor['before']} -> {survivor['after']}  "
        f"# src={survivor['stamp']} <why nothing can observe it>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("header", help="path relative to the repository root")
    parser.add_argument("--build", default="build-tests", help="configured build directory")
    parser.add_argument("--target", default="tfidf_tests", help="test target to rebuild")
    parser.add_argument("--limit", type=int, default=None, help="stop after N mutants")
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=180,
        help="per-test timeout passed to ctest. This project sets a TIMEOUT "
        "property of 300s, which CMake does not let --timeout override, so "
        "here it bounds only the outer backstop",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full record here")
    args = parser.parse_args(argv)

    record = campaign(args.header, args.build, args.target, args.limit, args.test_timeout)
    if args.json:
        write_canonical_json(args.json, record)

    counts = record["counts"]
    print(
        f"\n{record['file']}: killed {counts['killed']}, survived {counts['survived']}, "
        f"stillborn {counts['stillborn']} ({record['seconds']}s)"
    )

    failed = False
    for survivor in record["unclaimed"]:
        print(
            f"UNCLAIMED SURVIVOR {record['file']}:{survivor['line']}:{survivor['column']}  "
            f"{survivor['before']} -> {survivor['after']}  {survivor['source']}\n"
            f"  Kill it with a test, or argue it in {EQUIVALENTS.name} as:\n"
            f"  {suggested_entry(str(record['file']), survivor)}",
            file=sys.stderr,
        )
        failed = True
    # An entry that matched no survivor is the other failure mode: the mutant
    # is killed now, or the line moved, and either way the claim needs
    # re-reading. Only meaningful over a complete run.
    if record["partial"]:
        print(
            f"--limit was set, so {len(record['stale'])} unmatched claims are not "
            f"reported: a truncated run cannot tell stale from unreached.",
            file=sys.stderr,
        )
    for entry in [] if record["partial"] else record["stale"]:
        print(
            f"STALE CLAIM {record['file']}:{entry['line']}:{entry['column']}  "
            f"{entry['before']} -> {entry['after']} matched no survivor",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
