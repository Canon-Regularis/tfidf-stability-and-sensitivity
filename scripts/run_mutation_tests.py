#!/usr/bin/env python3
"""Change the code on purpose and see whether the suite notices.

`check_test_vacuity.py` proves a test *can* fail. It cannot prove a test fails
for the right reason: an assertion that compares a value against itself, or one
whose tolerance swallows the effect it claims to measure, passes that check and
tests nothing. Mutation is the only mechanical answer. Break the source, rerun
the tests, and any mutant that survives marks behaviour nothing asserts.

Why this rather than mutmut. The operators below are chosen for what this project
actually gets wrong, which is not the generic set. Its bugs live in comparison
boundaries (`tie_cliques` turns on `<=` against `<`, and `tie_chains` on `>`),
in constants that are exactly 0, 1 or 2, and in the direction of a subtraction
that is asserted bit-for-bit. A generic run spends most of its budget on string
and container mutations that nothing here depends on. Being narrow also makes it
fast enough to finish, which a whole-suite campaign is not: the full suite is
three minutes, so a single-module run scoped to that module's own test file is
the difference between twenty minutes and a week.

Safety. The mutant has to be the module the tests import, and `conftest.py` puts
the working tree's `src/` on `sys.path`, so the file is edited in place. The
original bytes are written to a sibling backup first, restored in a `finally`,
and verified by digest afterwards. A backup left behind means a previous run was
killed mid-mutant: the tool refuses to start until `--restore` clears it, rather
than treating a mutated file as pristine.

Usage::

    python scripts/run_mutation_tests.py src/tfidf_stability/ranking/tie_groups.py \\
        --tests tests/test_tie_groups_tau.py
    python scripts/run_mutation_tests.py --restore src/.../tie_groups.py

Exits non-zero if any mutant survives.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKUP_SUFFIX = ".mutation-backup"

#: Comparison flips. The pairs that matter are the boundary ones: `<=` against
#: `<` decides whether a score exactly `tau` away joins a clique, which is the
#: only place tie groups are interesting.
_COMPARE = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

#: Arithmetic swaps. Addition against subtraction catches an off-by-one in an
#: index and a sign error in a margin, both of which this code does by hand.
_BINOP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}

#: Constants worth perturbing. Anything else (a digest length, a display width)
#: is not load-bearing, and mutating it produces a survivor that means nothing.
#: Keyed by (type, value): 0 and 0.0 are equal and hash alike, so a plain dict
#: would silently collapse the int and float entries into one.
_CONSTANTS: dict[tuple[type, object], object] = {
    (int, 0): 1,
    (int, 1): 0,
    (int, 2): 1,
    (float, 0.0): 1.0,
    (float, 1.0): 0.0,
    (float, 0.5): 1.0,
}


@dataclass(frozen=True, slots=True)
class Mutant:
    line: int
    col: int
    kind: str
    before: str
    after: str

    def describe(self) -> str:
        return f"line {self.line:>4}  {self.kind:<10} {self.before} -> {self.after}"


class _Mutator(ast.NodeTransformer):
    """Applies exactly one mutation, identified by its index in document order."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.seen = -1
        self.applied: Mutant | None = None

    def _hit(self, node: ast.AST, kind: str, before: str, after: str) -> bool:
        self.seen += 1
        if self.seen != self.target:
            return False
        self.applied = Mutant(
            getattr(node, "lineno", 0), getattr(node, "col_offset", 0), kind, before, after
        )
        return True

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            replacement = _COMPARE.get(type(op))
            if replacement is None:
                continue
            if self._hit(node, "compare", type(op).__name__, replacement.__name__):
                node.ops[i] = replacement()
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        replacement = _BINOP.get(type(node.op))
        if replacement is not None and self._hit(
            node, "binop", type(node.op).__name__, replacement.__name__
        ):
            node.op = replacement()
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        swap = ast.Or if isinstance(node.op, ast.And) else ast.And
        if self._hit(node, "boolop", type(node.op).__name__, swap.__name__):
            node.op = swap()
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if isinstance(value, bool):
            if self._hit(node, "constant", repr(value), repr(not value)):
                return ast.copy_location(ast.Constant(value=not value), node)
            return node
        key = (type(value), value)
        if key in _CONSTANTS:
            replacement = _CONSTANTS[key]
            if self._hit(node, "constant", repr(value), repr(replacement)):
                return ast.copy_location(ast.Constant(value=replacement), node)
        return node


def _count_sites(source: str) -> int:
    mutator = _Mutator(target=-1)
    mutator.visit(ast.parse(source))
    return mutator.seen + 1


def _apply(source: str, index: int) -> tuple[str, Mutant] | None:
    tree = ast.parse(source)
    mutator = _Mutator(target=index)
    mutated = mutator.visit(tree)
    if mutator.applied is None:
        return None
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated), mutator.applied


def _run_tests(tests: list[str], timeout: float) -> bool:
    """True if the suite passes, meaning the mutant survived."""
    try:
        finished = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *tests,
                "-q",
                "-x",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            cwd=REPO,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False  # a hang is a detection: the mutant changed behaviour
    return finished.returncode == 0


def _campaign(
    source_path: Path, original: str, limit: int, tests: list[str], timeout: float
) -> tuple[int, int, list[Mutant]]:
    """Run every mutant in turn. The caller owns the backup and the restore."""
    survivors: list[Mutant] = []
    killed = skipped = 0
    baseline = ast.unparse(ast.parse(original))
    for index in range(limit):
        outcome = _apply(original, index)
        if outcome is None:
            skipped += 1
            continue
        mutated_source, mutant = outcome
        if mutated_source == baseline:
            skipped += 1  # the mutation left the tree unchanged
            continue
        source_path.write_text(mutated_source, encoding="utf-8")
        if _run_tests(tests, timeout):
            survivors.append(mutant)
            print(f"  SURVIVED  {mutant.describe()}")
        else:
            killed += 1
    return killed, skipped, survivors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="module to mutate")
    parser.add_argument(
        "--tests", nargs="+", default=None, help="pytest targets (default: whole suite)"
    )
    parser.add_argument("--max-mutants", type=int, default=0, help="0 = every site")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per mutant")
    parser.add_argument("--restore", action="store_true", help="recover from a killed run and exit")
    args = parser.parse_args()

    source_path = (REPO / args.source).resolve() if not args.source.is_absolute() else args.source
    backup_path = source_path.with_suffix(source_path.suffix + BACKUP_SUFFIX)

    if args.restore:
        if not backup_path.exists():
            print(f"no backup at {backup_path.name}; nothing to restore")
            return 0
        source_path.write_bytes(backup_path.read_bytes())
        backup_path.unlink()
        print(f"restored {source_path.name}")
        return 0

    if backup_path.exists():
        print(
            f"{backup_path.name} exists, so a previous run was killed and "
            f"{source_path.name} may still hold a mutant.\n"
            f"Recover with: python scripts/run_mutation_tests.py --restore {args.source}",
            file=sys.stderr,
        )
        return 2

    original = source_path.read_text(encoding="utf-8")
    original_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    tests = args.tests or ["tests/"]

    total = _count_sites(original)
    limit = min(total, args.max_mutants) if args.max_mutants else total
    print(f"{source_path.name}: {total} mutable sites, running {limit}")
    print(f"tests: {' '.join(tests)}\n")

    # The unmutated suite must pass, or every mutant "survives" for the wrong
    # reason and the whole run is meaningless.
    if not _run_tests(tests, args.timeout):
        print("the suite fails before any mutation; fix that first", file=sys.stderr)
        return 2

    backup_path.write_bytes(source_path.read_bytes())
    started = time.monotonic()
    try:
        killed, skipped, survivors = _campaign(source_path, original, limit, tests, args.timeout)
    finally:
        # Restoration is unconditional, and verified: a mutant left in the working
        # tree would be a far worse outcome than a failed run.
        source_path.write_bytes(backup_path.read_bytes())
        backup_path.unlink()
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != original_digest:
            raise RuntimeError(f"restore failed for {source_path}")

    elapsed = time.monotonic() - started
    scored = killed + len(survivors)
    score = 100.0 * killed / scored if scored else 100.0
    print(f"\n{killed} killed, {len(survivors)} survived, {skipped} skipped in {elapsed:.0f}s")
    print(f"mutation score: {score:.1f}%")
    if survivors:
        print("\nEach survivor is behaviour no test asserts:", file=sys.stderr)
        for mutant in survivors:
            print(
                f"  {source_path.relative_to(REPO)}:{mutant.line}: {mutant.describe()}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
