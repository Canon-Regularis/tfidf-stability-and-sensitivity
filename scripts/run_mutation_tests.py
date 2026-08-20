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

The working tree is never modified. An earlier version edited the module in
place, because `tests/conftest.py` puts `src/` on `sys.path` and the tests have
to import the mutant. It guarded that with a backup and a `finally`, which is not
enough: `finally` does not run when the process is killed, and on Windows
`terminate()` maps to `TerminateProcess`, so neither a SIGTERM handler nor an
`atexit` hook fires either. Measured, not assumed: killing a live campaign left
the module mutated and the backup behind. One such run was then committed, taking
340 lines of source down to `ast.unparse` output with a binary search bound
flipped the wrong way, which turned a 1.4-second test file into a hang.

So the campaign runs against a copy instead. `conftest.py` computes `src/`
relative to its own location, so a tree containing `src/` and `tests/` is
self-contained: point pytest at the copy and the copy's modules are what get
imported. Killing the run now loses a temporary directory and nothing else.

Usage::

    python scripts/run_mutation_tests.py src/tfidf_stability/ranking/tie_groups.py \\
        --tests tests/test_tie_groups_tau.py

Exits non-zero if any mutant survives.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: What a run needs to be a working checkout: the package, the suite that
#: imports it, the pytest configuration that declares markers and turns warnings
#: into errors, and the assets the package loads from the repository root.
#: `data/` is not optional: preprocessing resolves the frozen stopword list as
#: `parents[3] / "data" / "assets"`, so leaving it out fails the fixture that
#: builds the normative pipeline and errors the suite before a mutant runs.
#:
#: `cpp/` is here for the same reason one level along. test_public_api_surface.py
#: checks `types.py`'s aliases against `cpp/include/tfidf/core/types.hpp`, so a
#: sandbox without the headers fails that file outright -- and it is the file
#: worth adding to a scoped campaign, because the contracts it asserts are the
#: package-wide ones (frozen dataclasses, enum serialisation) that no single
#: module's own test file covers.
#:
#: `scripts/` for the third instance of it: test_benchmark_smoke.py runs
#: scripts/benchmark.py as a subprocess, so without it that file errors and
#: the campaign exits before mutating anything -- which reads as a broken
#: runner rather than a missing directory.
_SANDBOX_CONTENTS = ("src", "tests", "configs", "cpp", "data", "scripts", "pyproject.toml")

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
        self.applied = Mutant(getattr(node, "lineno", 0), kind, before, after)
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
    mutator = _Mutator(target=index)
    mutated = mutator.visit(ast.parse(source))
    if mutator.applied is None:
        return None
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated), mutator.applied


def _make_sandbox() -> Path:
    """A throwaway checkout holding the package, the suite and the pytest config."""
    sandbox = Path(tempfile.mkdtemp(prefix="tfidf-mutation-"))
    # The compiled extension is copied with everything else. Excluding it made
    # native_available() false in the sandbox, which turned the differential
    # tests into a collection error and the baseline run into a failure.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".hypothesis")
    for name in _SANDBOX_CONTENTS:
        source = REPO / name
        if source.is_dir():
            shutil.copytree(source, sandbox / name, ignore=ignore)
        elif source.exists():
            shutil.copy2(source, sandbox / name)

    # Copying the tree is not enough on its own. This project is installed
    # editable, and scikit-build-core does that by putting a
    # ScikitBuildRedirectingFinder on sys.meta_path pointing at the real src/.
    # Meta-path finders are consulted before sys.path, so conftest.py inserting
    # the sandbox at sys.path[0] loses and the tests import the working tree.
    # Measured: replacing the sandbox module with `raise RuntimeError` still gave
    # 26 passed, which would have scored every mutant a survivor.
    #
    # sitecustomize is imported during interpreter startup, before pytest or
    # conftest, so it is the one place early enough to drop that finder.
    (sandbox / "sitecustomize.py").write_text(
        chr(10).join(
            (
                "import sys",
                "sys.meta_path[:] = [",
                "    finder",
                "    for finder in sys.meta_path",
                "    if 'Redirecting' not in type(finder).__name__",
                "    and 'editable' not in getattr(finder, '__module__', '')",
                "]",
            )
        ),
        encoding="utf-8",
    )
    return sandbox


def _run_tests(tests: list[str], timeout: float, cwd: Path) -> bool:
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
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            check=False,
            # PYTHONPATH so the sandbox's sitecustomize is found at startup.
            env={**os.environ, "PYTHONPATH": str(cwd)},
        )
    except subprocess.TimeoutExpired:
        return False  # a hang is a detection: the mutant changed behaviour
    return finished.returncode == 0


def _campaign(
    target: Path,
    original: str,
    limit: int,
    *,
    tests: list[str],
    timeout: float,
    cwd: Path,
) -> tuple[int, int, list[Mutant]]:
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
        target.write_text(mutated_source, encoding="utf-8")
        if _run_tests(tests, timeout, cwd):
            survivors.append(mutant)
            print(f"  SURVIVED  {mutant.describe()}", flush=True)
        else:
            killed += 1
    return killed, skipped, survivors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="module to mutate, relative to the repository")
    parser.add_argument(
        "--tests", nargs="+", default=None, help="pytest targets (default: whole suite)"
    )
    parser.add_argument("--max-mutants", type=int, default=0, help="0 = every site")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per mutant")
    args = parser.parse_args()

    relative = args.source.relative_to(REPO) if args.source.is_absolute() else args.source
    if not (REPO / relative).exists():
        print(f"no such module: {relative}", file=sys.stderr)
        return 2

    original = (REPO / relative).read_text(encoding="utf-8")
    tests = args.tests or ["tests/"]
    total = _count_sites(original)
    limit = min(total, args.max_mutants) if args.max_mutants else total

    sandbox = _make_sandbox()
    print(f"{relative.name}: {total} mutable sites, running {limit}")
    print(f"tests: {' '.join(tests)}")
    print(f"sandbox: {sandbox}\n", flush=True)
    try:
        target = sandbox / relative
        # A suite that fails before any mutation makes every mutant "survive" for
        # the wrong reason, so the whole run would be meaningless.
        if not _run_tests(tests, args.timeout, sandbox):
            print("the suite fails in the sandbox before any mutation", file=sys.stderr)
            return 2

        started = time.monotonic()
        killed, skipped, survivors = _campaign(
            target, original, limit, tests=tests, timeout=args.timeout, cwd=sandbox
        )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    elapsed = time.monotonic() - started
    scored = killed + len(survivors)
    score = 100.0 * killed / scored if scored else 100.0
    print(f"\n{killed} killed, {len(survivors)} survived, {skipped} skipped in {elapsed:.0f}s")
    print(f"mutation score: {score:.1f}%")
    if survivors:
        print("\nEach survivor is behaviour no test asserts:", file=sys.stderr)
        for mutant in survivors:
            print(f"  {relative}:{mutant.line}: {mutant.describe()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
