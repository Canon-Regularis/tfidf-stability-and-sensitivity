"""Mutate one module in a sandbox copy and report mutants the suite misses.

Operators flip comparisons, swap arithmetic and boolean operators and perturb
constants. The run mutates a temporary copy, and exits non-zero on a survivor.
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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tooling.allowlist import claims_python, parse_python, verdict, without_stamp

__all__ = ["without_stamp"]

REPO = Path(__file__).resolve().parents[1]

#: Paths copied into the sandbox so the suite runs there as a working checkout.
#: Every top-level path a test file reads must be listed, or the sandbox
#: baseline fails before a mutant is applied.
#: Rationale: docs/engineering_notes.md#sandbox-contents
_SANDBOX_CONTENTS = (
    "src",
    "tests",
    "configs",
    "cpp",
    "data",
    "docs",
    "examples",
    "scripts",
    # The suite imports `tooling.allowlist`, so a sandbox without it fails
    # the baseline before a single mutant is applied.
    "tooling",
    "pyproject.toml",
    "CITATION.cff",
    "README.md",
    # `check_versions.py` reads the project version from four files and
    # `tests/test_repository_gates.py` runs it, so all four must be present or
    # the sandbox baseline fails before a single mutant is applied.
    "CMakeLists.txt",
    # `check_docs.py` checks that every preset is documented;
    # `test_repository_gates.py` compares the two requirement files; and
    # `test_cpp_mutation_gate.py` reads the nightly's own matrix and the preset
    # that job configures.
    "CMakePresets.json",
    "requirements-dev.txt",
    ".github",
    # `test_render_summary.py` reads the committed reports, and every driver
    # defaults its output here. `test_mutation_gate.py` asserts this list covers
    # every top-level path the suite names, which is what keeps it complete.
    "reports",
    # `test_repository_gates.py` compares the hook patterns against every
    # directory holding a MANIFEST.sha256.
    ".pre-commit-config.yaml",
    # Write-only, and absent unless a C++ campaign was interrupted. The sandbox
    # check over-approximates, and `copy2` is skipped for a path that does not
    # exist.
    ".mutation-cpp-interrupted.json",
)

#: Comparison flips. The boundary pairs matter most: `<=` against `<` decides
#: whether a score exactly `tau` away joins a clique.
#: Rationale: docs/engineering_notes.md#mutation-operator-set
_COMPARE = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

#: Arithmetic swaps. Addition against subtraction catches an off-by-one in an
#: index and a sign error in a margin, both computed by hand in the package.
_BINOP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}

#: Constants to perturb. Other values (a digest length, a display width) are
#: not load-bearing, and mutating them produces survivors that mean nothing.
#: Keyed by (type, value): 0 and 0.0 are equal and hash alike, so a dict keyed
#: by value alone collapses the int and float entries into one.
_CONSTANTS: dict[tuple[type, object], int | float] = {
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
    """Create a throwaway checkout of the paths in ``_SANDBOX_CONTENTS``.

    The campaign mutates the copy, never the working tree.
    Rationale: docs/engineering_notes.md#mutation-sandbox-copy
    """
    sandbox = Path(tempfile.mkdtemp(prefix="tfidf-mutation-"))
    # The compiled extension is copied with everything else. Without it,
    # native_available() is false in the sandbox, the differential tests fail
    # to collect and the baseline run fails.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".hypothesis")
    for name in _SANDBOX_CONTENTS:
        source = REPO / name
        if source.is_dir():
            shutil.copytree(source, sandbox / name, ignore=ignore)
        elif source.exists():
            shutil.copy2(source, sandbox / name)

    # The editable install's `.pth` puts the real `src/` on `sys.path` and adds
    # an import finder for it. This sitecustomize removes both at interpreter
    # startup and prepends the sandbox's own `src/`, so the mutated copy is what
    # the tests import. Rationale: docs/engineering_notes.md#sandbox-sitecustomize
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
                "import os",
                f"_real = {str(REPO / 'src')!r}",
                f"_mine = {str(sandbox / 'src')!r}",
                # normcase: the `.pth` entry and this string come from different
                # tools, and on Windows they can differ in drive case or separator
                # while naming one directory. Removing the real tree, rather than
                # placing it behind, means a later prepend cannot reorder them.
                "_key = lambda p: os.path.normcase(os.path.abspath(p))",
                "sys.path[:] = [p for p in sys.path if _key(p) != _key(_real)]",
                "sys.path.insert(0, _mine)",
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


#: Claims that a surviving mutant is behaviourally indistinguishable, with the
#: reason. The file's own header carries the format and states why an entry
#: that matches nothing is a failure.
_EQUIVALENTS = REPO / "configs" / "equivalent_mutants.txt"

_Key = tuple[int, str, str, str]


def _load_equivalents(module: Path) -> dict[_Key, str]:
    """Documented equivalent mutants for one module, keyed by what was mutated.

    Reads ``_EQUIVALENTS`` at call time, so a test pointing the module at another
    file gets that file. The format itself lives in ``tooling.allowlist``.
    """
    if not _EQUIVALENTS.exists():
        return {}
    return claims_python(parse_python(_EQUIVALENTS.read_text(encoding="utf-8")), module)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    print(f"\n{killed} killed, {len(survivors)} survived, {skipped} skipped in {elapsed:.0f}s")

    if scored == 0:
        # A campaign that mutated nothing (a module with no mutable sites, or
        # every mutant skipped) must not report a perfect score and exit 0, so
        # it fails instead.
        print(
            f"::error::no mutant was scored for {relative}: {skipped} skipped. "
            f"This gate proved nothing.",
            file=sys.stderr,
        )
        return 2

    score = 100.0 * killed / scored
    print(f"mutation score: {score:.1f}%")
    if survivors:
        print("\nEach survivor is behaviour no test asserts:", file=sys.stderr)
        for mutant in survivors:
            print(f"  {relative}:{mutant.line}: {mutant.describe()}", file=sys.stderr)

    return _verdict(relative, survivors, partial=limit < total)


def _verdict(relative: Path, survivors: Sequence[Mutant], *, partial: bool = False) -> int:
    """Exit code once documented equivalents are accounted for.

    An unexplained survivor fails; so does a claim matching no survivor, since a
    stale claim is a blanket suppression. A ``partial`` run withholds the second failure.
    """
    claims = _load_equivalents(relative)
    if not claims:
        return 1 if survivors else 0

    # `allowlist.verdict` counts rather than collects, because two mutable sites
    # on one line can produce the same key -- `@dataclass(frozen=True,
    # slots=True)` gives two identical `constant True -> False` mutants -- and
    # reporting "1 documented" against two survivors would read as a gap.
    split = verdict([(m.line, m.kind, m.before, m.after) for m in survivors], claims)
    unexplained, stale = split.unclaimed, split.stale

    print(f"\n{split.covered} of {len(survivors)} survivor(s) documented as equivalent")

    if unexplained:
        print(f"\nUndocumented survivors in {relative}:", file=sys.stderr)
        for line, kind, before, after in unexplained:
            print(f"  {relative}:{line}: {kind} {before} -> {after}", file=sys.stderr)

    if partial and stale:
        # A claim the run never reached is unmatched rather than stale. The
        # C++ campaign withholds the same failure for the same reason.
        print(
            f"\n{len(stale)} unmatched claim(s) not reported: --max-mutants "
            f"stopped the run, and a truncated run cannot tell stale from unreached.",
            file=sys.stderr,
        )
    elif stale:
        print(f"\nStale entries in {_EQUIVALENTS.name}:", file=sys.stderr)
        for line, kind, before, after in stale:
            print(
                f"  {relative}:{line}: {kind} {before} -> {after} no longer survives",
                file=sys.stderr,
            )

    return 1 if unexplained or (stale and not partial) else 0


if __name__ == "__main__":
    raise SystemExit(main())
