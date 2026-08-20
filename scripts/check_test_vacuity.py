#!/usr/bin/env python3
"""Find tests that pass without establishing anything.

A green suite is only evidence if the tests could have failed. This repository
already guards the run-level version of that: `ctest --no-tests=error` catches an
empty C++ suite, and determinism.yml counts executed tests because every
differential test is skipif-guarded and pytest exits 0 when all of them skip.
Both answer "did anything run". Neither answers "did what ran assert anything",
which is the same failure one level down.

The checks below are the vacuity modes that have actually appeared here. Two are
already defended by hand, in comments that say why: test_scoring_kernels.py has a
test whose name is the guard, and test_perturbation_bounds.py counts the
documents it examined because "without this the test passes vacuously whenever
every non-edited document happens to see a zero shift". A hand-written guard
protects the one test somebody thought about; this protects the rest.

What is deliberately not checked: whether an assertion is *strong*. `assert x ==
approx(y, rel=1)` is nearly content-free and no parser can tell. Mutation testing
is the tool for that question and is not wired up here.

Usage::

    python scripts/check_test_vacuity.py [--tests tests/] [-v]

Exits non-zero if any finding is reported, so it can gate CI beside
check_docs.py and check_layout.py.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: A call whose name contains one of these is treated as establishing something.
#: Deliberately generous: a false negative here costs one unexamined test, while a
#: false positive trains the reader to ignore the whole report.
_ASSERTING = ("assert", "raises", "warns", "fail", "check", "verify", "expect")

#: `pytest.raises(Exception)` passes when the code under test raises anything at
#: all, including a TypeError from a signature the test itself got wrong. Narrower
#: builtins are listed because they are the ones that catch a typo rather than the
#: behaviour: AttributeError from a renamed method, NameError from a stale import.
_TOO_BROAD = {"Exception", "BaseException", "AttributeError", "NameError", "ImportError"}

#: `pytest.raises(X)` without `match=` asserts only the type, so two guards of
#: the same class are interchangeable: swapping them, or reaching the second when
#: the test meant the first, still passes. Most of this package's guards raise the
#: same handful of project exceptions and differ only in what they say.
#:
#: Exempt where there is no message to match. `SystemExit` carries an exit code
#: rather than prose, and the tests that raise it assert `.code` instead, which is
#: stricter than any pattern would be.
_NO_MESSAGE = {"SystemExit", "KeyboardInterrupt", "StopIteration", "GeneratorExit"}


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    line: int
    test: str
    kind: str
    detail: str

    def format(self) -> str:
        rel = self.path.relative_to(REPO) if self.path.is_relative_to(REPO) else self.path
        return f"{rel}:{self.line}: {self.kind}: {self.test}: {self.detail}"


def _is_test(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")


def _call_name(node: ast.expr) -> str:
    """Dotted name of a call target, lowercased. `pytest.raises` -> 'pytest.raises'."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts)).lower()


def _establishes_something(fn: ast.AST) -> bool:
    """True if the body could fail: an assert, a raise, or a call that asserts."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert | ast.Raise):
            return True
        if isinstance(node, ast.Call) and any(w in _call_name(node.func) for w in _ASSERTING):
            return True
        # `with pytest.raises(...)` parses as a Call and is caught above, but a
        # bare `pytest.raises` passed around as a value would not be.
        if isinstance(node, ast.Attribute) and any(w in node.attr.lower() for w in _ASSERTING):
            return True
    return False


def _constant_assert(node: ast.Assert) -> str | None:
    """An assertion on a literal can never fail."""
    test = node.test
    if isinstance(test, ast.Constant):
        if test.value:
            return f"assert {test.value!r} is always true"
        return None  # `assert False` is a deliberate unreachable marker
    # `assert (a, b)` is the classic typo: a non-empty tuple is always truthy.
    if isinstance(test, ast.Tuple) and test.elts:
        return "assert on a non-empty tuple is always true (stray comma?)"
    return None


def _empty_parametrize(deco: ast.expr) -> str | None:
    if not isinstance(deco, ast.Call) or "parametrize" not in _call_name(deco.func):
        return None
    if len(deco.args) < 2:
        return None
    values = deco.args[1]
    if isinstance(values, ast.List | ast.Tuple | ast.Set) and not values.elts:
        return "parametrize with no cases collects nothing"
    return None


def _unmarked_property(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """A Hypothesis test that does not carry the marker naming it as one.

    `pyproject.toml` registers `property` and runs under `--strict-markers`, so
    the category is declared; the marker is what makes it selectable. Nightly
    runs the property tests at a raised example count and currently names the
    files by hand, which drifts the moment a `@given` lands in a file nobody
    thought to add. Enforcing the marker here means the selector can become
    `-m property` and stay correct without anyone maintaining a list.
    """
    names = {
        _call_name(d.func) if isinstance(d, ast.Call) else _call_name(d) for d in fn.decorator_list
    }
    if not any(n == "given" or n.endswith(".given") for n in names):
        return None
    if any(n.endswith("mark.property") for n in names):
        return None
    return "@given test is not marked `@pytest.mark.property`"


def _broad_raises(node: ast.Call) -> str | None:
    if "raises" not in _call_name(node.func) or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Name) and first.id in _TOO_BROAD:
        return f"raises({first.id}) passes for any error, including one the test caused"
    return None


def _unmatched_raises(node: ast.Call) -> str | None:
    """A `pytest.raises` that pins the type but not what the guard says.

    A tuple of acceptable types is exempt: that form says "any of these", which a
    single pattern cannot express, and it is used where the failure mode is
    genuinely several (a fuzzed parser reaching different layers).
    """
    if "raises" not in _call_name(node.func) or not node.args:
        return None
    if any(keyword.arg == "match" for keyword in node.keywords):
        return None

    first = node.args[0]
    if isinstance(first, ast.Tuple):
        return None
    name = first.id if isinstance(first, ast.Name) else getattr(first, "attr", "")
    if name in _NO_MESSAGE:
        return None
    return f"raises({name or '...'}) without match= asserts the type but not the message"


#: Checks that run against every call in a test body, paired with the kind they
#: report. A tuple rather than two branches in `scan`, so adding the next one is
#: an entry here rather than another arm.
_CALL_CHECKS: tuple[tuple[str, Callable[[ast.Call], str | None]], ...] = (
    ("broad-raises", _broad_raises),
    ("unmatched-raises", _unmatched_raises),
)


def _can_be_empty(iterable: ast.expr) -> bool:
    """Whether the loop might run zero times for a reason the test cannot see.

    A loop over a literal, a range, or a fixture is fine: if it were empty the
    test would be obviously broken. The dangerous iterables are the ones a
    condition narrows, because the condition can exclude everything while the
    data still looks healthy.
    """
    if isinstance(iterable, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
        return any(gen.ifs for gen in iterable.generators)
    if isinstance(iterable, ast.Call):
        return any(w in _call_name(iterable.func) for w in ("filter", "zip", "islice", "takewhile"))
    return False


def _guarded_asserts(loop: ast.stmt) -> bool:
    """Every assertion in the loop sits behind an `if`, so an iteration may assert nothing."""
    asserts = [n for n in ast.walk(loop) if isinstance(n, ast.Assert)]
    if not asserts:
        return False
    guarded = {
        id(n)
        for branch in ast.walk(loop)
        if isinstance(branch, ast.If)
        for n in ast.walk(branch)
        if isinstance(n, ast.Assert)
    }
    return all(id(n) in guarded for n in asserts)


def _asserts_only_inside_loop(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Every assertion is inside a loop that might not reach them.

    This is the mode test_perturbation_bounds.py guards by hand, counting the
    documents it examined because "without this the test passes vacuously
    whenever every non-edited document happens to see a zero shift".

    Looping is not itself suspicious: most of these iterate a fixture that would
    be obviously broken if empty. What is suspicious is a loop whose iterable is
    narrowed by a condition, or whose assertions are all behind an `if`, since
    either can skip every assertion while the data still looks healthy. A count
    asserted outside the loop is the fix, and its presence is what clears this.
    """
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For | ast.AsyncFor | ast.While)]
    if not loops:
        return None
    inner = {id(n) for loop in loops for n in ast.walk(loop) if isinstance(n, ast.Assert)}
    total = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    if not total or any(id(n) not in inner for n in total):
        return None  # something outside every loop can still fail

    for loop in loops:
        if isinstance(loop, ast.For | ast.AsyncFor) and _can_be_empty(loop.iter):
            return "loop iterates a filtered sequence; nothing asserts it was non-empty"
        if _guarded_asserts(loop):
            return "every assertion is behind an `if` inside the loop; no iteration need reach one"
    return None


def _scan_test(path: Path, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Finding]:
    """Every check that applies to one test function."""
    body = [
        n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    if not body or all(isinstance(n, ast.Pass) for n in body):
        return [Finding(path, fn.lineno, fn.name, "empty", "body is empty or only a docstring")]

    findings: list[Finding] = []

    # A test can legitimately assert by not raising, but only if it says so: the
    # name is the assertion, and an unnamed one is indistinguishable from a test
    # somebody forgot to finish.
    declares_no_raise = any(
        w in fn.name for w in ("not_raise", "no_error", "importable", "smoke", "serialis")
    )
    if not _establishes_something(fn) and not declares_no_raise:
        findings.append(
            Finding(path, fn.lineno, fn.name, "no-assertion", "nothing in the body can fail")
        )

    for deco in fn.decorator_list:
        if detail := _empty_parametrize(deco):
            findings.append(Finding(path, fn.lineno, fn.name, "empty-parametrize", detail))

    if detail := _unmarked_property(fn):
        findings.append(Finding(path, fn.lineno, fn.name, "unmarked-property", detail))

    for node in ast.walk(fn):
        if isinstance(node, ast.Assert) and (detail := _constant_assert(node)):
            findings.append(Finding(path, node.lineno, fn.name, "constant-assert", detail))
        if isinstance(node, ast.Call):
            findings.extend(
                Finding(path, node.lineno, fn.name, kind, detail)
                for kind, detector in _CALL_CHECKS
                if (detail := detector(node))
            )

    if detail := _asserts_only_inside_loop(fn):
        findings.append(Finding(path, fn.lineno, fn.name, "loop-only-assert", detail))

    return findings


def scan(path: Path) -> list[Finding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # a test file that cannot parse is its own problem
        return [Finding(path, exc.lineno or 0, "<module>", "unparseable", str(exc.msg))]

    findings: list[Finding] = []
    for fn in ast.walk(tree):
        if _is_test(fn):
            assert isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
            findings.extend(_scan_test(path, fn))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=REPO / "tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every file scanned")
    args = parser.parse_args()

    files = sorted(args.tests.rglob("test_*.py"))
    if not files:
        print(f"no test files under {args.tests}", file=sys.stderr)
        return 1

    findings: list[Finding] = []
    for path in files:
        found = scan(path)
        findings.extend(found)
        if args.verbose:
            print(f"  {path.relative_to(REPO)}: {len(found)} finding(s)")

    n_tests = sum(
        1
        for path in files
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if _is_test(node)
    )

    if not findings:
        print(f"no vacuous tests found ({n_tests} tests across {len(files)} files)")
        return 0

    by_kind: dict[str, int] = {}
    for f in sorted(findings, key=lambda f: (str(f.path), f.line)):
        print(f.format())
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1

    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
    print(f"\n{len(findings)} finding(s) across {n_tests} tests: {summary}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
