"""The repository's own gates, tested for their ability to fail.

`scripts/check_*.py` are the checks CI runs over the repository itself rather
than over its output. Two of the five had no owning test: `check_layout.py` and
`check_versions.py`. Nothing established that either could report a problem, and
a gate nobody has seen fail is indistinguishable from a gate that cannot.

That is not hypothetical here. `check_versions.py` grew a `--tag` argument to
stop a release publishing one version under another, its docstring named the
scenario, and nothing ever passed the flag: the only invocation in the
repository was `ci.yml`, without `--tag`, and `ci.yml` triggers on branches and
pull requests rather than tags. The guard worked. It was wired to nothing.

These call the checkers' own `check()` functions with crafted inputs, so nothing
under version control is touched. The checkers read the real tree, which is what
makes the passing direction meaningful: it says this repository is consistent
now, not merely that a fixture is.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def _script(name: str) -> ModuleType:
    """Import a checker from `scripts/`. Local by house convention.

    Registered in `sys.modules` before execution, matching
    `tests/test_mutation_gate.py`: these modules are read by other tooling and a
    half-initialised entry is worse than none.
    """
    spec = importlib.util.spec_from_file_location(f"_gate_{name}", REPO / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_gate_{name}"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# check_versions.py
# ---------------------------------------------------------------------------
def test_the_version_gate_passes_on_this_repository() -> None:
    """The baseline. Every rejection below is vacuous if this fails."""
    assert _script("check_versions").check() == []


def test_a_release_tag_naming_another_version_is_rejected() -> None:
    """The scenario the `--tag` argument exists for.

    Its docstring: "`git tag v0.3.0` on a tree declaring 0.2.0 would build, test,
    gate and publish 0.2.0 under a 0.3.0 release." The comparison is what stops
    that, and until `release.yml` was made to pass the tag, nothing invoked it.
    """
    problems = _script("check_versions").check("v9.9.9")

    assert problems, "a tag disagreeing with every file in the tree must be reported"
    assert any("stated differently" in p for p in problems)
    assert any("9.9.9" in p for p in problems), "and the report must name the offending value"


def test_the_v_prefix_is_not_itself_a_disagreement() -> None:
    """`v0.2.0` and `0.2.0` are one claim written two ways.

    A gate treating them as different would fail every correctly-tagged release,
    which is the failure mode that gets a check deleted rather than fixed.
    """
    gate = _script("check_versions")
    stated = gate.check()
    assert stated == [], "the premise: the tree agrees with itself"

    for tag in ("0.2.0", "v0.2.0"):
        assert gate.check(tag) == [], f"{tag} agrees with the tree and must pass"


def test_every_file_the_gate_reads_is_one_that_exists() -> None:
    """A source that has been renamed away reports "no version found" forever.

    The gate treats a missing file as a problem rather than skipping it, so a
    stale entry would fail every run; this catches the rename at the point it
    happens instead.
    """
    gate = _script("check_versions")

    for relative in gate._SOURCES:
        assert (REPO / relative).exists(), f"{relative} is named by the gate but is not there"


# ---------------------------------------------------------------------------
# check_layout.py
# ---------------------------------------------------------------------------
def test_the_layout_gate_passes_on_this_repository() -> None:
    """The baseline: every C++ subpackage has a Python counterpart or is `core`."""
    assert _script("check_layout").main() == 0


def test_a_cpp_directory_with_no_python_counterpart_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one direction this gate checks, shown to fail.

    Adding `_CPP_ONLY = set()` removes `core`'s exemption, which is the cheapest
    way to present the gate with a genuine orphan without writing to `cpp/`.
    `core` is a real C++ directory with no Python module, so unexempting it is
    exactly the condition the gate exists to catch.
    """
    gate = _script("check_layout")
    monkeypatch.setattr(gate, "_CPP_ONLY", set())

    assert gate.main() == 1, "an unexempted C++ directory with no Python module must fail"


def test_the_exemption_is_what_makes_the_baseline_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contrastive with the test above, so neither passes for the wrong reason.

    `core` is exempt by design -- numeric policy, the floating-point guard and
    the build configuration, none of which the reference backend needs a module
    for. Restoring the exemption must restore the pass, or the failure above
    would be evidence of something else.
    """
    gate = _script("check_layout")
    monkeypatch.setattr(gate, "_CPP_ONLY", {"core"})

    assert gate.main() == 0
