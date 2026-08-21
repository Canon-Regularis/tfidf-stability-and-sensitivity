"""The equivalent-mutant allowlist and the verdict it feeds.

A mutation that no test kills is usually a gap. Sometimes it is not: the
expression cannot be reached, or reaches the same answer another way, and no
test could tell the difference. `configs/equivalent_mutants.txt` records those
with their reasons, and `run_mutation_tests.py` reads it to decide whether a
campaign passed.

That decision is the whole value of the nightly job, so it is tested here rather
than trusted. Two directions matter equally: an undocumented survivor must fail,
and a documented one that no longer survives must fail too. Without the second,
the file quietly becomes a blanket suppression -- entries accumulate, nothing
ever prunes them, and the campaign goes green whatever the tests do.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_mutation_tests.py"
ALLOWLIST = REPO / "configs" / "equivalent_mutants.txt"


def _runner() -> ModuleType:
    """Import ``scripts/run_mutation_tests.py``. Local by house convention.

    Registered in ``sys.modules`` before execution because the module defines
    slotted dataclasses, and ``dataclasses`` resolves ``__module__`` through
    that table while processing them.
    """
    spec = importlib.util.spec_from_file_location("_mutation_runner", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mutation_runner"] = module
    spec.loader.exec_module(module)
    return module


def _mutant(runner: ModuleType, line: int, kind: str, before: str, after: str) -> object:
    return runner.Mutant(line=line, kind=kind, before=before, after=after)


# ---------------------------------------------------------------------------
# The allowlist file itself
# ---------------------------------------------------------------------------
def test_every_claim_names_a_module_that_exists() -> None:
    """A claim about a file that was deleted or renamed is an argument about
    nothing, and it would sit here unnoticed because no campaign would ever look
    for it."""
    missing = set()
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        statement, _, _reason = raw.partition("#")
        fields = statement.split()
        if len(fields) >= 6 and not (REPO / fields[0]).exists():
            missing.add(fields[0])

    assert missing == set(), f"claims about absent modules: {sorted(missing)}"


def test_every_claim_carries_a_reason() -> None:
    """The reason is what separates a documented equivalence from a silenced
    finding. The loader ignores an entry without one, so a claim written without
    a reason would fail open -- present in the file, absent from the gate."""
    unreasoned = []
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        statement, _, reason = raw.partition("#")
        if statement.split() and not reason.strip():
            unreasoned.append(statement.strip())

    assert unreasoned == []


def test_the_allowlist_covers_more_than_one_module() -> None:
    """The premise of the file. If it held a single module's claims the gate
    would be doing almost nothing."""
    modules = {
        line.split()[0]
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.split() and not line.startswith("#")
    }

    assert len(modules) > 15, f"only {len(modules)} modules documented"


def test_claims_are_loaded_only_for_the_module_asked_about() -> None:
    """Keyed by path, so one module's equivalences cannot excuse another's
    survivors."""
    runner = _runner()

    numerics = runner._load_equivalents(Path("src/tfidf_stability/utils/numerics.py"))
    sort_keys = runner._load_equivalents(Path("src/tfidf_stability/ranking/sort_keys.py"))

    assert len(numerics) > 1
    assert len(sort_keys) == 1
    assert runner._load_equivalents(Path("src/tfidf_stability/not_a_module.py")) == {}


def test_a_claim_resolves_to_its_reason() -> None:
    """The key is the whole mutation, not just the line: two mutable sites share
    a line often enough that a line-only key would excuse the wrong one."""
    runner = _runner()
    claims = runner._load_equivalents(Path("src/tfidf_stability/ranking/sort_keys.py"))

    assert (155, "compare", "LtE", "Lt") in claims
    assert "transitive" in claims[(155, "compare", "LtE", "Lt")]


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def test_a_documented_survivor_does_not_fail_the_campaign() -> None:
    """The reason the file exists. Every module has equivalents, so without this
    the nightly fails every night whatever the tests do."""
    runner = _runner()
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    survivors = [_mutant(runner, 155, "compare", "LtE", "Lt")]

    assert runner._verdict(module, survivors) == 0


def test_an_undocumented_survivor_fails_the_campaign() -> None:
    """The gap the tool exists to find, which the allowlist must not swallow."""
    runner = _runner()
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    survivors = [
        _mutant(runner, 155, "compare", "LtE", "Lt"),
        _mutant(runner, 42, "constant", "1", "0"),
    ]

    assert runner._verdict(module, survivors) == 1


def test_a_claim_that_matches_no_survivor_fails_the_campaign() -> None:
    """The anti-accretion half. The mutant is killed now, or the line moved;
    either way the argument has to be re-read rather than left standing."""
    runner = _runner()
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._verdict(module, []) == 1


def test_a_module_with_no_claims_still_fails_on_any_survivor() -> None:
    """Absence of an allowlist entry must never read as permission. A module
    nobody has documented behaves exactly as the tool did before this file."""
    runner = _runner()
    module = Path("src/tfidf_stability/persistence/save_load.py")

    assert runner._load_equivalents(module) == {}, "the premise: nothing documented here"
    assert runner._verdict(module, [_mutant(runner, 1, "constant", "1", "0")]) == 1
    assert runner._verdict(module, []) == 0


@pytest.mark.parametrize(
    ("line", "kind", "before", "after"),
    [
        (155, "compare", "LtE", "GtE"),
        (155, "constant", "LtE", "Lt"),
        (156, "compare", "LtE", "Lt"),
    ],
)
def test_a_near_miss_is_not_covered_by_the_claim(
    line: int, kind: str, before: str, after: str
) -> None:
    """Every field is part of the key. A different mutation on the same line, or
    the same mutation one line over, is a different claim and needs its own."""
    runner = _runner()
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._verdict(module, [_mutant(runner, line, kind, before, after)]) == 1


def test_one_claim_covers_every_mutant_sharing_its_key() -> None:
    """The documented limitation, pinned so it stays deliberate.

    `@dataclass(frozen=True, slots=True)` is two mutable sites on one line, and
    both yield `constant True -> False`. The key cannot separate them, so a
    single claim excuses both -- which is why the file's header requires such a
    claim to justify every mutation matching it, not whichever was looked at
    first.

    The count is reported over survivors rather than over claims, so two
    survivors under one claim read as two accounted for and not as one still
    outstanding.
    """
    runner = _runner()
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    twins = [
        _mutant(runner, 155, "compare", "LtE", "Lt"),
        _mutant(runner, 155, "compare", "LtE", "Lt"),
    ]

    assert runner._verdict(module, twins) == 0


def test_a_claim_does_not_cover_the_same_mutation_one_line_over() -> None:
    """A note, not a check: the mutation's index in document order would
    separate the twins above, but it shifts whenever anything earlier in the
    file is edited, so every claim in a module would go stale on an unrelated
    change. The line-based key trades that for the collision.

    What is asserted is the consequence: a claim is keyed by line, so moving the
    expression invalidates it.
    """
    runner = _runner()
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    moved = [_mutant(runner, 156, "compare", "LtE", "Lt")]

    assert runner._verdict(module, moved) == 1, "the claim at 155 must not cover 156"
