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

import ast
import hashlib
import importlib.util
import shutil
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


def test_a_module_with_no_claims_still_fails_on_any_survivor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence of an allowlist entry must never read as permission. A module
    nobody has documented behaves exactly as the tool did before this file.

    The empty allowlist is substituted rather than borrowed, so the premise does
    not depend on which modules the real file happens to document. The module
    named carries real entries, which places the emptiness in the substituted
    file rather than in the module.
    """
    runner = _runner()
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._load_equivalents(module), "the premise: this module is documented"

    empty = tmp_path / "equivalent_mutants.txt"
    empty.write_text("# an allowlist making no claim about anything\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_EQUIVALENTS", empty)

    assert runner._load_equivalents(module) == {}, "and now, by construction, it is not"
    assert runner._verdict(module, [_mutant(runner, 155, "compare", "LtE", "Lt")]) == 1
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


# ---------------------------------------------------------------------------
# The sandbox: the campaign must execute the mutant, not the working tree
# ---------------------------------------------------------------------------
# Everything above tests the verdict. None of it tests the premise the verdict
# rests on -- that the tests a campaign runs import the *mutated copy*. They did
# not, and nothing noticed, because a campaign in that state reports every
# mutant as a survivor and a survivor is a normal result.
#
# scikit-build-core's editable `.pth` has two lines: one installs a
# `sys.meta_path` finder, the other appends the real `src/` to `sys.path`.
# `_make_sandbox` undid the first. The second still put the working tree on the
# path, and `tests/conftest.py` adds `src/` only `if find_spec("tfidf_stability")
# is None` -- which it then was not, so the sandbox was never inserted.
#
# Measured while the second line stood: `margins.py` scored 0 killed of 6 and
# `stratify.py` 0 of 29, including flips their own test files fail on when run
# by hand. Both score normally once the sandbox really is first.
#
# It bit a developer checkout rather than the nightly: `requirements-dev.txt`
# does not install this package, so in CI `find_spec` returns None and conftest
# inserts the sandbox's own `src/`. The editable install is what supplies the
# competing answer. That is precisely why it needs a test -- the environment
# where it breaks is the one no job runs in.
@pytest.mark.slow
def test_a_test_run_inside_the_sandbox_imports_the_sandbox_copy() -> None:
    """A module written only into the sandbox must be the one a test sees.

    Marked slow, and its faster sibling below is the one the PR tier runs. That
    sibling is strictly stronger -- proving the sandbox wins subsumes proving it
    is reachable -- so the PR tier keeps the assertion that matters and this one
    adds the cleaner diagnostic on the nightly, where the campaign it guards
    runs anyway. Both spend their time copying the tree and starting a pytest
    subprocess; neither has a cheap form.

    The probe is a new module rather than an edit to an existing one, so the
    assertion cannot pass by coincidence: if the working tree is imported
    instead, the module is simply not there and the run fails on ImportError.
    That is the same signal a mutated module gives, arrived at without having to
    mutate anything.
    """
    runner = _runner()
    sandbox = runner._make_sandbox()
    try:
        probe = sandbox / "src" / "tfidf_stability" / "_sandbox_probe.py"
        probe.write_text('VALUE = "sandbox"\n', encoding="utf-8", newline="")
        test = sandbox / "tests" / "test_zz_sandbox_probe.py"
        test.write_text(
            "from tfidf_stability._sandbox_probe import VALUE\n\n\n"
            "def test_probe() -> None:\n"
            '    assert VALUE == "sandbox"\n',
            encoding="utf-8",
            newline="",
        )

        assert not (REPO / "src" / "tfidf_stability" / "_sandbox_probe.py").exists(), (
            "the premise: this module exists only inside the sandbox"
        )
        assert runner._run_tests(["tests/test_zz_sandbox_probe.py"], timeout=180.0, cwd=sandbox), (
            "the sandbox copy of the package was not what the test imported"
        )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_the_sandbox_shadows_the_working_tree_rather_than_sitting_behind_it() -> None:
    """The stronger half: not merely reachable, but *first*.

    A probe module is enough to prove the sandbox is on the path somewhere. It
    is not enough to prove it wins, and winning is the whole question -- every
    module a campaign mutates exists in both trees. Here the sandbox's copy of a
    real module is replaced outright, so the only way the run can fail is if the
    sandbox is what got imported.
    """
    runner = _runner()
    sandbox = runner._make_sandbox()
    try:
        shadowed = sandbox / "src" / "tfidf_stability" / "utils" / "numerics.py"
        assert shadowed.exists(), "the premise: the sandbox holds its own copy of this module"
        shadowed.write_text(
            'raise RuntimeError("the sandbox copy was imported")\n',
            encoding="utf-8",
            newline="",
        )
        test = sandbox / "tests" / "test_zz_sandbox_shadow.py"
        test.write_text(
            "import tfidf_stability.utils.numerics  # noqa: F401\n\n\n"
            "def test_import() -> None:\n"
            "    pass\n",
            encoding="utf-8",
            newline="",
        )

        assert not runner._run_tests(
            ["tests/test_zz_sandbox_shadow.py"], timeout=180.0, cwd=sandbox
        ), "the working tree's numerics.py shadowed the sandbox's, so no mutant is ever executed"
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


# ---------------------------------------------------------------------------
# The allowlist is keyed by line number, so an edit above one invalidates it
# ---------------------------------------------------------------------------
# Every entry names `(path, line, kind, before, after)`. Insert a comment above
# one and the mutation it describes moves; the entry now points at whatever else
# is on that line, or at nothing. The campaign then reports the same mutant
# twice over -- stale here, undocumented there -- which reads as two findings
# and is neither.
#
# That went unchecked because a campaign is the only thing that noticed, and
# `nightly.yml` runs campaigns for three modules while this file documents
# twenty. Adding five comment lines to `noise_floor.py` shifted nine of its
# entries and no gate anywhere would have said so.
#
# It does not need a campaign. Whether a line still carries the mutation an
# entry describes is answerable from the AST alone, in about a second for the
# whole file, so it belongs in the PR tier rather than the nightly.
def _sites_by_key(runner: ModuleType, module: Path) -> set[tuple[int, str, str, str]]:
    """Every mutable site in `module`, keyed the way the allowlist keys them.

    One pass rather than one per site. `_Mutator` is built to apply a single
    mutation chosen by index, so the obvious enumeration re-parses the module
    once per site: over the twenty modules this file documents that was 23
    seconds through `_apply`, which also unparses, and 11 through the mutator
    alone. `_hit` is called at every site whatever the target, so overriding it
    records the lot in a single visit and mutates nothing -- `target=-1` matches
    no index, so the base call always declines.

    The count is cross-checked against `_count_sites`, which walks the same
    visitor independently. If the two ever disagree this enumeration has missed
    a site and the check above would silently stop covering it.
    """
    source = module.read_text(encoding="utf-8")

    class _Recorder(runner._Mutator):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__(target=-1)
            self.sites: list[tuple[int, str, str, str]] = []

        def _hit(self, node: ast.AST, kind: str, before: str, after: str) -> bool:
            declined = super()._hit(node, kind, before, after)
            self.sites.append((getattr(node, "lineno", 0), kind, before, after))
            return declined

    recorder = _Recorder()
    recorder.visit(ast.parse(source))
    assert len(recorder.sites) == runner._count_sites(source), (
        f"{module.name}: recorded {len(recorder.sites)} sites but the campaign counts "
        f"{runner._count_sites(source)}"
    )
    return set(recorder.sites)


def test_every_allowlist_entry_still_names_a_mutation_that_exists() -> None:
    """An entry pointing at a line that no longer carries that mutation is dead.

    Dead in a way that costs twice: the argument it records is lost, and the
    mutant it was written for is now undocumented and will fail a campaign that
    should have passed. Both were true of eleven entries after a round of edits
    that changed no behaviour at all.

    The failure message carries the whole key, because the fix is a renumber and
    the only thing needed to make it is the line the mutation moved to.
    """
    runner = _runner()
    entries = [
        line.split()
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert entries, "the premise: the allowlist is not empty"

    by_module: dict[str, list[list[str]]] = {}
    for fields in entries:
        by_module.setdefault(fields[0], []).append(fields)

    dangling: list[str] = []
    checked = 0
    for path, fields_list in by_module.items():
        module = REPO / path
        assert module.exists(), f"{path}: the allowlist names a module that is gone"
        sites = _sites_by_key(runner, module)
        for fields in fields_list:
            key = (int(fields[1]), fields[2], fields[3], fields[5])
            if key not in sites:
                dangling.append(f"{path}:{fields[1]} {fields[2]} {fields[3]} -> {fields[5]}")
            checked += 1

    assert checked == len(entries), "every entry was looked up"
    assert not dangling, "allowlist entries naming a mutation that is no longer there:\n  " + (
        "\n  ".join(dangling)
    )


def test_the_allowlist_names_only_modules_that_are_still_in_the_package() -> None:
    """A renamed or deleted module leaves entries that can never match again.

    Separate from the test above so the message says which of the two happened:
    a moved line and a moved file need different fixes.
    """
    named = {
        line.split()[0]
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert named, "the premise: the allowlist names some modules"
    missing = sorted(path for path in named if not (REPO / path).exists())

    assert not missing, f"the allowlist names modules that no longer exist: {missing}"


def _stamped_entries() -> list[tuple[str, int, str, str]]:
    """Every allowlist entry as `(path, line, stamp, reason)`.

    Local by house convention. Separate from `_sites_by_key` above, which asks
    whether the mutation exists rather than whether the line it sits on is still
    the line the entry was written about.
    """
    entries: list[tuple[str, int, str, str]] = []
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        statement, _, reason = raw.partition("#")
        fields = statement.split()
        if len(fields) < 6 or raw.lstrip().startswith("#"):
            continue
        head, _, rest = reason.strip().partition(" ")
        entries.append((fields[0], int(fields[1]), head, rest))
    return entries


def _fingerprint(line: str) -> str:
    """Hash of the source line with whitespace collapsed.

    Whitespace only, so reindenting a block does not invalidate every entry in
    it while a changed expression still does.
    """
    return hashlib.sha256(" ".join(line.split()).encode("utf-8")).hexdigest()[:8]


def test_every_allowlist_entry_still_describes_the_line_it_names() -> None:
    """The half `..._still_names_a_mutation_that_exists` does not cover.

    That test asks whether a site with the entry's `(line, kind, before, after)`
    exists, not whether it is the site the entry was written about: a renumber
    onto the nearest same-signature line satisfies it. `src=<8 hex>` prefixes
    each reason with a fingerprint of the source line, and the failure prints
    the line now sitting there.
    """
    entries = _stamped_entries()
    assert entries, "the premise: the allowlist is not empty"

    sources: dict[str, list[str]] = {}
    drifted: list[str] = []
    checked = 0

    for path, line_no, stamp, reason in entries:
        assert stamp.startswith("src="), (
            f"{path}:{line_no} carries no source stamp; regenerate it rather than "
            f"leaving an entry nothing can check"
        )
        if path not in sources:
            sources[path] = (REPO / path).read_text(encoding="utf-8").splitlines()
        body = sources[path]
        assert 1 <= line_no <= len(body), f"{path}:{line_no} is past the end of the file"

        actual = _fingerprint(body[line_no - 1])
        if stamp != f"src={actual}":
            drifted.append(
                f"{path}:{line_no}\n"
                f"      recorded {stamp}, actual src={actual}\n"
                f"      the line now reads: {body[line_no - 1].strip()[:88]}\n"
                f"      the entry says:     {reason[:88]}"
            )
        checked += 1

    assert checked == len(entries), "every entry was fingerprinted"
    assert not drifted, (
        "allowlist entries whose line no longer holds what they describe. Renumber "
        "by matching each entry's recorded reason against the source, not by "
        "nearest line, then restamp:\n  " + "\n  ".join(drifted)
    )


def test_a_stamp_pointing_at_a_different_expression_is_rejected() -> None:
    """A well-formed stamp taken from another line does not verify.

    Two real entries from the same file with their stamps exchanged: both name a
    site that exists and both stamps are well formed, so only the fingerprint
    separates them.
    """
    entries = [e for e in _stamped_entries() if e[0].endswith("numerics.py")]
    assert len(entries) >= 2, "the premise: numerics.py carries several entries"

    (_, first_line, first_stamp, _), (_, second_line, second_stamp, _) = entries[0], entries[1]
    assert first_stamp != second_stamp, "the premise: the two lines differ"

    body = (REPO / entries[0][0]).read_text(encoding="utf-8").splitlines()
    assert _fingerprint(body[first_line - 1]) == first_stamp.removeprefix("src=")
    assert _fingerprint(body[second_line - 1]) != first_stamp.removeprefix("src="), (
        "swapping two entries' stamps must not still verify"
    )


def test_an_entry_whose_only_reason_is_its_fingerprint_is_refused(tmp_path: Path) -> None:
    """The emptiness test runs after the `src=` stamp is stripped, not before.

    A reason may carry a `src=<8 hex>` prefix, which the loader removes because
    it is bookkeeping for the drift guard rather than part of the argument.
    Testing the raw text for emptiness accepted an entry whose whole reason was
    that prefix, leaving an empty argument behind: a survivor silenced with no
    stated reason, which is the suppression this file's header refuses.

    The stamped-and-argued form must still load, or the fix would silence the
    real allowlist instead.
    """
    runner = _runner()
    allowlist = tmp_path / "equivalent_mutants.txt"
    entry = "src/tfidf_stability/utils/numerics.py  112  compare  GtE -> Gt  #"
    module = Path("src/tfidf_stability/utils/numerics.py")

    allowlist.write_text(f"{entry} src=2a87a12d\n", encoding="utf-8")
    runner._EQUIVALENTS = allowlist
    assert runner._load_equivalents(module) == {}, "a stamp is not an argument"

    allowlist.write_text(f"{entry}\n", encoding="utf-8")
    assert runner._load_equivalents(module) == {}, "and neither is nothing at all"

    allowlist.write_text(f"{entry} src=2a87a12d both branches agree\n", encoding="utf-8")
    claims = runner._load_equivalents(module)
    assert claims == {(112, "compare", "GtE", "Gt"): "both branches agree"}, (
        "a stamped entry that does argue its case must still be honoured"
    )
