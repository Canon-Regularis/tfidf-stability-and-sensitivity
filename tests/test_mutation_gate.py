"""The equivalent-mutant allowlist and the verdict it feeds.

`configs/equivalent_mutants.txt` records mutations no test can kill, with a
reason each, and `tooling/mutation_python.py` reads it to pass or fail a
campaign. An undocumented survivor fails, and so does a claim matching no survivor.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

import pytest

from tooling import mutation_python as runner
from tooling import renumber_allowlists as renumber
from tooling.allowlist import fingerprint as _fingerprint
from tooling.allowlist import parse_python

REPO = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO / "configs" / "equivalent_mutants.txt"


def _mutant(line: int, kind: str, before: str, after: str) -> object:
    return runner.Mutant(line=line, kind=kind, before=before, after=after)


# ---------------------------------------------------------------------------
# The allowlist file itself
# ---------------------------------------------------------------------------
def test_every_claim_names_a_module_that_exists() -> None:
    """Every allowlist claim names a module that exists in the tree.

    No campaign runs for an absent module, so a stale claim would go unnoticed.
    """
    missing = set()
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        statement, _, _reason = raw.partition("#")
        fields = statement.split()
        if len(fields) >= 6 and not (REPO / fields[0]).exists():
            missing.add(fields[0])

    assert missing == set(), f"claims about absent modules: {sorted(missing)}"


def test_every_claim_carries_a_reason() -> None:
    """Every allowlist claim carries a reason.

    The loader ignores an entry without one, so an unreasoned claim fails open:
    present in the file, absent from the gate.
    """
    unreasoned = []
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        statement, _, reason = raw.partition("#")
        if statement.split() and not reason.strip():
            unreasoned.append(statement.strip())

    assert unreasoned == []


def test_the_allowlist_covers_more_than_one_module() -> None:
    """The allowlist documents claims for more than one module.

    A file holding a single module's claims would leave the gate almost idle.
    """
    modules = {
        line.split()[0]
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.split() and not line.startswith("#")
    }

    assert len(modules) > 15, f"only {len(modules)} modules documented"


def test_claims_are_loaded_only_for_the_module_asked_about() -> None:
    """Claims are keyed by path, so one module's cannot excuse another's."""

    numerics = runner._load_equivalents(Path("src/tfidf_stability/utils/numerics.py"))
    sort_keys = runner._load_equivalents(Path("src/tfidf_stability/ranking/sort_keys.py"))

    assert len(numerics) > 1
    assert len(sort_keys) == 1
    assert runner._load_equivalents(Path("src/tfidf_stability/not_a_module.py")) == {}


def test_a_claim_resolves_to_its_reason() -> None:
    """A claim is keyed by the whole mutation and resolves to its reason.

    Two mutable sites often share a line, so a line-only key could excuse the
    wrong one.
    """
    claims = runner._load_equivalents(Path("src/tfidf_stability/ranking/sort_keys.py"))

    assert (123, "compare", "LtE", "Lt") in claims
    assert "transitive" in claims[(123, "compare", "LtE", "Lt")]


# ---------------------------------------------------------------------------
# The sandbox: what a campaign needs to be a working checkout
# ---------------------------------------------------------------------------

#: A top-level entry named as `REPO / "<name>"`. The capture must look like a
#: path, or the pattern matches its own source in this file.
_TOP_LEVEL_READ = re.compile(r'REPO / "([\w.@+-]+)"')

#: Directories whose files name repository paths. A test that opens a repository
#: file needs it in the sandbox; `scripts/` and `tooling/` are scanned because a
#: test runs a gate script, which opens files of its own.
_SCANNED = ("tests", "scripts", "tooling")

#: Top-level names a gate probes that the sandbox must not hold. A sandbox is not
#: a checkout, and `scripts/check_vendored.py` tests for `.git` to tell them apart.
_PROBED_NOT_COPIED = frozenset({".git"})


def _top_level_reads() -> dict[str, set[str]]:
    """Every top-level repository entry the suite and its gates name, by file."""
    found: dict[str, set[str]] = {}
    for directory in _SCANNED:
        for path in sorted((REPO / directory).glob("*.py")):
            names = set(_TOP_LEVEL_READ.findall(path.read_text(encoding="utf-8")))
            if names:
                found[f"{directory}/{path.name}"] = names
    return found


def test_the_sandbox_holds_every_top_level_path_the_suite_names() -> None:
    """The sandbox copies every top-level path the suite or its gates name.

    A campaign runs against a copy, so a file only the real tree has fails the
    baseline before any mutant. The check over-approximates: a path a script
    only writes into is copied too, at the cost of one extra copy per campaign.
    """
    contents = set(runner._SANDBOX_CONTENTS)
    assert not contents & _PROBED_NOT_COPIED, "a sandbox holding .git would be a checkout"
    missing = {
        where: sorted(names - contents - _PROBED_NOT_COPIED)
        for where, names in _top_level_reads().items()
        if names - contents - _PROBED_NOT_COPIED
    }

    assert missing == {}, (
        f"named from the repository root but never copied into the sandbox, so "
        f"the baseline fails before any mutant: {missing}"
    )


def test_the_scan_finds_the_paths_it_is_meant_to_check() -> None:
    """The scan matches the paths it is meant to find.

    A pattern that matched nothing would make the sandbox check pass vacuously.
    """
    reads = _top_level_reads()
    named = {name for names in reads.values() for name in names}

    assert len(reads) > 8, f"only {len(reads)} files name a top-level path"
    assert "CMakePresets.json" in reads.get("scripts/check_docs.py", set())
    assert ".github" in reads.get("tests/test_cpp_mutation_gate.py", set())
    assert not any(not name.replace(".", "").replace("-", "").isalnum() for name in named), (
        f"the pattern captured something that is not a path: {sorted(named)}"
    )


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def test_a_documented_survivor_does_not_fail_the_campaign() -> None:
    """A survivor with an allowlist claim does not fail the campaign.

    Every module has equivalents, so without this the nightly fails every night.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    survivors = [_mutant(123, "compare", "LtE", "Lt")]

    assert runner._verdict(module, survivors) == 0


def test_an_undocumented_survivor_fails_the_campaign() -> None:
    """A survivor with no allowlist claim fails the campaign."""
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    survivors = [
        _mutant(123, "compare", "LtE", "Lt"),
        _mutant(42, "constant", "1", "0"),
    ]

    assert runner._verdict(module, survivors) == 1


def test_a_claim_that_matches_no_survivor_fails_the_campaign() -> None:
    """A claim that matches no survivor fails the campaign.

    The mutant is killed now, or the line moved; either way the argument must
    be re-read rather than left standing.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._verdict(module, []) == 1


def test_a_module_with_no_claims_still_fails_on_any_survivor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module with no claims fails on any survivor and passes on none.

    An empty allowlist is substituted rather than borrowed, so the premise does
    not depend on which modules the real file documents. The module named
    carries real entries, which places the emptiness in the substituted file.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._load_equivalents(module), "the premise: this module is documented"

    empty = tmp_path / "equivalent_mutants.txt"
    empty.write_text("# an allowlist making no claim about anything\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_EQUIVALENTS", empty)

    assert runner._load_equivalents(module) == {}, "and now, by construction, it is not"
    assert runner._verdict(module, [_mutant(123, "compare", "LtE", "Lt")]) == 1
    assert runner._verdict(module, []) == 0


@pytest.mark.parametrize(
    ("line", "kind", "before", "after"),
    [
        (123, "compare", "LtE", "GtE"),
        (123, "constant", "LtE", "Lt"),
        (124, "compare", "LtE", "Lt"),
    ],
)
def test_a_near_miss_is_not_covered_by_the_claim(
    line: int, kind: str, before: str, after: str
) -> None:
    """Every field of the key must match for a claim to cover a mutant.

    A different mutation on the same line, or the same mutation one line over,
    needs a claim of its own.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._verdict(module, [_mutant(line, kind, before, after)]) == 1


def test_one_claim_covers_every_mutant_sharing_its_key() -> None:
    """One claim covers every survivor that shares its key.

    `@dataclass(frozen=True, slots=True)` is two mutable sites on one line that
    both yield `constant True -> False`; the key cannot separate them, so the
    allowlist header requires a claim to justify every mutation matching it.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    twins = [
        _mutant(123, "compare", "LtE", "Lt"),
        _mutant(123, "compare", "LtE", "Lt"),
    ]

    assert runner._verdict(module, twins) == 0


def test_a_claim_does_not_cover_the_same_mutation_one_line_over() -> None:
    """A claim is keyed by line, so moving the expression invalidates it.

    A document-order index would separate twins on one line but shifts on any
    edit earlier in the file.
    Rationale: docs/engineering_notes.md#line-keyed-allowlist
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    moved = [_mutant(124, "compare", "LtE", "Lt")]

    assert runner._verdict(module, moved) == 1, "the claim at 123 must not cover 124"


# ---------------------------------------------------------------------------
# The sandbox: the campaign must execute the mutant, not the working tree
# ---------------------------------------------------------------------------


# Rationale: docs/engineering_notes.md#sandbox-import-precedence
@pytest.mark.slow
def test_a_test_run_inside_the_sandbox_imports_the_sandbox_copy() -> None:
    """A module written only into the sandbox is the one a test run there sees.

    The probe is a new module, so a run that imports the working tree fails on
    ImportError instead of passing. The faster sibling below is strictly stronger;
    this one is slow, so only ci.yml's reference and native jobs run it.
    """
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
    """The sandbox copy of a real module is imported ahead of the working tree.

    Every module a campaign mutates exists in both trees, so reachability is not
    enough. The sandbox's copy is replaced outright, and the run can only fail
    if that copy is what gets imported.
    """
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


# Line drift is checked from the AST, in the PR tier rather than by a campaign.
# Rationale: docs/engineering_notes.md#allowlist-drift-check
def _sites_by_key(module: Path) -> set[tuple[int, str, str, str]]:
    """Every mutable site in `module`, keyed the way the allowlist keys them.

    One visit with `target=-1` records every `_hit` and mutates nothing; the
    count is cross-checked against `_count_sites`, which walks the same visitor.
    Rationale: docs/engineering_notes.md#allowlist-site-enumeration
    """
    source = module.read_text(encoding="utf-8")

    class _Recorder(runner._Mutator):
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
    """Every allowlist entry names a mutation that exists at its line.

    A dead entry costs twice: its argument is lost, and the mutant it covers is
    undocumented and fails a campaign that should pass. The failure message
    carries the whole key, because the fix is a renumber to the new line.
    """
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
        sites = _sites_by_key(module)
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
    """Every allowlist entry names a module that is still in the package.

    Separate from the test above so the message says which happened: a moved
    line and a moved file need different fixes.
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

    Separate from `_sites_by_key`, which asks whether the mutation exists rather
    than whether the line it sits on is the one the entry describes.
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


def test_every_allowlist_entry_still_describes_the_line_it_names() -> None:
    """Every entry's `src=` stamp matches the fingerprint of the line it names.

    The mutation-exists check is satisfied by a renumber onto the nearest line
    with the same signature. The stamp prefixes each reason with a fingerprint
    of the source line, and the failure prints the line now sitting there.
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


def test_an_entry_whose_only_reason_is_its_fingerprint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry whose reason is only a `src=` stamp is refused by the loader.

    The loader strips the stamp before testing the reason for emptiness, so a
    stamp alone leaves no argument and the entry is dropped. A stamped entry
    that also argues its case still loads.
    """
    allowlist = tmp_path / "equivalent_mutants.txt"
    entry = "src/tfidf_stability/utils/numerics.py  112  compare  GtE -> Gt  #"
    module = Path("src/tfidf_stability/utils/numerics.py")

    allowlist.write_text(f"{entry} src=2a87a12d\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_EQUIVALENTS", allowlist)
    assert runner._load_equivalents(module) == {}, "a stamp is not an argument"

    allowlist.write_text(f"{entry}\n", encoding="utf-8")
    assert runner._load_equivalents(module) == {}, "and neither is nothing at all"

    allowlist.write_text(f"{entry} src=2a87a12d both branches agree\n", encoding="utf-8")
    claims = runner._load_equivalents(module)
    assert claims == {(112, "compare", "GtE", "Gt"): "both branches agree"}, (
        "a stamped entry that does argue its case must still be honoured"
    )


# ---------------------------------------------------------------------------
# An argument nothing rechecks
# ---------------------------------------------------------------------------

#: Argued modules the nightly matrix does not re-run, each with its reason. The
#: "matched no survivor" check runs only for modules the matrix names, so each
#: entry is a module still to schedule; scheduling one needs a measured scoped
#: test list, because too narrow a list reports false survivors.
_NOT_YET_SCHEDULED: dict[str, str] = {
    "src/tfidf_stability/ranking/attributes.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/utils/numerics.py": (
        "reachable from 27 test files; a full scope is 217 minutes against a 90-minute cap"
    ),
    "src/tfidf_stability/datasets/synthetic.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/vectorisation/sparse.py": (
        "reachable from 20 test files; a full scope is 182 minutes"
    ),
    "src/tfidf_stability/vectorisation/vocabulary.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/vectorisation/tfidf.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/benchmarks/tfidf_perf.py": (
        "the benchmark harness; its timings are not asserted, so every mutant survives"
    ),
    "src/tfidf_stability/similarity/cosine.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/perturbation/experiments.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/cli/main.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/ranking/ranker.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/utils/io.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/vectorisation/df_counts.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/preprocessing/tokenise.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/preprocessing/stopwords.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/preprocessing/ngrams.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/profiles/user_profile.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/analysis/stratify.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/analysis/query_grid.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/cli/commands.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/persistence/save_load.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
    "src/tfidf_stability/_native/__init__.py": (
        "no scoped test list measured yet; too narrow a list reports false survivors"
    ),
}


def _scheduled() -> set[str]:
    """Every module the nightly mutation matrix runs a campaign for.

    The workflow is the single source of truth: this reads it rather than
    restating it, so the two cannot drift.
    """
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")
    )
    return {e["module"] for e in workflow["jobs"]["mutation"]["strategy"]["matrix"]["include"]}


def _matrix_entries() -> list[dict[str, str]]:
    """The matrix rows, for the checks that need the test lists as well."""
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")
    )
    return list(workflow["jobs"]["mutation"]["strategy"]["matrix"]["include"])


def test_every_argued_module_is_scheduled_or_deferred_with_a_reason() -> None:
    """Every argued module is scheduled, or deferred with a reason.

    The "matched no survivor" check fires only for modules the matrix runs, so
    an unscheduled, undeferred module is a silent gap. Stated as one test over
    four relations, matching `tests/test_cpp_mutation_gate.py`.
    """
    scheduled, pending = _scheduled(), set(_NOT_YET_SCHEDULED)
    claimed = {path for path, *_ in _stamped_entries()}

    assert claimed, "no module carries an argued equivalence; the check is vacuous"
    assert not scheduled & pending, f"scheduled and deferred at once: {sorted(scheduled & pending)}"
    assert not claimed - scheduled - pending, (
        "modules whose equivalence arguments nothing rechecks, and which are not "
        f"listed as pending: {sorted(claimed - scheduled - pending)}"
    )
    assert not {m for m in pending if not (REPO / m).exists()}, (
        f"deferred modules that no longer exist: "
        f"{sorted(m for m in pending if not (REPO / m).exists())}"
    )
    assert all(reason.strip() for reason in _NOT_YET_SCHEDULED.values()), (
        "a deferral with no reason is indistinguishable from an oversight"
    )


def test_every_scheduled_module_and_its_tests_exist() -> None:
    """Every scheduled module and the test paths it names exist on disk.

    A matrix entry names paths as bare strings that nothing resolves. A missing
    module makes the runner exit; a missing test path is a pytest collection
    error, which the runner scores as `killed` for every mutant.
    """
    entries = _matrix_entries()
    assert entries, "no module is scheduled; the check is vacuous"

    missing = []
    for entry in entries:
        if not (REPO / entry["module"]).is_file():
            missing.append(f"module {entry['module']}")
        for test in str(entry["tests"]).split():
            if not (REPO / test).is_file():
                missing.append(f"{entry['module']} names {test}")
    assert not missing, "nightly mutation matrix points at paths that do not exist: " + "; ".join(
        missing
    )


# ---------------------------------------------------------------------------
# The renumbering tool: what it moves, and what it refuses to move
# ---------------------------------------------------------------------------
def _crafted(tmp_path: Path, source: str, line: int, argument: str = "an argument") -> list[object]:
    """A one-entry Python allowlist over a crafted module, parsed.

    The stamp is taken from ``source``'s line ``line``, so the entry starts out
    describing exactly what it names.
    """
    module = tmp_path / "src" / "crafted.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(source, encoding="utf-8")
    stamp = _fingerprint(source.split("\n")[line - 1])
    text = f"src/crafted.py  {line}  compare  LtE -> Lt  # src={stamp} {argument}\n"
    return parse_python(text)


def test_the_renumberer_proposes_nothing_on_a_clean_tree() -> None:
    """The renumberer proposes no moves or refusals on the live allowlists.

    Every entry in both allowlists names the line its stamp describes.
    """
    for path, parse in renumber.ALLOWLISTS.items():
        moves, refusals = renumber.plan_file(parse(path.read_text(encoding="utf-8")))

        assert moves == [], f"{path.name} has entries whose line has moved"
        assert refusals == [], f"{path.name} has entries the tool cannot place"


def test_a_stamped_line_that_moved_down_is_relocated(tmp_path: Path) -> None:
    """An edit above a covered line moves its entry to the line the stamp finds."""
    entries = _crafted(tmp_path, "a = 1\nif x <= y:\nb = 2\n", 2)
    (tmp_path / "src" / "crafted.py").write_text(
        "# added\n# added\na = 1\nif x <= y:\nb = 2\n", encoding="utf-8"
    )

    moves, refusals = renumber.plan_file(entries, repo=tmp_path)

    assert refusals == []
    assert [(m.entry.line, m.to_line) for m in moves] == [(2, 4)]


def test_an_entry_whose_expression_changed_is_refused(tmp_path: Path) -> None:
    """A changed expression is refused rather than relocated.

    Only a reader can say whether the argument still holds, so the tool reports
    and changes nothing.
    """
    entries = _crafted(tmp_path, "a = 1\nif x <= y:\nb = 2\n", 2)
    (tmp_path / "src" / "crafted.py").write_text("a = 1\nif x < y:\nb = 2\n", encoding="utf-8")

    moves, refusals = renumber.plan_file(entries, repo=tmp_path)

    assert moves == []
    assert len(refusals) == 1
    assert "no line carries its stamp" in refusals[0].detail


def test_an_entry_whose_stamp_matches_two_lines_is_refused_as_ambiguous(tmp_path: Path) -> None:
    """A stamp matching two lines is refused as ambiguous.

    Thirteen of the Python entries and five of the C++ ones sit on line text
    that is not unique in its file, so a stamp alone cannot place an entry.
    """
    entries = _crafted(tmp_path, "a = 1\nif x <= y:\nb = 2\n", 2)
    # The entry's own line must stop matching first: an entry still sitting on a
    # line its stamp describes is left alone whatever else the file holds.
    (tmp_path / "src" / "crafted.py").write_text(
        "a = 1\nb = 2\nif x <= y:\nif x <= y:\n", encoding="utf-8"
    )

    moves, refusals = renumber.plan_file(entries, repo=tmp_path)

    assert moves == []
    assert len(refusals) == 1
    assert "ambiguous" in refusals[0].detail
    assert "[3, 4]" in refusals[0].detail, "the reader is told which lines to choose between"


def test_a_refusal_carries_the_argument_a_reader_must_re_read(tmp_path: Path) -> None:
    """A refusal carries the entry's reason.

    The reason is what the reader has to judge, so the refusal includes it
    rather than leaving it to be looked up.
    """
    entries = _crafted(
        tmp_path, "a = 1\nif x <= y:\n", 2, argument="the branches agree at equal magnitude"
    )
    (tmp_path / "src" / "crafted.py").write_text("a = 1\nif x < y:\n", encoding="utf-8")

    _moves, refusals = renumber.plan_file(entries, repo=tmp_path)

    assert refusals[0].entry.reason == "the branches agree at equal magnitude"


def test_a_move_rewrites_only_the_position_field(tmp_path: Path) -> None:
    """A move rewrites the line field and nothing else.

    A renumbered allowlist then diffs as one number per entry rather than as a
    reformatting.
    """
    entries = _crafted(tmp_path, "a = 1\nif x <= y:\n", 2, argument="unchanged prose")
    (tmp_path / "src" / "crafted.py").write_text("# added\na = 1\nif x <= y:\n", encoding="utf-8")
    moves, _refusals = renumber.plan_file(entries, repo=tmp_path)

    before = entries[0].raw
    after = renumber.apply_moves(before, moves)

    assert after == before.replace("  2  ", "  3  ", 1)
    assert "unchanged prose" in after


def test_a_truncated_run_does_not_report_a_claim_it_never_reached() -> None:
    """A truncated run does not report an unreached claim as stale.

    `--max-mutants` stops the campaign early, so a claim past the cut is
    unmatched rather than stale, and reporting it would fail a smoke run on good
    entries. The C++ campaign's `partial` flag withholds the same failure.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._load_equivalents(module), "the premise: this module is documented"
    assert runner._verdict(module, [], partial=True) == 0


def test_a_complete_run_still_reports_the_same_claim_as_stale() -> None:
    """A complete run reports the same unmatched claim as stale.

    Withholding the failure on a truncated run must not withhold it on a full
    one.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")

    assert runner._verdict(module, [], partial=False) == 1


def test_a_truncated_run_still_fails_on_an_undocumented_survivor() -> None:
    """A truncated run still fails on an undocumented survivor.

    Truncation withholds one failure, not both; a survivor the run saw is
    evidence whatever the cut was.
    """
    module = Path("src/tfidf_stability/ranking/sort_keys.py")
    survivors = [_mutant(42, "constant", "1", "0")]

    assert runner._verdict(module, survivors, partial=True) == 1
