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

import hashlib
import importlib.util
import re
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


def test_every_file_that_states_a_version_is_one_the_gate_reads() -> None:
    r"""The converse of the test above: a source the gate never learned about.

    That one iterates `gate._SOURCES` and asserts each path exists, so the list
    under test supplies its own expectation: it catches a source that was renamed
    away and cannot catch one that was never added. Here the expectation comes
    from the tree instead, by searching for the shape of a version declaration.
    `CMakeLists.txt` states one, and its `VERSION` becomes `PROJECT_VERSION`,
    then `kVersion`, then the native module's `__version__` and
    `build_info()["version"]`, which is hashed into every RunManifest.
    """
    gate = _script("check_versions")
    known = set(gate._SOURCES)

    # Files that state THIS project's version, by the forms it actually uses.
    candidates: dict[str, re.Pattern[str]] = {
        "pyproject.toml": re.compile(r'^version\s*=\s*"[0-9]'),
        "CITATION.cff": re.compile(r"^version:\s*[\"']?[0-9]"),
        "CMakeLists.txt": re.compile(r"^VERSION\s+[0-9]"),
        "src/tfidf_stability/__init__.py": re.compile(r'^__version__\s*[:=].*"[0-9]'),
    }

    stating: set[str] = set()
    for relative, pattern in candidates.items():
        path = REPO / relative
        if not path.exists():
            continue
        if any(
            pattern.match(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
        ):
            stating.add(relative)

    assert stating, "the premise: at least one file states a version"
    missing = stating - known
    assert not missing, (
        f"these files state a version the gate does not read: {sorted(missing)}. "
        f"A version stated in a file nobody compares is how CITATION.cff came to "
        f"disagree in the first place."
    )


def test_the_version_gate_reads_every_place_its_docstring_claims() -> None:
    """The count in the gate's docstring and the length of `_SOURCES` agree.

    A table shorter than the count its own prose advertises hides a missing
    source, so the two are compared in both directions.
    """
    gate = _script("check_versions")
    doc = gate.__doc__ or ""

    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    claimed = next((n for word, n in words.items() if f"{word} files name it" in doc.lower()), None)

    assert claimed is not None, (
        "the docstring no longer states how many files name the version; it did, "
        "and the number is what made the omission visible"
    )
    assert claimed == len(gate._SOURCES), (
        f"the docstring claims {claimed} files name the version but the gate "
        f"reads {len(gate._SOURCES)}: {sorted(gate._SOURCES)}"
    )


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


# ---------------------------------------------------------------------------
# check_vendored.py
# ---------------------------------------------------------------------------
# The gate has two halves: a digest comparison against each MANIFEST.sha256, and
# a scan for files no manifest lists. Both are exercised below.
def _vendored_tree(root: Path, files: dict[str, str]) -> Path:
    """A directory with a MANIFEST.sha256 naming `files`, digests computed.

    Local by house convention, and synthetic rather than pointed at the real
    `cpp/third_party`, so a test can add and corrupt files without touching a
    vendored byte.
    """
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for relative, content in sorted(files.items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
        lines.append(f"{hashlib.sha256(content.encode('utf-8')).hexdigest()}  {relative}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _vendored_gate(monkeypatch: pytest.MonkeyPatch, repo: Path) -> ModuleType:
    gate = _script("check_vendored")
    monkeypatch.setattr(gate, "REPO", repo)
    return gate


def test_the_vendored_gate_passes_on_this_repository() -> None:
    """The baseline. Every rejection below is vacuous if this fails."""
    assert _script("check_vendored").main() == 0


def test_a_changed_vendored_byte_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest comparison: a listed file whose bytes no longer match."""
    _vendored_tree(tmp_path / "vendor", {"lib/header.h": "// upstream\n"})
    gate = _vendored_gate(monkeypatch, tmp_path)

    assert gate.check() == [], "the premise: the tree verifies before it is touched"

    (tmp_path / "vendor" / "lib" / "header.h").write_text("// edited\n", encoding="utf-8")

    problems = gate.check()
    assert problems, "a changed byte must be reported"
    assert "digest changed" in problems[0]


def test_a_file_added_inside_a_vendored_subdirectory_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unlisted-file scan, which must descend below the manifest's own level.

    Every file `cpp/third_party` vendors sits one level down, in `doctest/` and
    `nanobench/`, so a scan skipping directories is inert on the real tree. An
    unlisted file has no recorded digest, so nothing compares it to anything: it
    ships with the provenance claim of the directory it sits in and none of its
    own.
    """
    _vendored_tree(tmp_path / "vendor", {"lib/header.h": "// upstream\n"})
    gate = _vendored_gate(monkeypatch, tmp_path)

    (tmp_path / "vendor" / "lib" / "extra.h").write_text("// unlisted\n", encoding="utf-8")

    problems = gate.check()
    assert problems, "a file added inside a vendored subdirectory must be reported"
    assert "absent from MANIFEST.sha256" in problems[0]


def test_an_entire_unlisted_vendored_library_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same scan at directory scale: a whole unlisted dependency.

    The test above adds one file inside a directory the manifest already covers;
    this one adds a directory no manifest names at all.
    """
    _vendored_tree(tmp_path / "vendor", {"lib/header.h": "// upstream\n"})
    gate = _vendored_gate(monkeypatch, tmp_path)

    (tmp_path / "vendor" / "catch2").mkdir()
    (tmp_path / "vendor" / "catch2" / "catch.hpp").write_text("// vendored\n", encoding="utf-8")

    assert gate.check(), "an unlisted vendored library must be reported"


def test_a_listed_file_that_is_missing_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction: the manifest names something that is not there."""
    _vendored_tree(tmp_path / "vendor", {"lib/header.h": "// upstream\n"})
    gate = _vendored_gate(monkeypatch, tmp_path)

    (tmp_path / "vendor" / "lib" / "header.h").unlink()

    problems = gate.check()
    assert problems
    assert "listed but missing" in problems[0]


def test_a_tree_with_no_manifest_at_all_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate that finds nothing to check must say so rather than pass.

    Deleting every MANIFEST.sha256 would otherwise leave the gate green: zero
    files verified across zero manifests, reported as success.
    """
    gate = _vendored_gate(monkeypatch, tmp_path)

    problems = gate.check()
    assert problems == ["no MANIFEST.sha256 found anywhere -- vendoring is unverified"]
