"""Tests for the documentation reference checker.

``scripts/check_docs.py`` gates CI. Its docstring half resolves runtime
objects, so the rejections are asserted here and a resolver that accepts
every name fails this file rather than passing the gate vacuously.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_docs.py"


def _checker() -> ModuleType:
    """Import ``scripts/check_docs.py`` as a module.

    The helper is local to this file by house convention.
    """
    spec = importlib.util.spec_from_file_location("_check_docs_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Normal: the tree is clean, and the checker says so in numbers
# ---------------------------------------------------------------------------
def test_the_repository_currently_passes_its_own_documentation_check() -> None:
    """The current tree passes the check; the rejections below depend on it."""
    assert _checker().check() == []


def test_the_docstring_half_actually_examined_something() -> None:
    """The docstring scan finds a large number of cross-references.

    A scan that found nothing would pass as quietly as one that checked
    every reference, and the modules cite each other densely enough that
    the count cannot be small.
    """
    checker = _checker()
    _problems, n_refs = checker.check_docstrings({})

    assert n_refs > 150, f"only {n_refs} cross-references found; the scan missed files"


# ---------------------------------------------------------------------------
# The resolver: the five places a cross-reference target may live
# ---------------------------------------------------------------------------
def test_a_target_naming_nothing_anywhere_is_refused() -> None:
    """A target that exists nowhere is rejected.

    A resolver that fell through to True would report a clean tree for
    every input.
    """
    checker = _checker()
    from tfidf_stability.ranking import margins

    assert not checker._resolves("no_such_name_at_all", margins, None, {})


@pytest.mark.parametrize(
    ("target", "why"),
    [
        ("boundary_margin", "a function in the module itself"),
        ("Margin", "a class in the module itself"),
        ("Margin.flip_radius", "an attribute reached through that class"),
        ("AssertionError", "a builtin"),
        ("math.isnan", "the standard library, by dotted path"),
        ("tfidf_stability.utils.numerics.same_bits", "the package, fully qualified"),
        ("AttributeTable", "a sibling module's export, never imported here"),
    ],
)
def test_each_place_a_cross_reference_may_point_is_resolved(target: str, why: str) -> None:
    """Each form a docstring cross-reference takes resolves.

    A resolver that handled only the local module would reject valid
    references.
    """
    checker = _checker()
    from tfidf_stability.ranking import margins

    exported = {
        "AttributeTable": __import__(
            "tfidf_stability.ranking.attributes", fromlist=["AttributeTable"]
        ).AttributeTable
    }

    assert checker._resolves(target, margins, None, exported), why


def test_an_attribute_of_the_enclosing_class_resolves_without_qualification() -> None:
    """An unqualified attribute name resolves against the enclosing class.

    A method's docstring writes ``:attr:`is_conclusive``` for its own class's
    attribute; without the owner in scope the name is a module-level lookup
    and is rejected.
    """
    checker = _checker()
    from tfidf_stability.analysis import stability_profile
    from tfidf_stability.analysis.stability_profile import CertificateAudit

    assert checker._resolves("is_conclusive", stability_profile, CertificateAudit, {})
    assert not checker._resolves("is_conclusive", stability_profile, None, {})


# ---------------------------------------------------------------------------
# Citing a test: read, never imported
# ---------------------------------------------------------------------------
def test_a_cited_test_that_exists_resolves() -> None:
    """A cited test that exists in the named module resolves.

    `ranking/distances.py` cites the test pinning its near-metric
    counterexample. The lookup reads the file, so the checker never imports
    a test module and its plugins.
    """
    assert _checker()._defined_in_tests(
        "test_ordering_distances", "test_fks_is_a_near_metric_not_a_metric"
    )


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("test_ordering_distances", "test_this_name_was_renamed_away"),
        ("test_module_that_does_not_exist", "anything"),
    ],
)
def test_a_cited_test_that_does_not_exist_is_refused(module: str, name: str) -> None:
    """A missing module and a missing function in a real module are refused."""
    assert not _checker()._defined_in_tests(module, name)


# ---------------------------------------------------------------------------
# Anchors into the addenda
# ---------------------------------------------------------------------------
def test_every_g_number_in_the_addenda_is_reachable_as_an_anchor() -> None:
    """Every G-number heading in the addenda is reachable as an anchor.

    Docstrings cite ``spec_addenda.md#g13`` and the like. The anchor form is
    the lowercased first word of the heading, so a renamed heading breaks
    every citation of it.
    """
    checker = _checker()
    anchors = checker._anchors((REPO / "docs" / "spec_addenda.md").read_text(encoding="utf-8"))

    assert "g13" in anchors, "the correctly-rounded logarithm addendum"
    assert "g3" in anchors, "the master edge-case table"
    assert "g999" not in anchors, "the premise: unknown anchors really are absent"


# ---------------------------------------------------------------------------
# The build presets
# ---------------------------------------------------------------------------
def _presets_file(tmp_path: Path, names: list[str], hidden: Sequence[str] = ()) -> Path:
    """A minimal ``CMakePresets.json`` holding just the names given."""
    path = tmp_path / "CMakePresets.json"
    entries = [{"name": n} for n in names] + [{"name": n, "hidden": True} for n in hidden]
    path.write_text(json.dumps({"configurePresets": entries}), encoding="utf-8")
    return path


def _document(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "index.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_preset_half_actually_examined_something() -> None:
    """The preset scan finds the project's presets.

    A checker that found no presets would pass as quietly as one that checked
    them all, and the project defines a dozen.
    """
    checker = _checker()
    _problems, n_presets = checker.check_presets(sorted((REPO / "docs").glob("*.md")))

    assert n_presets > 5, f"only {n_presets} presets found; the scan missed the file"


def test_a_preset_that_no_document_names_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset that no document names is a finding.

    A reader cannot invoke a preset the documentation does not mention.
    """
    checker = _checker()
    monkeypatch.setattr(checker, "PRESETS", _presets_file(tmp_path, ["ci", "bench"]))

    problems, _n = checker.check_presets([_document(tmp_path, "Build with `--preset ci`.\n")])

    assert problems == ["CMakePresets.json: preset `bench` is named in no document"]


def test_a_documented_preset_that_does_not_exist_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A documented preset that does not exist is a finding.

    A renamed preset leaves a command in the documents that configures
    nothing.
    """
    checker = _checker()
    monkeypatch.setattr(checker, "PRESETS", _presets_file(tmp_path, ["ci"]))

    problems, _n = checker.check_presets(
        [_document(tmp_path, "Run `--preset ci`, then `cmake --preset gone`.\n")]
    )

    assert problems == ["index.md: --preset gone names no preset in CMakePresets.json"]


def test_a_hidden_preset_need_not_be_documented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hidden preset is neither required in the documents nor counted.

    A hidden preset exists to be inherited from and cannot be invoked, so
    requiring it in the documents would demand prose about nothing.
    """
    checker = _checker()
    monkeypatch.setattr(checker, "PRESETS", _presets_file(tmp_path, ["ci"], hidden=["base"]))

    problems, n_presets = checker.check_presets([_document(tmp_path, "Use `--preset ci`.\n")])

    assert problems == []
    assert n_presets == 1, "and it is not counted as checked either"


def test_an_absent_presets_file_is_reported_rather_than_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent presets file is a finding rather than an exception.

    The gate runs inside the mutation sandbox, which copies a chosen list of
    paths rather than the whole tree. A finding names the file to add, where
    a `FileNotFoundError` from a nightly job names only the exception.
    """
    checker = _checker()
    monkeypatch.setattr(checker, "PRESETS", tmp_path / "CMakePresets.json")

    problems, n_presets = checker.check_presets([_document(tmp_path, "Use `--preset ci`.\n")])

    assert n_presets == 0
    assert problems == ["CMakePresets.json not found, so no preset could be checked"]
    assert checker.check() != [], "and the gate still fails, rather than passing vacuously"


def test_a_preset_named_only_in_prose_does_not_count_as_documented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset named only in prose does not count as documented.

    The name has to appear as code; otherwise a preset called `debug` or
    `bench` would read as documented by any sentence that used the word.
    """
    checker = _checker()
    monkeypatch.setattr(checker, "PRESETS", _presets_file(tmp_path, ["bench"]))

    prose = _document(tmp_path, "The bench numbers are not reproducible.\n")
    quoted = _document(tmp_path / "sub", "Use `--preset bench`.\n")

    assert checker.check_presets([prose])[0] == [
        "CMakePresets.json: preset `bench` is named in no document"
    ]
    assert checker.check_presets([quoted])[0] == []
