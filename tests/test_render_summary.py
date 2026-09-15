"""The report summary renderer.

`reports/summary.md` is generated and committed, so the thing that can go wrong
is the two drifting apart. `--check` is what catches that, and a check is only
evidence if it can fail, so both directions are asserted here.

The renderer's whole job is presentation: it must not introduce a number, and it
must put the ones it copies in an order the file itself cannot express. Both are
tested against the committed reports rather than against a fixture, because a
fixture would pass while the real artefact rotted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "render_summary.py"
REPORTS = REPO / "reports"


def _renderer() -> ModuleType:
    """Import ``scripts/render_summary.py``. Local by house convention."""
    spec = importlib.util.spec_from_file_location("_render_summary", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_render_summary"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The committed summary matches the committed reports
# ---------------------------------------------------------------------------
def test_the_committed_summary_is_current() -> None:
    """The baseline, and the reason the file is committed at all: a summary that
    disagreed with its reports would be a second set of numbers to trust."""
    rendered = _renderer().render(REPORTS)

    assert (REPORTS / "summary.md").read_text(encoding="utf-8") == rendered


def test_rendering_twice_gives_the_same_bytes() -> None:
    """No clock, no environment, no dictionary-order dependence. Without this
    `--check` would fail at random and stop being believed."""
    renderer = _renderer()

    assert renderer.render(REPORTS) == renderer.render(REPORTS)


# ---------------------------------------------------------------------------
# It presents, and does not compute
# ---------------------------------------------------------------------------
def test_every_result_digest_reaches_the_summary() -> None:
    """The digest is what ties a quoted number to the run behind it. A summary
    that dropped one would be quoting numbers with no provenance."""
    rendered = _renderer().render(REPORTS)

    for name in ("similarity", "stability_profile", "tie_break_ablations"):
        record = json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))
        assert record["result_digest"][:16] in rendered, f"{name} is quoted without its digest"


def test_the_k_blocks_are_restored_to_numeric_order() -> None:
    """`canonical_json` sorts keys as strings, so the file reads k1, k10, k20,
    k5, k50. The sort is what the digest rests on and cannot change, so the
    ordering is the renderer's job."""
    renderer = _renderer()

    assert renderer._ks({"k50": 0, "k1": 0, "k10": 0, "k5": 0, "k20": 0}) == [
        "k1",
        "k5",
        "k10",
        "k20",
        "k50",
    ]


def test_the_margin_table_runs_k_upwards() -> None:
    """The same claim against the real artefact, since `_ks` could be right and
    unused."""
    rendered = _renderer().render(REPORTS)
    table = rendered.split("### E1")[1].split("###")[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "|" in line]
    ks = [line.split("|")[1].strip() for line in rows[2:]]

    assert ks == sorted(ks, key=int), f"the k column is out of order: {ks}"


def test_an_ngram_is_rendered_so_a_reader_can_see_its_parts() -> None:
    """The joiner is U+001F, which cannot occur inside a token and also cannot be
    seen. Rendered raw, a bigram reads as one long token."""
    renderer = _renderer()

    assert renderer._term("alpha\x1fbeta") == "`alpha` + `beta`"
    assert "\x1f" not in renderer.render(REPORTS), "a raw joiner reached the document"


def test_a_column_no_row_fills_is_dropped() -> None:
    """An always-empty cell reads as a value that went missing. The contrasting
    case is what shows the column is dropped for being empty rather than last."""
    renderer = _renderer()

    empty = renderer._table(["a", ""], [["1", ""], ["2", ""]])
    filled = renderer._table(["a", ""], [["1", ""], ["2", "x"]])

    assert empty[0] == "| a |"
    assert "x" in filled[-1], "a column one row fills must survive"


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------
def test_a_missing_report_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    """`export_intermediates.py` is optional, and a checkout that has not run it
    should still get a summary of what it does have."""
    renderer = _renderer()
    (tmp_path / "similarity.json").write_bytes((REPORTS / "similarity.json").read_bytes())

    rendered = renderer.render(tmp_path)

    assert "## Similarity" in rendered
    assert "## Stability profile" not in rendered


def test_no_reports_at_all_renders_a_document_that_says_so(tmp_path: Path) -> None:
    """Rather than an empty file or a traceback, either of which reads as a
    broken renderer instead of an empty directory."""
    assert "No reports found." in _renderer().render(tmp_path)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_check_passes_on_a_summary_that_matches(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The live direction."""
    renderer = _renderer()
    for name in ("similarity.json", "stability_profile.json", "tie_break_ablations.json"):
        (tmp_path / name).write_bytes((REPORTS / name).read_bytes())
    (tmp_path / "summary.md").write_text(renderer.render(tmp_path), encoding="utf-8")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", ["render_summary.py", "--reports", str(tmp_path), "--check"])
        assert renderer.main() == 0
    assert "is current" in capsys.readouterr().out


def test_check_fails_on_a_summary_that_has_drifted(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The half that makes the gate worth running: a stale summary must fail,
    and must name the script that regenerates it."""
    renderer = _renderer()
    for name in ("similarity.json", "stability_profile.json"):
        (tmp_path / name).write_bytes((REPORTS / name).read_bytes())
    (tmp_path / "summary.md").write_text("# Experiment summary\n\nstale\n", encoding="utf-8")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", ["render_summary.py", "--reports", str(tmp_path), "--check"])
        assert renderer.main() == 1
    assert "render_summary.py" in capsys.readouterr().err


def test_check_fails_when_the_summary_is_absent_rather_than_writing_it(
    tmp_path: Path,
) -> None:
    """`--check` reports; it never repairs. A check that quietly wrote the file
    would pass on every run and prove nothing."""
    renderer = _renderer()
    (tmp_path / "similarity.json").write_bytes((REPORTS / "similarity.json").read_bytes())

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", ["render_summary.py", "--reports", str(tmp_path), "--check"])
        assert renderer.main() == 1
    assert not (tmp_path / "summary.md").exists()
