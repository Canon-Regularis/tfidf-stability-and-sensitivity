"""Tests for the report summary renderer.

`reports/summary.md` is committed; `--check` passes when it is current and
fails when it is not. The renderer copies numbers and computes none. The tests
read the committed reports, not a fixture, so they fail when those drift.
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
    """The committed summary matches a fresh render of the committed reports.

    A summary that disagreed with its reports would be a second set of numbers.
    """
    rendered = _renderer().render(REPORTS)

    assert (REPORTS / "summary.md").read_text(encoding="utf-8") == rendered


def test_rendering_twice_gives_the_same_bytes() -> None:
    """Two renders of the same reports produce identical text.

    The output depends on no clock, environment or dictionary order, so
    `--check` gives the same answer on every run.
    """
    renderer = _renderer()

    assert renderer.render(REPORTS) == renderer.render(REPORTS)


# ---------------------------------------------------------------------------
# It presents, and does not compute
# ---------------------------------------------------------------------------
def test_every_result_digest_reaches_the_summary() -> None:
    """Every report's result digest prefix appears in the summary.

    The digest ties each quoted number to the run that produced it.
    """
    rendered = _renderer().render(REPORTS)

    for name in ("similarity", "stability_profile", "tie_break_ablations"):
        record = json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))
        assert record["result_digest"][:16] in rendered, f"{name} is quoted without its digest"


def test_the_k_blocks_are_restored_to_numeric_order() -> None:
    """`_ks` orders k-block keys numerically rather than as strings.

    `canonical_json` sorts keys as strings (k1, k10, k20, k5, k50) and the
    digest depends on that sort, so the renderer restores numeric order.
    """
    renderer = _renderer()

    assert renderer._ks({"k50": 0, "k1": 0, "k10": 0, "k5": 0, "k20": 0}) == [
        "k1",
        "k5",
        "k10",
        "k20",
        "k50",
    ]


def test_the_margin_table_runs_k_upwards() -> None:
    """The rendered E1 margin table lists k in ascending numeric order.

    The check runs on the rendered output, so it fails if `_ks` is unused.
    """
    rendered = _renderer().render(REPORTS)
    table = rendered.split("### E1")[1].split("###")[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "|" in line]
    ks = [line.split("|")[1].strip() for line in rows[2:]]

    assert ks == sorted(ks, key=int), f"the k column is out of order: {ks}"


def test_an_ngram_is_rendered_so_a_reader_can_see_its_parts() -> None:
    """An n-gram renders as backticked parts joined by a visible plus sign.

    The joiner U+001F cannot occur inside a token and is invisible, so a raw
    bigram would read as one long token.
    """
    renderer = _renderer()

    assert renderer._term("alpha\x1fbeta") == "`alpha` + `beta`"
    assert "\x1f" not in renderer.render(REPORTS), "a raw joiner reached the document"


def test_a_column_no_row_fills_is_dropped() -> None:
    """A column that no row fills is dropped from the table.

    An always-empty cell reads as a missing value. The filled case shows the
    column is dropped for being empty, not for being last.
    """
    renderer = _renderer()

    empty = renderer._table(["a", ""], [["1", ""], ["2", ""]])
    filled = renderer._table(["a", ""], [["1", ""], ["2", "x"]])

    assert empty[0] == "| a |"
    assert "x" in filled[-1], "a column one row fills must survive"


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------
def test_a_missing_report_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    """A missing report is skipped and the remaining reports still render.

    `export_intermediates.py` is optional, so a checkout without its output
    still gets a summary of the reports it has.
    """
    renderer = _renderer()
    (tmp_path / "similarity.json").write_bytes((REPORTS / "similarity.json").read_bytes())

    rendered = renderer.render(tmp_path)

    assert "## Similarity" in rendered
    assert "## Stability profile" not in rendered


def test_no_reports_at_all_renders_a_document_that_says_so(tmp_path: Path) -> None:
    """An empty reports directory renders a document stating that none exist.

    An empty file or a traceback would read as a broken renderer.
    """
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
    """`--check` exits 1 on a stale summary and names the regenerating script."""
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
    """`--check` exits 1 on a missing summary and does not create the file.

    A check that wrote the file would pass on every run.
    """
    renderer = _renderer()
    (tmp_path / "similarity.json").write_bytes((REPORTS / "similarity.json").read_bytes())

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", ["render_summary.py", "--reports", str(tmp_path), "--check"])
        assert renderer.main() == 1
    assert not (tmp_path / "summary.md").exists()
