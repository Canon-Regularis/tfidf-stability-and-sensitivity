"""The benchmark harness, at a size where it cannot be slow.

A benchmark prints numbers whatever happens, so one that has stopped comparing
the two backends still looks healthy. The target is the bit-identity guard
between a speedup and a wrong answer.

Nothing asserts a duration: wall-clock time is a property of the machine, so a
threshold flakes on a loaded runner and says nothing about the code. Structure is
asserted instead: every path measured, reference-only mode working, speedups
reported only beside a completed identity check, and the check firing on a
one-ulp divergence.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tfidf_stability._native import native_available, unavailable_reason
from tfidf_stability.benchmarks.tfidf_perf import (
    BitIdentityError,
    Workload,
    check_same_bits,
    check_same_order,
    format_report,
    measure,
    run_benchmarks,
)
from tfidf_stability.utils.io import canonical_json

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "benchmark.py"

#: Small enough for a full run in a second or two, large enough that the
#: generator still emits exact duplicates and twin pairs: over distinctly scored
#: documents the ranking rows never reach the tie-break.
TINY = Workload(n_docs=24, vocab_size=40, n_queries=2, query_length=4, k=3, seed=20260811)


def _names(report: object) -> list[str]:
    return [c.name for c in report.comparisons]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------
def test_reference_only_run_is_complete() -> None:
    """The no-compiler configuration, forced.

    Runs even where the native backend exists; otherwise this path is exercised
    only on machines that cannot report a failure in it.
    """
    report = run_benchmarks(TINY, repeats=1, use_native=False)

    assert report.native is False
    assert report.native_reason
    assert report.build is None
    for comparison in report.comparisons:
        assert comparison.native is None
        assert comparison.speedup is None
        assert comparison.reference.seconds > 0.0


def test_every_path_that_matters_is_measured() -> None:
    names = " | ".join(_names(run_benchmarks(TINY, repeats=1, use_native=False)))

    assert "fit" in names
    assert "build index" in names
    assert "TAAT" in names
    assert "DAAT" in names
    for policy in ("naive", "neumaier", "pairwise", "exact"):
        assert f"({policy})" in names, f"reduction policy {policy} is not benchmarked"
    assert "rank" in names
    assert "top-3" in names


def test_timings_report_how_many_repeats_they_took_the_minimum_of() -> None:
    """A minimum without its sample size is not a measurement."""
    report = run_benchmarks(TINY, repeats=2, use_native=False)

    assert report.repeats == 2
    for comparison in report.comparisons:
        assert comparison.reference.repeats == 2
        assert comparison.reference.inner >= 1
        assert comparison.reference.calls == 2 * comparison.reference.inner


def test_report_survives_canonical_json() -> None:
    """The runner writes the report to disk, so it has to be serialisable."""
    report = run_benchmarks(TINY, repeats=1, use_native=False)
    payload = json.loads(canonical_json(report.as_dict()))

    assert payload["workload"]["n_docs"] == TINY.n_docs
    assert payload["repeats"] == 1
    assert all("verified" in c for c in payload["comparisons"])


def test_formatted_report_shows_the_check_beside_the_speedup() -> None:
    text = format_report(run_benchmarks(TINY, repeats=1, use_native=False))

    assert "workload" in text
    assert "minimum of 1 batches" in text
    assert "checked" in text


@pytest.mark.native
@pytest.mark.differential
@pytest.mark.skipif(not native_available(), reason=unavailable_reason() or "no native backend")
def test_native_speedups_are_reported_only_with_a_completed_check() -> None:
    """Every native row must carry the evidence licensing its ratio."""
    report = run_benchmarks(TINY, repeats=1)

    assert report.native is True
    assert report.build is not None
    assert report.build["fast_math"] is False, "a fast-math build must never be timed as equivalent"

    compared = [c for c in report.comparisons if c.native is not None]
    assert len(compared) >= 8, "the native backend covers more paths than were compared"
    for comparison in compared:
        assert comparison.native is not None
        assert comparison.native.seconds > 0.0
        assert comparison.speedup is not None
        assert comparison.speedup > 0.0
        assert "identical" in comparison.verified, comparison.name


# ---------------------------------------------------------------------------
# Guarding the guard
# ---------------------------------------------------------------------------
def test_a_one_ulp_divergence_is_caught() -> None:
    """The smallest possible disagreement, which a tolerance would wave through."""
    reference = [0.1, 0.2, 0.3]
    native = [0.1, math.nextafter(0.2, math.inf), 0.3]

    with pytest.raises(BitIdentityError, match="value 1 differs"):
        check_same_bits(reference, native, "scores")


def test_signed_zero_divergence_is_caught() -> None:
    """``-0.0 == 0.0`` is true and their bit patterns differ; bits win."""
    with pytest.raises(BitIdentityError):
        check_same_bits([0.0], [-0.0], "scores")


def test_a_truncated_native_result_is_caught() -> None:
    """A backend that scored fewer documents must not be timed as a winner."""
    with pytest.raises(BitIdentityError, match="3 values but native produced 2"):
        check_same_bits([1.0, 2.0, 3.0], [1.0, 2.0], "scores")


def test_a_transposed_permutation_is_caught() -> None:
    with pytest.raises(BitIdentityError, match="position 1 differs"):
        check_same_order((3, 1, 2), (3, 2, 1), "ranking")


def test_identical_results_pass_and_say_what_was_checked() -> None:
    assert check_same_bits([1.0, 2.0], [1.0, 2.0], "scores") == "2 values bit-identical"
    assert check_same_order((0, 1), (0, 1), "ranking").startswith("permutation of 2")


# ---------------------------------------------------------------------------
# The measurement primitive
# ---------------------------------------------------------------------------
def test_measure_runs_a_warm_up_and_every_requested_batch() -> None:
    """Counts calls rather than timing them; only the count is deterministic."""
    calls = 0

    def counted() -> None:
        nonlocal calls
        calls += 1

    timing = measure(counted, label="counted", backend="reference", repeats=3)

    assert timing.repeats == 3
    assert timing.seconds > 0.0
    # One warm-up, an unknown number of calibration batches, then the timed ones.
    assert calls > 1 + 3 * timing.inner


def test_measure_rejects_a_meaningless_repeat_count() -> None:
    with pytest.raises(ValueError, match="repeats"):
        measure(lambda: None, label="x", backend="reference", repeats=0)


# ---------------------------------------------------------------------------
# The runner script
# ---------------------------------------------------------------------------
def _load_script() -> ModuleType:
    """Import ``scripts/benchmark.py`` as a module, without spawning a process."""
    spec = importlib.util.spec_from_file_location("_benchmark_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_script_runs_and_writes_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Covers the wiring the library tests cannot: argument parsing and output."""
    destination = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--docs",
            str(TINY.n_docs),
            "--vocab",
            str(TINY.vocab_size),
            "--queries",
            str(TINY.n_queries),
            "--query-length",
            str(TINY.query_length),
            "-k",
            str(TINY.k),
            "--repeats",
            "1",
            "--reference-only",
            "--json",
            str(destination),
        ],
    )

    assert _load_script().main() == 0
    assert "workload" in capsys.readouterr().out
    assert json.loads(destination.read_text(encoding="utf-8"))["workload"]["n_docs"] == TINY.n_docs
