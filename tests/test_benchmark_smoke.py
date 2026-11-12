"""Structural checks on the benchmark harness at a smoke-test size.

No test asserts a duration; the checks are that every path is measured and a
speedup appears only beside a bit-identity check that fires on one ulp.
Rationale: docs/engineering_notes.md#benchmark-tests-assert-structure
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
    BenchmarkReport,
    BitIdentityError,
    Comparison,
    Timing,
    Workload,
    check_same_bits,
    check_same_order,
    format_report,
    measure,
    run_benchmarks,
)
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.io import canonical_json
from tfidf_stability.utils.numerics import Reduction

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
    """A forced reference-only run reports no native timings or speedups.

    The native backend is forced off even where it exists, so this path is
    exercised on every machine.
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
    """Each timing records the repeat count its minimum was taken over."""
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
    """Every native row carries a completed identity check beside its speedup."""
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
    """A one-ulp difference in a single value fails the bit-identity check.

    A tolerance-based comparison would accept a difference this small.
    """
    reference = [0.1, 0.2, 0.3]
    native = [0.1, math.nextafter(0.2, math.inf), 0.3]

    with pytest.raises(BitIdentityError, match="value 1 differs"):
        check_same_bits(reference, native, "scores")


def test_signed_zero_divergence_is_caught() -> None:
    """Signed zeros compare equal but differ in bit pattern, so the check fails."""
    with pytest.raises(BitIdentityError, match=r"!= native -0\.0"):
        check_same_bits([0.0], [-0.0], "scores")


def test_a_truncated_native_result_is_caught() -> None:
    """A native result with fewer values than the reference fails the check."""
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

    # The exact count makes the warm-up visible. Calibration times batches of
    # 1, 2, 4, ... inner (2 * inner - 1 calls), the timed phase adds
    # repeats * inner, and the warm-up adds one more. An inequality would leave
    # the warm-up unobserved.
    calibration = 2 * timing.inner - 1
    timed = timing.repeats * timing.inner
    assert calls == 1 + calibration + timed, (
        f"{calls} calls for inner={timing.inner}, repeats={timing.repeats}; "
        f"expected one warm-up plus {calibration} calibration and {timed} timed"
    )


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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


# The report is assembled from plain dataclasses rather than timed. A
# nanosecond operation, a multi-second one and a run that found a compiler are
# properties of the machine when timed and properties of the renderer when
# constructed.
def _timing(seconds: float, backend: str = "reference") -> Timing:
    return Timing(label="op", backend=backend, seconds=seconds, repeats=1, inner=1)


def _report(*comparisons: Comparison, **kw: object) -> BenchmarkReport:
    fields: dict[str, object] = {
        "workload": TINY,
        "repeats": 1,
        "comparisons": comparisons,
        "native": False,
        "native_reason": "forced off",
        "build": None,
        "n_features": 40,
        "nnz": 100,
    }
    fields.update(kw)
    return BenchmarkReport(**fields)  # type: ignore[arg-type]


def test_a_duration_is_rendered_in_the_unit_that_keeps_its_significant_digits() -> None:
    """Each duration is rendered in the unit that keeps its significant digits.

    A dot product is nanoseconds and a full fit is seconds; one unit for both
    would round the fast row to zero or push the slow row off the column.
    """
    text = format_report(
        _report(
            Comparison("nanosecond_op", _timing(5e-7), None, "checked"),
            Comparison("microsecond_op", _timing(5e-4), None, "checked"),
            Comparison("millisecond_op", _timing(0.5), None, "checked"),
            Comparison("second_op", _timing(2.0), None, "checked"),
        )
    )
    assert "500.0 ns" in text
    assert "500.00 us" in text
    assert "500.00 ms" in text
    assert "2.000 s" in text


def test_a_report_with_no_caveats_ends_at_the_table_rather_than_an_empty_heading() -> None:
    """A report with no notes ends at the table rather than an empty heading.

    The notes block is appended only when something was recorded.
    """
    text = format_report(_report(Comparison("op", _timing(1e-4), None, "checked")))
    assert "note" not in text
    assert text.rstrip().endswith("checked")


def test_a_native_run_names_the_compiler_and_whether_its_build_was_reproducible() -> None:
    """A native run's header names the compiler, build type and reproducibility.

    A fast-math build is faster and is not the same computation, so the header
    describes the build that produced the speedup.
    """
    build = {
        "compiler_id": "GNU",
        "compiler_ver": "13.2.0",
        "build_type": "Release",
        "reproducible": True,
    }
    text = format_report(
        _report(
            Comparison("op", _timing(1e-4), _timing(5e-5, "native"), "checked"),
            native=True,
            native_reason=None,
            build=build,
        )
    )
    assert "GNU 13.2.0" in text
    assert "Release" in text
    assert "reproducible = True" in text


def test_a_ranking_of_a_different_length_is_refused_before_any_speedup_is_reported() -> None:
    """Rankings of different lengths are refused before any speedup is reported.

    An element-by-element comparison would check only the shorter one.
    """
    with pytest.raises(BitIdentityError, match="Refusing to report a speedup"):
        check_same_order((0, 1, 2), (0, 1), "ranking")


# ---------------------------------------------------------------------------
# The speedup, and the workload it was measured on
# ---------------------------------------------------------------------------
def test_a_speedup_is_the_reference_cost_divided_by_the_native_one() -> None:
    """The speedup is the reference cost divided by the native cost.

    A product would still move in the right direction whenever both timings
    do, so nothing downstream would reveal it.
    """
    faster = Comparison("op", _timing(2.0), _timing(0.5, "native"), "checked")
    assert faster.speedup == 4.0

    slower = Comparison("op", _timing(0.5), _timing(2.0, "native"), "checked")
    assert slower.speedup == 0.25, "a slowdown is reported, not clamped"


def test_a_native_timing_of_zero_yields_no_speedup_rather_than_infinity() -> None:
    """A native timing of zero yields no speedup rather than infinity.

    A measured zero means the timer could not resolve the call. The guard is
    `<= 0`, not `< 0`, so the division does not happen.
    """
    assert Comparison("op", _timing(1.0), _timing(0.0, "native"), "checked").speedup is None
    assert Comparison("op", _timing(1.0), None, "checked").speedup is None


def test_a_tiny_workload_still_plants_a_duplicate_and_a_twin_pair() -> None:
    """A ten-document workload still plants one duplicate and one twin pair.

    The counts scale with corpus size and are floored at one: below twenty
    documents the scaling rounds to zero, and a corpus with no ties would
    leave the tie-break comparison nothing to measure.
    """
    spec = Workload(n_docs=10, vocab_size=40, n_queries=2, query_length=3, k=2).spec()
    assert spec.n_exact_duplicates == 1
    assert spec.n_twin_pairs == 1
    assert spec.n_users == 0, "interactions belong to the profile experiments"

    bigger = Workload(n_docs=100, vocab_size=200, n_queries=2, query_length=3, k=2).spec()
    assert bigger.n_exact_duplicates == 5, "and it scales above the floor"


def test_the_identity_check_counts_what_it_compared_in_words() -> None:
    """The identity check reports how many values it compared, in words.

    The string is the `checked` column beside every ratio.
    """
    assert check_same_bits((1.0,), (1.0,), "idf") == "1 value bit-identical"
    assert check_same_bits((1.0, 2.0), (1.0, 2.0), "idf") == "2 values bit-identical"


@pytest.mark.parametrize(
    ("seconds", "rendered", "if_it_slipped"),
    [
        (1e-6, "1.00 us", "1000.0 ns"),
        (1e-3, "1.00 ms", "1000.00 us"),
        (1.0, "1.000 s", "1000.00 ms"),
    ],
)
def test_a_duration_exactly_on_a_unit_boundary_uses_the_larger_unit(
    seconds: float, rendered: str, if_it_slipped: str
) -> None:
    """A duration exactly on a unit boundary is rendered in the larger unit.

    The thresholds are strict, so 1e-6 seconds is one microsecond. The whole
    rendered value is matched, not the unit alone: the table header contains
    "speedup", so a bare " s" appears in every report.
    """
    text = format_report(_report(Comparison("op", _timing(seconds), None, "checked")))
    assert rendered in text
    assert if_it_slipped not in text


def test_a_native_run_that_reported_no_build_says_so_instead_of_crashing() -> None:
    """A native run with no build info renders "unavailable" in the header.

    The build line requires both `native` and a non-None `build`, so a report
    with `native=True` and `build=None` falls back to the unavailable line.
    """
    text = format_report(_report(Comparison("op", _timing(1e-4), None, "checked"), native=True))
    assert "unavailable" in text


def test_a_measured_cost_is_a_duration_and_not_a_clock_reading() -> None:
    """A measured cost is a duration, not a clock reading.

    `perf_counter` has an arbitrary origin, so a batch timed as `now + start`
    reports thousands of seconds. The bound here is not a performance
    threshold: a no-op taking over a second is broken by any reading.
    """
    timing = measure(lambda: None, label="noop", backend="reference", repeats=1)

    assert 0.0 <= timing.seconds < 1.0, f"a no-op measured {timing.seconds}s"
    assert math.isfinite(timing.seconds)
    assert timing.inner >= 1
    assert timing.calls == timing.repeats * timing.inner


def test_the_reported_cost_is_per_call_not_per_batch() -> None:
    """The reported cost is per call, not per batch.

    Calibration grows the batch until it clears the minimum batch time, so a
    batch holds many calls. Reporting the batch cost inflates the per-call
    figure by a factor that differs per operation.
    """
    timing = measure(lambda: None, label="noop", backend="reference", repeats=1)

    # A no-op is far below the batch floor, so calibration must have grown the
    # batch; the per-call cost is then a small fraction of one batch.
    assert timing.inner > 1, "the premise: calibration actually scaled the batch up"
    assert timing.seconds < 1e-3, "a no-op costs nanoseconds per call, not milliseconds"


# ---------------------------------------------------------------------------
# The batch size is capped, so a fast callable cannot run away with the clock
# ---------------------------------------------------------------------------
def test_the_inner_batch_stops_doubling_at_the_cap(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The inner batch stops doubling at `_MAX_INNER`.

    The loop grows the batch until it takes long enough to time; the cap is
    the other exit. Both constants are lowered here: at the real cap of 2^22
    the test would make four million calls.
    """
    from tfidf_stability.benchmarks import tfidf_perf

    monkeypatch.setattr(tfidf_perf, "_MAX_INNER", 8)
    # No batch of any size reaches this, so the cap is the only way out.
    monkeypatch.setattr(tfidf_perf, "_MIN_BATCH_SECONDS", 1e9)

    timing = tfidf_perf.measure(lambda: None, label="noop", backend="reference", repeats=1)

    assert timing.inner == 8, "the last doubling that satisfies inner < cap lands on the cap"


def test_a_callable_slow_enough_to_time_is_not_batched_at_all(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A callable that already exceeds the threshold is not batched.

    This is the other exit from the same loop; batching a slow call would
    multiply the measurement's cost for nothing.
    """
    from tfidf_stability.benchmarks import tfidf_perf

    monkeypatch.setattr(tfidf_perf, "_MIN_BATCH_SECONDS", 0.0)

    timing = tfidf_perf.measure(lambda: None, label="noop", backend="reference", repeats=1)

    assert timing.inner == 1


def test_the_reported_cost_is_divided_by_the_batch_size(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The reported cost is `best / inner`, the per-call figure.

    Two runs that chose different batch sizes stay comparable, and `inner` is
    recorded beside the number.
    """
    from tfidf_stability.benchmarks import tfidf_perf

    monkeypatch.setattr(tfidf_perf, "_MAX_INNER", 4)
    monkeypatch.setattr(tfidf_perf, "_MIN_BATCH_SECONDS", 1e9)

    timing = tfidf_perf.measure(lambda: None, label="noop", backend="reference", repeats=1)

    assert timing.inner == 4
    assert timing.seconds >= 0.0
    assert timing.seconds < 1e9, "a per-call cost, not the batch's total"


# ---------------------------------------------------------------------------
# "reference only" is two different statements, and the report keeps them apart
# ---------------------------------------------------------------------------
def test_an_operation_with_no_native_counterpart_says_so_in_its_own_words() -> None:
    """An operation with no native counterpart names that reason in its row.

    Fitting has no native side: idf is evaluated once in exact decimal
    arithmetic so the core never sees a logarithm (G13). The generic phrase
    would read as "the backend was not built here".
    """
    report = run_benchmarks(TINY, repeats=1, use_native=False)
    fit = next(c for c in report.comparisons if c.name.startswith("fit"))

    assert fit.verified == "no native counterpart exists"
    assert fit.native is None
    assert "G13" in fit.note, "and the note says why"


def test_an_operation_whose_backend_was_simply_absent_uses_the_generic_phrase() -> None:
    """An operation whose backend was not exercised uses the generic phrase.

    Building the index has a native counterpart; a reference-only run skips
    it, so the reason is about the run rather than the operation.
    """
    report = run_benchmarks(TINY, repeats=1, use_native=False)
    index = next(c for c in report.comparisons if c.name.startswith("build index"))

    assert index.verified == "reference only"
    assert index.native is None


def test_no_reference_only_comparison_leaves_the_reason_blank() -> None:
    """Every reference-only row carries a non-empty reason.

    An empty cell would be indistinguishable from a skipped check.
    """
    report = run_benchmarks(TINY, repeats=1, use_native=False)

    assert all(c.verified for c in report.comparisons)
    assert all(c.as_dict()["verified"] for c in report.comparisons)


# ---------------------------------------------------------------------------
# Fixture construction: what both backends are handed, built once and never timed
# ---------------------------------------------------------------------------


# Reached directly, as `test_datasets.py` reaches the generator's deterministic
# primitives. Nothing the fixture gets wrong shows up in a report: the numbers
# are timings, and a fixture that timed the wrong thing would still produce a
# plausible table.
def test_queries_are_drawn_across_the_corpus_rather_than_all_from_one_document() -> None:
    """Queries are drawn across the corpus, capped at one per document.

    The stride is `max(1, len(features) // max(1, n_queries))`. Without the
    outer clamp, more queries than documents gives a stride of zero, and every
    query is `features[0]`, so the scoring row times the same query repeatedly.
    """
    from tfidf_stability.benchmarks.tfidf_perf import _build_fixture

    crowded = Workload(n_docs=24, vocab_size=40, n_queries=30, query_length=4, k=3, seed=20260811)
    fixture = _build_fixture(crowded)

    assert len(fixture.queries) == 24, "capped at one query per document"
    distinct = {(q.indices, q.values) for q in fixture.queries}
    assert len(distinct) > 1, "a stride of zero would make every query the first document's"


def test_a_query_set_smaller_than_the_corpus_still_spreads_over_it() -> None:
    """A query set smaller than the corpus still spreads over it.

    This is the ordinary arm of the stride expression.
    """
    from tfidf_stability.benchmarks.tfidf_perf import _build_fixture

    fixture = _build_fixture(TINY)

    assert len(fixture.queries) == TINY.n_queries
    assert len({(q.indices, q.values) for q in fixture.queries}) > 1


def test_the_ranking_rows_are_scored_from_the_first_query() -> None:
    """The fixture's scores are those of the first query against the corpus.

    The ranking and top-k rows are timed on this vector, and the field's
    comment names it; the margin structure those rows exercise belongs to
    this query.
    """
    from tfidf_stability.benchmarks.tfidf_perf import _build_fixture

    fixture = _build_fixture(TINY)
    expected = cosine_against_corpus(
        fixture.queries[0], fixture.documents, fixture.model.norms, Reduction.NAIVE
    )

    assert fixture.scores == expected
    other = cosine_against_corpus(
        fixture.queries[1], fixture.documents, fixture.model.norms, Reduction.NAIVE
    )
    assert fixture.scores != other, "the two queries do score differently"


@pytest.mark.parametrize(
    ("label", "native_call", "verify"),
    [
        ("neither half", None, None),
        ("a call with nothing to check it", lambda: None, None),
        ("a check with nothing to run", None, lambda: "checked"),
    ],
)
def test_a_comparison_missing_either_half_of_the_native_pair_is_reference_only(
    label: str,
    native_call: object,
    verify: object,
) -> None:
    """A comparison missing either half of the native pair is reference-only.

    An unverified native timing is a speedup on a possibly divergent result,
    and a verifier with nothing to run reports a check that never happened.
    With `and` for `or`, a half pair calls its missing half.
    """
    from tfidf_stability.benchmarks.tfidf_perf import _compare

    comparison = _compare(
        "half a pair",
        reference_call=lambda: None,
        native_call=native_call,  # type: ignore[arg-type]
        verify=verify,  # type: ignore[arg-type]
        repeats=1,
    )

    assert comparison.native is None
    assert comparison.speedup is None
    assert comparison.verified == "reference only"


def test_a_batch_that_exactly_meets_the_threshold_is_long_enough(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A batch that exactly meets `_MIN_BATCH_SECONDS` is long enough.

    The comparison is `<`, so a batch landing on the threshold is not doubled.
    The timer is replaced because a real reading equal to the threshold to
    the last bit cannot be arranged.
    """
    from tfidf_stability.benchmarks import tfidf_perf

    monkeypatch.setattr(tfidf_perf, "_MAX_INNER", 8)
    monkeypatch.setattr(tfidf_perf, "_time_batch", lambda fn, inner: tfidf_perf._MIN_BATCH_SECONDS)

    timing = tfidf_perf.measure(lambda: None, label="exact", backend="reference", repeats=1)

    assert timing.inner == 1, "the first batch already met the threshold"
    assert timing.seconds == tfidf_perf._MIN_BATCH_SECONDS


def test_a_batch_just_under_the_threshold_is_doubled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A batch one ulp under the threshold is doubled up to the cap.

    This is the other side of the same comparison.
    """
    from tfidf_stability.benchmarks import tfidf_perf

    just_under = math.nextafter(tfidf_perf._MIN_BATCH_SECONDS, 0.0)
    monkeypatch.setattr(tfidf_perf, "_MAX_INNER", 8)
    monkeypatch.setattr(tfidf_perf, "_time_batch", lambda fn, inner: just_under)

    timing = tfidf_perf.measure(lambda: None, label="under", backend="reference", repeats=1)

    assert timing.inner == 8, "never satisfied, so it grows to the cap"


def test_a_workload_asking_for_no_queries_is_not_validated_and_fails_late() -> None:
    """A workload with `n_queries=0` is not validated and fails late.

    The stride's inner `max(1, n_queries)` keeps the floor division alive, so
    construction reaches `queries[0]` in the `scores` field and raises a bare
    `IndexError`. The behaviour is pinned; the failure still stops the run.
    """
    empty = Workload(n_docs=24, vocab_size=40, n_queries=0, query_length=4, k=3, seed=20260811)

    from tfidf_stability.benchmarks.tfidf_perf import _build_fixture

    with pytest.raises(IndexError, match="list index out of range"):
        _build_fixture(empty)


@pytest.mark.skipif(not native_available(), reason=unavailable_reason() or "no native backend")
def test_the_index_row_verifies_the_construction_it_times() -> None:
    """The index row verifies the index it times, not one built elsewhere.

    Only the timed expression has to be right for the ratio to mean anything.
    The test asserts the check covers one norm per document rather than a
    stale or partial set, and that the row keeps its scope-mismatch note.
    """
    report = run_benchmarks(TINY, repeats=1)
    index = next(c for c in report.comparisons if c.name.startswith("build index"))

    assert index.native is not None, "the native backend is available in this run"
    assert index.verified == f"{TINY.n_docs} values bit-identical"
    assert "not like-for-like" in index.note, "and the row still declares its scope mismatch"
