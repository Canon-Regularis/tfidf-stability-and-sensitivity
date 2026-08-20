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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
# The report is assembled from plain dataclasses rather than measured, because
# the cases that matter here -- a nanosecond operation, a multi-second one, a run
# that found a compiler -- are properties of the machine when they are measured
# and properties of the renderer when they are constructed.
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
    """Four decades of cost land in one table: a dot product is nanoseconds and a
    full fit is seconds. Printing both in one unit would either round the fast
    row to zero or push the slow row off the column.
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
    """The notes block is deduplicated and appended only when something was
    recorded; an empty one would read as a caveat the run did not make."""
    text = format_report(_report(Comparison("op", _timing(1e-4), None, "checked")))
    assert "note" not in text
    assert text.rstrip().endswith("checked")


def test_a_native_run_names_the_compiler_and_whether_its_build_was_reproducible() -> None:
    """The speedup is only meaningful beside the flags that produced it: a
    fast-math build is faster and is not the same computation, so the header
    carries the build rather than leaving a reader to assume it.
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
    """Two rankings of different lengths are not a near miss; comparing them
    element-by-element would silently check only the shorter one."""
    with pytest.raises(BitIdentityError, match="Refusing to report a speedup"):
        check_same_order((0, 1, 2), (0, 1), "ranking")


# ---------------------------------------------------------------------------
# The speedup, and the workload it was measured on
# ---------------------------------------------------------------------------
def test_a_speedup_is_the_reference_cost_divided_by_the_native_one() -> None:
    """The headline number. Multiplying instead of dividing gives a figure that
    still moves in the right direction whenever both timings do, so nothing
    downstream would look wrong."""
    faster = Comparison("op", _timing(2.0), _timing(0.5, "native"), "checked")
    assert faster.speedup == 4.0

    slower = Comparison("op", _timing(0.5), _timing(2.0, "native"), "checked")
    assert slower.speedup == 0.25, "a slowdown is reported, not clamped"


def test_a_native_timing_of_zero_yields_no_speedup_rather_than_infinity() -> None:
    """A measured zero means the timer could not resolve the call, not that it
    was infinitely fast. Guarding on `<= 0` rather than `< 0` is what keeps the
    division from happening at all."""
    assert Comparison("op", _timing(1.0), _timing(0.0, "native"), "checked").speedup is None
    assert Comparison("op", _timing(1.0), None, "checked").speedup is None


def test_a_tiny_workload_still_plants_a_duplicate_and_a_twin_pair() -> None:
    """Scaled rather than fixed so a smoke-test workload still holds the exact
    ties the ranking rows exercise. Below twenty documents the scaling rounds to
    zero, and the floor is what stops the benchmark timing a corpus with no ties
    in it -- which is the case the tie-break comparison exists to measure."""
    spec = Workload(n_docs=10, vocab_size=40, n_queries=2, query_length=3, k=2).spec()
    assert spec.n_exact_duplicates == 1
    assert spec.n_twin_pairs == 1
    assert spec.n_users == 0, "interactions belong to the profile experiments"

    bigger = Workload(n_docs=100, vocab_size=200, n_queries=2, query_length=3, k=2).spec()
    assert bigger.n_exact_duplicates == 5, "and it scales above the floor"


def test_the_identity_check_counts_what_it_compared_in_words() -> None:
    """The string is what appears in the `checked` column beside every ratio, so
    it is the reader's evidence that the speedup was licensed at all."""
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
    """The thresholds are strict, so 1e-6 seconds is one microsecond and not a
    thousand nanoseconds. Off by one comparison and every row at a decade
    boundary jumps a unit, which is where a reader is most likely to be
    comparing two rows against each other.

    The whole rendered value is matched, not the unit alone: the table's own
    header contains the word "speedup", so asserting that a bare " s" appears
    somewhere in the report is satisfied whatever the row says. Mutation testing
    found that -- in this test rather than in the code.
    """
    text = format_report(_report(Comparison("op", _timing(seconds), None, "checked")))
    assert rendered in text
    assert if_it_slipped not in text


def test_a_native_run_that_reported_no_build_says_so_instead_of_crashing() -> None:
    """Both conditions are required before the build line is rendered. A backend
    that loaded but could not describe itself is a real state -- an older
    extension without the build-info symbol -- and the header has to survive it."""
    text = format_report(_report(Comparison("op", _timing(1e-4), None, "checked"), native=True))
    assert "unavailable" in text


def test_a_measured_cost_is_a_duration_and_not_a_clock_reading() -> None:
    """`perf_counter` has an arbitrary origin, typically machine uptime, so a
    batch timed as `now + start` instead of `now - start` reports a number in the
    thousands of seconds. It would look like a plausible column of timings and
    every ratio computed from it would be meaningless.

    The bound asserted is not a performance threshold: a no-op callable taking
    longer than a second is broken by any reading, and this is the assertion the
    module's "nothing asserts a duration" rule is carved out for.
    """
    timing = measure(lambda: None, label="noop", backend="reference", repeats=1)

    assert 0.0 <= timing.seconds < 1.0, f"a no-op measured {timing.seconds}s"
    assert math.isfinite(timing.seconds)
    assert timing.inner >= 1
    assert timing.calls == timing.repeats * timing.inner


def test_the_reported_cost_is_per_call_not_per_batch() -> None:
    """Calibration grows the batch until it clears the minimum batch time, so a
    batch holds many calls. Reporting the batch cost -- or multiplying by the
    batch size instead of dividing -- inflates the per-call figure by the
    calibration factor, which differs per operation and so silently reweights
    every comparison in the table against every other.
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
    """`while inner < _MAX_INNER and ...`. The loop grows the batch until it
    takes long enough to time; the cap is the other exit, for a callable so
    cheap that no reachable batch size crosses the threshold.

    Both constants are lowered here rather than reached honestly: at the real
    `_MAX_INNER` of 2^22 the test would make four million calls to prove it
    stopped.
    """
    from tfidf_stability.benchmarks import tfidf_perf

    monkeypatch.setattr(tfidf_perf, "_MAX_INNER", 8)
    # No batch of any size reaches this, so the cap is the only way out.
    monkeypatch.setattr(tfidf_perf, "_MIN_BATCH_SECONDS", 1e9)

    timing = tfidf_perf.measure(lambda: None, label="noop", backend="reference", repeats=1)

    assert timing.inner == 8, "the last doubling that satisfies inner < cap lands on the cap"


def test_a_callable_slow_enough_to_time_is_not_batched_at_all(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The other exit from the same loop, so the cap above is shown to be the
    cap rather than the only value `inner` can take. A single call already
    exceeding the threshold needs no batching, and batching it would multiply
    the measurement's cost for nothing."""
    from tfidf_stability.benchmarks import tfidf_perf

    monkeypatch.setattr(tfidf_perf, "_MIN_BATCH_SECONDS", 0.0)

    timing = tfidf_perf.measure(lambda: None, label="noop", backend="reference", repeats=1)

    assert timing.inner == 1


def test_the_reported_cost_is_divided_by_the_batch_size(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`seconds=best / inner`. A batched measurement reports the per-call cost,
    so two runs that happened to choose different batch sizes stay comparable --
    which is the whole point of recording `inner` beside the number."""
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
    """`reference_only_reason or "reference only"`. Fitting has no native side
    by design: idf is evaluated once in exact decimal arithmetic so the core
    never sees a logarithm (G13), and there is nothing to compare against on any
    machine.

    Collapsing that into the generic phrase would read as "the backend was not
    built here", inviting a reader to expect a speedup once it is -- and there
    is none to come.
    """
    report = run_benchmarks(TINY, repeats=1, use_native=False)
    fit = next(c for c in report.comparisons if c.name.startswith("fit"))

    assert fit.verified == "no native counterpart exists"
    assert fit.native is None
    assert "G13" in fit.note, "and the note says why"


def test_an_operation_whose_backend_was_simply_absent_uses_the_generic_phrase() -> None:
    """The contrast. Building the index does have a native counterpart; it was
    not exercised because this run asked for the reference alone, so the reason
    is about this run rather than about the operation."""
    report = run_benchmarks(TINY, repeats=1, use_native=False)
    index = next(c for c in report.comparisons if c.name.startswith("build index"))

    assert index.verified == "reference only"
    assert index.native is None


def test_no_reference_only_comparison_leaves_the_reason_blank() -> None:
    """Every row of the published table has to say what licensed it. An empty
    cell would be indistinguishable from a check that was skipped."""
    report = run_benchmarks(TINY, repeats=1, use_native=False)

    assert all(c.verified for c in report.comparisons)
    assert all(c.as_dict()["verified"] for c in report.comparisons)


# ---------------------------------------------------------------------------
# Fixture construction: what both backends are handed, built once and never timed
# ---------------------------------------------------------------------------
# Reached directly, as `test_datasets.py` reaches the generator's deterministic
# primitives. Nothing the fixture gets wrong shows up in a report -- the numbers
# are timings, and a fixture that timed the wrong thing would still produce a
# plausible table.
def test_queries_are_drawn_across_the_corpus_rather_than_all_from_one_document() -> None:
    """`stride = max(1, len(features) // max(1, n_queries))`. Asking for more
    queries than there are documents makes the floor division zero, and a stride
    of zero indexes `features[0]` for every one of them.

    The benchmark would then time the same query twenty times over, and its
    scoring row would report the cost of whichever document happened to be
    first rather than a spread across the corpus.
    """
    from tfidf_stability.benchmarks.tfidf_perf import _build_fixture

    crowded = Workload(n_docs=24, vocab_size=40, n_queries=30, query_length=4, k=3, seed=20260811)
    fixture = _build_fixture(crowded)

    assert len(fixture.queries) == 24, "capped at one query per document"
    distinct = {(q.indices, q.values) for q in fixture.queries}
    assert len(distinct) > 1, "a stride of zero would make every query the first document's"


def test_a_query_set_smaller_than_the_corpus_still_spreads_over_it() -> None:
    """The ordinary arm of the same expression, so the clamp above is shown to
    be the crowded case rather than the only behaviour."""
    from tfidf_stability.benchmarks.tfidf_perf import _build_fixture

    fixture = _build_fixture(TINY)

    assert len(fixture.queries) == TINY.n_queries
    assert len({(q.indices, q.values) for q in fixture.queries}) > 1


def test_the_ranking_rows_are_scored_from_the_first_query() -> None:
    """`scores=cosine_against_corpus(queries[0], ...)`. The ranking and top-k
    rows are timed on this one vector, and the docstring says which. Scoring a
    different query would leave the field's own comment false and the margin
    structure those rows exercise unrelated to the query named beside them.
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
    """`if native_call is None or verify is None`. Both halves are required:
    timing a native call nobody verified is exactly the speedup-on-a-divergent-
    result this module exists to refuse, and a verifier with nothing to run
    would report a check that never happened.

    With `and` in place of `or`, the half-populated rows fall through to the
    comparison path and the missing half is called anyway.
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
    """`_time_batch(fn, inner) < _MIN_BATCH_SECONDS`. The loop grows the batch
    until it is long enough to time, not until it is comfortably past: a batch
    landing exactly on the threshold has met it, and doubling again would spend
    twice the wall clock to learn nothing.

    The timer is replaced rather than raced. A real reading equal to the
    threshold to the last bit is not something a test can arrange, so the
    boundary would otherwise be unreachable and the comparison untested at the
    one input that distinguishes it.
    """
    from tfidf_stability.benchmarks import tfidf_perf

    monkeypatch.setattr(tfidf_perf, "_MAX_INNER", 8)
    monkeypatch.setattr(tfidf_perf, "_time_batch", lambda fn, inner: tfidf_perf._MIN_BATCH_SECONDS)

    timing = tfidf_perf.measure(lambda: None, label="exact", backend="reference", repeats=1)

    assert timing.inner == 1, "the first batch already met the threshold"
    assert timing.seconds == tfidf_perf._MIN_BATCH_SECONDS


def test_a_batch_just_under_the_threshold_is_doubled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The other side of the same comparison, one ulp away, so the boundary
    above is shown to be a boundary rather than the only behaviour."""
    from tfidf_stability.benchmarks import tfidf_perf

    just_under = math.nextafter(tfidf_perf._MIN_BATCH_SECONDS, 0.0)
    monkeypatch.setattr(tfidf_perf, "_MAX_INNER", 8)
    monkeypatch.setattr(tfidf_perf, "_time_batch", lambda fn, inner: just_under)

    timing = tfidf_perf.measure(lambda: None, label="under", backend="reference", repeats=1)

    assert timing.inner == 8, "never satisfied, so it grows to the cap"


def test_a_workload_asking_for_no_queries_is_not_validated_and_fails_late() -> None:
    """`Workload` accepts `n_queries=0` and nothing checks it. The stride's
    inner `max(1, n_queries)` keeps the floor division alive, so construction
    gets as far as `scores=cosine_against_corpus(queries[0], ...)` and fails
    there with a bare `IndexError`.

    Pinned as the current behaviour rather than fixed: it is a tests-only
    change, and the failure at least stops the run. Without the clamp it would
    be a `ZeroDivisionError` from the stride instead -- equally unhelpful, and
    one line earlier.
    """
    empty = Workload(n_docs=24, vocab_size=40, n_queries=0, query_length=4, k=3, seed=20260811)

    from tfidf_stability.benchmarks.tfidf_perf import _build_fixture

    with pytest.raises(IndexError, match="list index out of range"):
        _build_fixture(empty)
