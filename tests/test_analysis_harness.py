"""Tests for the Stage 11 experiment harness.

The tests check that the harness computes what it claims, not whether the
results are interesting. Most inputs have an answer known by hand, including
the cases where the correct answer is a failure.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from tfidf_stability.analysis.noise_floor import (
    NoiseFloor,
    PolicyError,
    TauBand,
    measure_noise_floor,
    tau_band,
    verify_band_invariance,
)
from tfidf_stability.analysis.stability_profile import (
    DEFAULT_RATIOS,
    CertificateAudit,
    TransitionPoint,
    certificate_audit,
    transition_curve,
)
from tfidf_stability.analysis.summarise import (
    DEFAULT_PERCENTILES,
    ExperimentResult,
    percentile,
    summarise_values,
)
from tfidf_stability.perturbation.score_bounds import certified_radius
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.io import canonical_json
from tfidf_stability.utils.numerics import same_bits
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser


def _floor(eta: float) -> NoiseFloor:
    """A NoiseFloor with a chosen eta, for testing the band arithmetic alone."""
    return NoiseFloor(
        per_policy=(PolicyError("naive", 100, 1, eta, 1.0),),
        n_queries=1,
        n_documents=10,
    )


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------
def test_percentiles_never_interpolate() -> None:
    """Every reported percentile must be an observation that occurred.

    NumPy's default returns 2.5 for this median. No observed margin equals it, so
    it cannot be looked up in the raw data or compared with `same_bits`
    (`analysis/summarise.py`'s module docstring).
    """
    sample = [1.0, 2.0, 3.0, 4.0]
    assert percentile(sample, 50) == 2.0
    assert all(percentile(sample, p) in sample for p in (0, 1, 25, 50, 75, 99, 100))


def test_percentile_endpoints_are_the_extremes() -> None:
    assert percentile([1.0, 9.0], 0) == 1.0
    assert percentile([1.0, 9.0], 100) == 9.0


def test_percentile_of_an_empty_sample_is_nan() -> None:
    assert math.isnan(percentile([], 50))


def test_nan_is_counted_not_averaged() -> None:
    """NaN marks an undefined quantity (G16); zero is a measurement."""
    d = summarise_values("m", [1.0, 2.0, math.nan, 3.0])
    assert d.n == 3
    assert d.n_nan == 1
    assert d.mean == 2.0


def test_the_exact_tie_share_is_reported_separately() -> None:
    """G3's headline statistic; it vanishes into the percentiles otherwise."""
    d = summarise_values("m", [0.0, 0.0, 0.0, 1.0])
    assert d.share_zero == 0.75
    assert d.percentiles["p50"] == 0.0


def test_an_all_nan_sample_summarises_without_raising() -> None:
    d = summarise_values("m", [math.nan, math.nan])
    assert d.n == 0
    assert d.n_nan == 2
    assert d.n_infinite == 0
    assert math.isnan(d.mean)


def test_an_infinity_is_counted_not_averaged_and_is_not_a_nan() -> None:
    """An infinite margin is defined, so the upstream filter passes it on.

    `boundary_margin((inf, 1.0), 1)` returns one. Averaged into the statistics
    it makes `mean` and the upper percentiles infinite, which `canonical_json`
    writes as `null`. `n_infinite` and `n_nan` keep the two kinds apart.
    """
    d = summarise_values("m", [1.0, 2.0, math.inf, 3.0])
    assert d.n == 3
    assert d.n_nan == 0
    assert d.n_infinite == 1
    assert d.mean == 2.0
    assert d.maximum == 3.0, "the largest measurement, not the infinity"

    both = summarise_values("m", [1.0, math.nan, math.inf, -math.inf])
    assert (both.n, both.n_nan, both.n_infinite) == (1, 1, 2)
    assert both.minimum == 1.0
    assert both.maximum == 1.0
    assert both.as_dict()["n_infinite"] == 2


# ---------------------------------------------------------------------------
# The tau band
# ---------------------------------------------------------------------------
def test_tau_floor_is_exactly_twice_eta() -> None:
    """The factor 2 comes from the margin error e_i + e_j."""
    assert _floor(1e-16).tau_floor == 2e-16


def test_a_band_with_no_gap_inside_it_is_invariant() -> None:
    scores = [[1.0, 1.0, 0.5, 0.5, 0.25]]  # gaps: 0, 0.5, 0, 0.25
    band = tau_band(_floor(1e-16), scores)
    assert band.is_valid
    assert band.g_min == 0.25
    assert band.n_exact_ties == 2
    assert band.n_gaps_in_band == 0
    assert band.is_invariant
    assert verify_band_invariance(band, scores)


def test_the_display_tau_lies_strictly_inside_the_band() -> None:
    band = tau_band(_floor(1e-16), [[1.0, 0.5]])
    assert band.tau_floor < band.display_tau() < band.g_min


def test_an_empty_band_is_reported_rather_than_papered_over() -> None:
    """If arithmetic noise reaches the decision boundary there is no valid tau.

    That is a finding about the corpus (A1's and A2's regimes do not separate
    on it), so the band reports it instead of returning a plausible number.
    """
    # eta = 1.0, so tau_floor = 2.0, well above the 0.5 gap.
    band = tau_band(_floor(1.0), [[1.0, 0.5]])
    assert not band.is_valid
    assert not band.is_invariant
    assert math.isnan(band.decades)
    assert math.isnan(band.display_tau())
    assert not verify_band_invariance(band, [[1.0, 0.5]])


def test_a_corpus_of_only_exact_ties_gives_an_infinite_ceiling() -> None:
    band = tau_band(_floor(1e-16), [[1.0, 1.0, 1.0]])
    assert band.n_positive_gaps == 0
    assert band.g_min == math.inf
    assert band.is_valid


def test_the_noise_floor_varies_the_norms_not_only_the_dot_product(
    mini_features, mini_model
) -> None:
    """The noise floor varies the norms as well as the dot product.

    `TfidfModel.norms` is precomputed under the model's own reduction, so a
    policy passed to `cosine_against_corpus` alone holds the norms fixed. The
    norm sums the whole document vector, so fixing it understates the floor.
    """
    query = TfidfVectoriser.transform_query(list(mini_features[0])[:4], mini_model)
    floor = measure_noise_floor(mini_model, [query])
    assert floor.eta >= 0.0
    assert {p.policy for p in floor.per_policy} == {"naive", "neumaier", "pairwise"}
    # The exact policy is the reference, so it is never one of the instruments.
    assert all(p.policy != "exact" for p in floor.per_policy)


# ---------------------------------------------------------------------------
# A1: the transition curve and the certificate
# ---------------------------------------------------------------------------
@pytest.fixture
def a1_setup(mini_corpus, mini_features, mini_model):
    table = AttributeTable.from_records(mini_corpus)
    documents = [mini_model.document(i) for i in range(mini_model.n_documents)]
    vectors = [
        cosine_against_corpus(
            TfidfVectoriser.transform_query(list(f)[:4], mini_model),
            documents,
            mini_model.norms,
        )
        for f in mini_features
    ]
    return vectors, table


def test_no_flip_ever_occurs_inside_the_certified_radius(a1_setup) -> None:
    """Section 4.4's guarantee: a single flip inside the radius falsifies it."""
    vectors, table = a1_setup
    points, _, _ = transition_curve(vectors, table, 2, seed=1, trials=25)
    inside = [p for p in points if p.within_certificate]
    assert inside, "the ratio grid must sample below the certified radius"
    assert all(p.n_flips == 0 for p in inside)


def test_the_flip_rate_is_monotone_enough_to_be_a_transition(a1_setup) -> None:
    """Sampled, so not strictly monotone; it must still end above where it starts."""
    vectors, table = a1_setup
    points, _, _ = transition_curve(vectors, table, 2, seed=1, trials=25)
    assert points[-1].flip_rate >= points[0].flip_rate


def test_exact_tie_queries_are_excluded_from_the_a1_curve(a1_setup) -> None:
    """At m_k = 0 the tie-break decides the outcome, which is A2's regime.

    Averaging those queries into an A1 curve reads a tie-break effect as one
    of numerical stability.
    """
    vectors, table = a1_setup
    zero_margin = [0.5] * len(vectors[0])  # every score identical
    _, used, excluded = transition_curve([zero_margin], table, 2, seed=1, trials=5)
    assert used == 0
    assert excluded == 1


def test_the_transition_curve_is_reproducible_from_its_seed(a1_setup) -> None:
    vectors, table = a1_setup
    first, _, _ = transition_curve(vectors, table, 2, seed=42, trials=20)
    second, _, _ = transition_curve(vectors, table, 2, seed=42, trials=20)
    assert [p.n_flips for p in first] == [p.n_flips for p in second]


def test_the_transition_curve_does_not_touch_the_global_generator(a1_setup) -> None:
    """A local Random, so unrelated code cannot perturb an experiment."""
    import random

    vectors, table = a1_setup
    random.seed(99)
    before = random.random()
    random.seed(99)
    transition_curve(vectors, table, 2, seed=1, trials=5)
    assert random.random() == before


def test_the_certificate_is_sound(a1_setup) -> None:
    """Certified-but-changed must be zero, or section 4.4 is falsified."""
    vectors, table = a1_setup
    audit = certificate_audit(vectors, table, 2, seed=3, trials=25)
    assert audit.certified_changed == 0
    assert audit.is_sound


def test_the_audit_populates_both_rows(a1_setup) -> None:
    """Drawing only tiny perturbations would make soundness trivially true."""
    vectors, table = a1_setup
    audit = certificate_audit(vectors, table, 2, seed=3, trials=40, max_ratio=8.0)
    assert audit.certified_unchanged > 0
    assert audit.uncertified_unchanged + audit.uncertified_changed > 0


def test_conservatism_is_reported_rather_than_accuracy(a1_setup) -> None:
    """Conservatism is a share in [0, 1].

    `False` from the certificate means "not covered"; it claims nothing about
    whether the ranking changes.
    """
    vectors, table = a1_setup
    audit = certificate_audit(vectors, table, 2, seed=3, trials=40)
    assert 0.0 <= audit.conservatism <= 1.0


# ---------------------------------------------------------------------------
# The result envelope
# ---------------------------------------------------------------------------
def test_the_result_digest_ignores_volatile_fields() -> None:
    """Two runs of the same experiment must agree, so a reader can check one hash."""
    a = ExperimentResult("e", {"x": 1, "timestamp": "2026-01-01T00:00:00Z"})
    b = ExperimentResult("e", {"x": 1, "timestamp": "2027-09-09T09:09:09Z"})
    assert a.digest() == b.digest()


def test_the_result_digest_tracks_the_payload() -> None:
    assert ExperimentResult("e", {"x": 1}).digest() != ExperimentResult("e", {"x": 2}).digest()


def test_the_result_digest_tracks_the_parameters() -> None:
    """A result computed at a different k is a different result."""
    a = ExperimentResult("e", {"x": 1}, parameters={"k": 10})
    b = ExperimentResult("e", {"x": 1}, parameters={"k": 20})
    assert a.digest() != b.digest()


def test_the_record_carries_provenance_and_environment() -> None:
    result = ExperimentResult("e", {"x": 1}, data_provenance={"kind": "synthetic"})
    record = result.as_dict()
    assert record["data_provenance"]["kind"] == "synthetic"
    assert record["environment"]
    assert record["result_digest"] == result.digest()


# ---------------------------------------------------------------------------
# Serialisation: results must be readable by something other than Python
# ---------------------------------------------------------------------------
def _strict_loads(text: str):
    """Parse rejecting NaN/Infinity, the way every non-Python parser does."""

    def reject(token: str):
        raise ValueError(f"non-standard JSON token {token!r}")

    return json.loads(text, parse_constant=reject)


def test_a_non_finite_value_does_not_produce_invalid_json() -> None:
    """Non-finite floats are written as ``null``, not as bare tokens.

    NaN marks an undefined margin (G16); ``g_min`` is infinite with no positive
    gap. Python emits ``NaN``/``Infinity``, which its own loader accepts and
    every strict parser (JavaScript, jq, Go, Rust) rejects.
    """
    text = canonical_json({"undefined": math.nan, "unbounded": math.inf, "fine": 1.5})
    parsed = _strict_loads(text)
    assert parsed == {"undefined": None, "unbounded": None, "fine": 1.5}


def test_non_finite_values_are_sanitised_at_every_depth() -> None:
    """Nested, because a margin summary is several levels down in a payload."""
    text = canonical_json({"a": [{"b": [math.nan, {"c": -math.inf}]}]})
    assert _strict_loads(text) == {"a": [{"b": [None, {"c": None}]}]}


def test_a_full_experiment_result_serialises_to_strict_json() -> None:
    """End to end: a written result must be machine-readable."""
    result = ExperimentResult(
        experiment="e",
        payload={"margins": summarise_values("m", [math.nan, math.nan]).as_dict()},
        parameters={"k": 1},
    )
    _strict_loads(canonical_json(result.as_dict()))


# ---------------------------------------------------------------------------
# Degenerate tau bands
# ---------------------------------------------------------------------------
def test_a_zero_noise_floor_does_not_crash_the_invariance_check() -> None:
    """A zero `tau_floor` must not break the invariance check.

    `tau_floor` is 0 whenever every reduction policy is correctly rounded, as
    Neumaier measures. `math.log10(0)` raises, so the probe substitutes a value.
    """
    scores = [[1.0, 0.5, 0.25]]
    band = tau_band(_floor(0.0), scores)
    assert band.tau_floor == 0.0
    assert verify_band_invariance(band, scores)
    assert math.isfinite(band.display_tau())


def test_an_unbounded_band_never_yields_an_infinite_tau() -> None:
    """An infinite `g_min` must not yield an infinite display tau.

    With every score identical there is no positive gap, so `g_min` is infinite.
    The geometric midpoint would be `inf`, where every tie ball swallows the
    corpus.
    """
    scores = [[0.5, 0.5, 0.5]]
    band = tau_band(_floor(1e-16), scores)
    assert math.isinf(band.g_min)
    assert math.isfinite(band.display_tau())
    assert band.display_tau() >= band.tau_floor


# ---------------------------------------------------------------------------
# The reporting blocks, which are what a manifest carries
# ---------------------------------------------------------------------------


# Every as_dict below lands in a published JSON file. The tests assert what the
# block guarantees rather than a literal key list: it survives the canonical
# writer, and every float is paired with the hex form that makes a comparison
# across machines exact. A literal key list would pin the schema.
def test_a_policy_error_block_survives_the_canonical_writer() -> None:
    block = PolicyError("naive", 100, 25, 1e-16, 8.0).as_dict()
    assert json.loads(canonical_json(block, indent=None)) == block
    assert block["policy"] == "naive"


def test_the_share_of_differing_scores_is_the_ratio_it_claims() -> None:
    assert PolicyError("naive", 100, 25, 1e-16, 8.0).share_differing == 0.25


def test_a_policy_that_compared_nothing_reports_zero_rather_than_dividing() -> None:
    """An empty comparison reports a share of zero rather than raising.

    An empty comparison is not evidence of agreement, but it must not raise
    while a report is being assembled.
    """
    assert PolicyError("exact", 0, 0, 0.0, 0.0).share_differing == 0.0


def test_a_noise_floor_block_carries_the_hex_form_of_every_float() -> None:
    """Decimal rendering loses the last bit, and the last bit is the subject.

    Every `*_hex` field is paired with its decimal sibling by name, so a field
    added later is covered and a field reporting the wrong quantity is caught.
    The block carries `eta_hex`, `tau_floor_hex` and a `max_abs_hex` per policy.
    """
    block = _floor(1e-16).as_dict()
    assert json.loads(canonical_json(block, indent=None)) == block

    def check(mapping: dict[str, object], where: str) -> int:
        checked = 0
        for key, value in mapping.items():
            if key.endswith("_hex"):
                decimal = key.removesuffix("_hex")
                assert decimal in mapping, f"{where}{key} has no {decimal} beside it"
                assert float.fromhex(str(value)) == mapping[decimal], (
                    f"{where}{key} is {value}, which is not the hex form of "
                    f"{decimal}={mapping[decimal]}"
                )
                checked += 1
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        checked += check(item, f"{where}{key}[{i}].")
        return checked

    assert check(block, "") >= 3, "eta_hex, tau_floor_hex and at least one max_abs_hex"


def test_a_tau_band_block_round_trips_and_pins_its_endpoints_in_hex() -> None:
    band = tau_band(_floor(1e-16), [[1.0, 0.5, 0.25]])
    block = band.as_dict()
    assert json.loads(canonical_json(block, indent=None)) == block
    assert float.fromhex(block["tau_floor_hex"]) == block["tau_floor"]
    assert float.fromhex(block["g_min_hex"]) == block["g_min"]


def test_the_band_width_in_decades_is_the_logarithm_of_its_endpoints() -> None:
    band = tau_band(_floor(1e-16), [[1.0, 0.5, 0.25]])
    assert band.decades == pytest.approx(math.log10(band.g_min / band.tau_floor))


def test_a_band_with_a_zero_floor_reports_no_width_rather_than_minus_infinity() -> None:
    """log10(0) is not a number of decades, so the width is NaN, not -inf."""
    band = tau_band(_floor(0.0), [[1.0, 0.5]])
    assert band.tau_floor == 0.0
    assert math.isnan(band.decades)


# ---------------------------------------------------------------------------
# The band probe
# ---------------------------------------------------------------------------
def test_a_single_probe_samples_the_lower_endpoint_rather_than_failing() -> None:
    """One probe is a single tau, which the clamp sets just below `g_min`."""
    band = tau_band(_floor(1e-16), [[1.0, 0.5, 0.25]])
    assert verify_band_invariance(band, [[1.0, 0.5, 0.25]], probes=1)


def test_an_unbounded_band_is_probed_over_a_substitute_upper_end() -> None:
    """An infinite `g_min` is probed over a finite substitute upper end.

    `g_min` is infinite when no strictly-positive gap exists. Probing to
    infinity would sample nothing.
    """
    band = tau_band(_floor(1e-16), [[1.0, 1.0, 1.0]])
    assert math.isinf(band.g_min), "the premise: every gap is exactly zero"
    assert verify_band_invariance(band, [[1.0, 1.0, 1.0]]) is not None


# ---------------------------------------------------------------------------
# The certificate audit's own honesty
# ---------------------------------------------------------------------------
def test_an_audit_that_drew_no_certified_perturbation_is_not_conclusive() -> None:
    """An audit with an empty certified cell is sound but not conclusive.

    `is_sound` is `certified_changed == 0`, which an audit that certified
    nothing satisfies having checked the theorem zero times.
    """
    from tfidf_stability.analysis.stability_profile import CertificateAudit

    empty = CertificateAudit(
        certified_unchanged=0,
        certified_changed=0,
        uncertified_unchanged=5,
        uncertified_changed=3,
        n_undefined=0,
        n_exact_tie=0,
    )
    assert empty.is_sound, "vacuously, which is the point"
    assert not empty.is_conclusive, "and the count beside it says so"
    assert empty.n_certified == 0


def test_an_audit_with_certified_trials_is_conclusive_and_counts_them() -> None:
    from tfidf_stability.analysis.stability_profile import CertificateAudit

    audit = CertificateAudit(
        certified_unchanged=7,
        certified_changed=0,
        uncertified_unchanged=2,
        uncertified_changed=6,
        n_undefined=0,
        n_exact_tie=0,
    )
    assert audit.n_certified == 7
    assert audit.is_conclusive
    assert audit.conservatism == pytest.approx(2 / 8)
    assert json.loads(canonical_json(audit.as_dict(), indent=None)) == audit.as_dict()


def test_conservatism_over_no_uncertified_trials_is_undefined_not_zero() -> None:
    """Zero would claim every uncertified perturbation changed the ranking."""
    from tfidf_stability.analysis.stability_profile import CertificateAudit

    audit = CertificateAudit(
        certified_unchanged=3,
        certified_changed=0,
        uncertified_unchanged=0,
        uncertified_changed=0,
        n_undefined=0,
        n_exact_tie=0,
    )
    assert math.isnan(audit.conservatism)


def test_a_transition_point_block_round_trips_through_the_canonical_writer() -> None:
    from tfidf_stability.analysis.stability_profile import TransitionPoint

    point = TransitionPoint(ratio=1.1, n_flips=2, n_trials=1320)
    block = point.as_dict()
    assert json.loads(canonical_json(block, indent=None)) == block
    assert point.flip_rate == pytest.approx(2 / 1320)


def test_a_ratio_below_one_is_inside_the_certificate_and_at_one_is_not() -> None:
    """The certificate covers radii strictly below the certified radius.

    At exactly the certified radius the two scores meet, so `ratio == 1.0` is
    not covered.
    """
    from tfidf_stability.analysis.stability_profile import TransitionPoint

    assert TransitionPoint(ratio=0.99, n_flips=0, n_trials=10).within_certificate
    assert not TransitionPoint(ratio=1.0, n_flips=0, n_trials=10).within_certificate


# ---------------------------------------------------------------------------
# Candidate sets vary per query, so k can exceed one of them
# ---------------------------------------------------------------------------


# Section 7.1 gives every query its own candidate set, so a fixed k is larger
# than some of them. Both entry points count those rather than clamping: a
# clamped k measures the margin at a different rank and quietly averages two
# different quantities together.
def test_a_query_shorter_than_k_is_excluded_from_the_transition_rather_than_clamped() -> None:
    table = AttributeTable.from_records(
        [{"doc_id": f"d{i}", "popularity": 10 - i} for i in range(6)]
    )
    tall = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    short = [0.9, 0.8]  # fewer candidates than k
    points, n_used, n_excluded = transition_curve(
        [tall, short], table, k=4, seed=1, ratios=(0.5, 2.0), trials=4
    )

    assert n_excluded >= 1, "the short query must be counted out, not measured at a different rank"
    assert n_used >= 1, "the usable query must still be measured"
    assert len(points) == 2, "one point per requested ratio, regardless of exclusions"
    assert all(p.n_trials == n_used * 4 for p in points), (
        "an excluded query must contribute no trials, or the denominator is wrong"
    )


def test_a_query_shorter_than_k_is_undefined_in_the_audit_rather_than_clamped() -> None:
    table = AttributeTable.from_records(
        [{"doc_id": f"d{i}", "popularity": 10 - i} for i in range(6)]
    )
    tall = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    short = [0.9, 0.8]
    audit = certificate_audit([tall, short], table, k=4, seed=1, trials=4)

    assert audit.n_undefined >= 1, "a query with no rank k has no certificate there"


def test_a_query_whose_certificate_is_undefined_is_counted_not_audited() -> None:
    """k == N leaves no boundary gap, so no radius exists to certify against."""
    table = AttributeTable.from_records(
        [{"doc_id": f"d{i}", "popularity": 10 - i} for i in range(3)]
    )
    exactly_k = [0.9, 0.8, 0.7]  # k equals the candidate count
    audit = certificate_audit([exactly_k], table, k=3, seed=1, trials=4)

    assert audit.n_undefined == 1
    assert audit.n_certified == 0
    assert not audit.is_conclusive, "nothing was certified, so nothing was tested"


def test_a_certificate_with_a_non_numeric_radius_is_counted_undefined() -> None:
    """A certificate that is defined with a NaN radius counts as undefined.

    A NaN among the scores gives a certificate that reports itself defined while
    its radius is NaN. Every comparison against NaN is false, so `realised <
    radius` would answer "not certified" for every trial.
    """
    table = AttributeTable.from_records(
        [{"doc_id": f"d{i}", "popularity": 10 - i} for i in range(4)]
    )
    poisoned = [0.9, math.nan, 0.7, 0.6]
    cert = certified_radius(sorted(poisoned, reverse=True), 2)
    assert cert.defined, "the premise: it claims to be defined"
    assert math.isnan(cert.set_radius), "but carries no usable radius"

    audit = certificate_audit([poisoned], table, k=2, seed=1, trials=4)
    assert audit.n_undefined == 1, "a NaN radius is no radius, and must be counted as such"
    assert audit.n_certified == 0
    assert not audit.is_conclusive


# ---------------------------------------------------------------------------
# The noise floor needs a corpus long enough to have a floor
# ---------------------------------------------------------------------------
def test_a_long_document_corpus_makes_the_naive_fold_stray_from_exact() -> None:
    """A corpus of long documents makes the naive fold stray from exact.

    On the mini corpus every policy agrees with EXACT bit for bit, so the
    error-accumulation branch is not reached. Documents here are long enough
    that the norm summation, where the error accumulates, diverges.
    """
    rng = random.Random(11)
    vocab = [f"t{i}" for i in range(150)]
    documents = [[rng.choice(vocab) for _ in range(80)] for _ in range(50)]
    model = TfidfVectoriser().fit(documents, [f"d{i}" for i in range(50)])
    queries = [
        TfidfVectoriser.transform_query([rng.choice(vocab) for _ in range(40)], model)
        for _ in range(5)
    ]

    floor = measure_noise_floor(model, queries)
    by_policy = {p.policy: p for p in floor.per_policy}

    assert by_policy["naive"].n_differing > 0, (
        "no score strayed from exact, so the floor measured nothing"
    )
    assert floor.eta > 0.0, "a floor of exactly zero is the absence of a measurement"
    assert by_policy["naive"].max_ulps > 0.0
    assert by_policy["naive"].max_abs == floor.eta, (
        "eta is the largest deviation any instrument recorded"
    )
    assert by_policy["neumaier"].n_differing == 0, (
        "compensated summation is expected to agree; if it strayed the premise "
        "of using it as the recommended policy would be gone"
    )


def test_a_tau_beyond_the_band_moves_the_tie_structure_it_was_meant_to_preserve() -> None:
    """The invariance check must be able to answer no.

    Inside the band the tie groups are the exact-equality classes at every tau.
    A tau above g_min merges two separated scores, so the shape moves
    and the check reports it rather than passing regardless.
    """
    scores = [[1.0, 0.9, 0.5]]
    band = tau_band(_floor(1e-16), scores)
    assert verify_band_invariance(band, scores), "inside the band nothing moves"

    from tfidf_stability.analysis.noise_floor import TauBand

    too_wide = TauBand(
        tau_floor=band.tau_floor,
        g_min=0.5,  # above the 0.1 gap, so that pair merges partway through
        n_exact_ties=band.n_exact_ties,
        n_positive_gaps=band.n_positive_gaps,
        n_gaps_in_band=1,
    )
    assert not verify_band_invariance(too_wide, scores), (
        "a band spanning a real gap cannot leave the tie structure invariant"
    )


# ---------------------------------------------------------------------------
# percentile: a nearest-rank sample element, on a 0-to-100 scale
# ---------------------------------------------------------------------------


# There are two percentile functions in this package and they take different
# scales. This one is `p in [0, 100]`; `ranking.margins.summarise` takes
# quantiles in `[0, 1]`. Passing 0.5 here asks for the half-a-percent point and
# returns the sample minimum, which is a plausible number and the wrong one.
@pytest.mark.parametrize(("p", "expected"), [(0, 1.0), (25, 1.0), (50, 2.0), (75, 3.0), (100, 4.0)])
def test_the_percentile_scale_runs_to_a_hundred_and_not_to_one(p: float, expected: float) -> None:
    """Every returned value is a sample element at the nearest rank.

    A margin distribution has an atom at exactly zero, and interpolating across
    it invents a value no query produced.
    """
    assert percentile([1.0, 2.0, 3.0, 4.0], p) == expected


def test_asking_on_the_wrong_scale_silently_returns_the_minimum() -> None:
    """`0.5` meant as a median is half a percent, which rounds to rank one.

    Pinned because both spellings exist in this package and neither raises: a
    caller who reaches for the wrong one gets the sample minimum reported as a
    median, and every downstream figure is quietly the wrong statistic.
    """
    sample = [1.0, 2.0, 3.0, 4.0]
    assert percentile(sample, 0.5) == 1.0, "half a percent"
    assert percentile(sample, 50) == 2.0, "the median it was probably meant to be"


@pytest.mark.parametrize(("p", "expected"), [(-10, 1.0), (200, 4.0), (-math.inf, 1.0)])
def test_a_percentile_outside_the_scale_clamps_to_an_end(p: float, expected: float) -> None:
    """An out-of-range percentile reports an extreme instead of raising.

    The `p <= 0` and `p >= 100` guards return an end rather than checking the
    range, so `p = -10` gives the minimum and `p = 200` the maximum.
    """
    assert percentile([1.0, 2.0, 3.0, 4.0], p) == expected


def test_a_percentile_of_an_empty_sample_is_undefined_rather_than_zero() -> None:
    """NaN marks an undefined quantity; an empty sample is no evidence of zero."""
    assert math.isnan(percentile([], 50))


def test_a_single_observation_is_every_percentile_of_itself() -> None:
    assert percentile([3.0], 0) == percentile([3.0], 50) == percentile([3.0], 100) == 3.0


def test_the_sample_is_assumed_sorted_and_is_not_checked() -> None:
    """The sample is assumed sorted and is not checked.

    Callers already hold a sorted array, and re-sorting per percentile would be
    the dominant cost. An unsorted sample returns a sample element that is not
    the percentile, and the failure is silent.
    """
    unsorted = [4.0, 1.0, 3.0, 2.0]
    assert percentile(unsorted, 50) == 1.0, "the element at the median rank, not the median"
    assert percentile(sorted(unsorted), 50) == 2.0


# ---------------------------------------------------------------------------
# TauBand: the two-sided constraint, and the two degenerate bands
# ---------------------------------------------------------------------------


# The band is `[tau_floor, g_min)`: above arithmetic noise and below the
# smallest gap the corpus exhibits. It is a plain frozen dataclass, so every
# branch is reachable by construction rather than by fitting a corpus that
# happens to produce one.
def _band(tau_floor: float, g_min: float, *, gaps_in_band: int = 0) -> TauBand:
    """A band built directly, local by house convention."""
    return TauBand(
        tau_floor=tau_floor,
        g_min=g_min,
        n_gaps_in_band=gaps_in_band,
        n_exact_ties=0,
        n_positive_gaps=1,
    )


def test_a_band_is_valid_only_while_the_floor_stays_strictly_below_the_smallest_gap() -> None:
    """The floor must sit strictly below the smallest gap.

    At equality there is no tau that is both above the noise and below every
    gap, so the constraint has no solution, which G23 calls a finding.
    """
    assert _band(1e-16, 1e-9).is_valid
    assert not _band(1e-9, 1e-9).is_valid, "the boundary itself is empty"
    assert not _band(1e-8, 1e-9).is_valid, "and beyond it inverted"


def test_an_empty_band_reports_every_derived_quantity_as_undefined() -> None:
    """An empty band is not invariant, and its width and quoted tau are NaN.

    No tau exists, so there is nothing to quote and nothing to be invariant
    over. Zero would read as a legitimate exact-tie baseline.
    """
    empty = _band(1e-9, 1e-9)

    assert not empty.is_invariant
    assert math.isnan(empty.decades)
    assert math.isnan(empty.display_tau())


def test_a_gap_inside_the_band_costs_invariance_without_costing_validity() -> None:
    """Validity and invariance are separate claims.

    A band can admit a tau while the tie structure still changes across it,
    which is the case where quoting a single tau would be misleading.
    """
    breached = _band(1e-16, 1e-9, gaps_in_band=1)

    assert breached.is_valid
    assert not breached.is_invariant
    assert not math.isnan(breached.display_tau()), "a tau still exists to quote"


def test_the_quoted_tau_is_the_geometric_midpoint_of_an_ordinary_band() -> None:
    """The quoted tau is the geometric midpoint, because the band spans decades.

    The arithmetic mean of `1e-16` and `1e-9` sits just below the upper
    endpoint and would look fitted to it.
    """
    band = _band(1e-16, 1e-9)

    assert band.display_tau() == math.sqrt(1e-16 * 1e-9)
    assert band.decades == pytest.approx(7.0)


def test_an_all_tied_corpus_quotes_the_floor_itself() -> None:
    """An all-tied corpus quotes the floor itself.

    `g_min` is infinite when no strictly-positive gap exists, so the midpoint
    would be infinite too and every tie ball would swallow the corpus. No gap
    exists to cross, so every admissible tau gives the same structure.
    """
    all_tied = _band(1e-16, math.inf)

    assert all_tied.is_valid
    assert all_tied.is_invariant
    assert all_tied.display_tau() == 1e-16
    assert math.isinf(all_tied.decades)


def test_a_corpus_with_no_measured_error_quotes_half_the_upper_endpoint() -> None:
    """A zero floor quotes half the upper endpoint.

    `tau_floor` is zero when every reduction policy agrees with exact
    arithmetic. The geometric midpoint would collapse to zero, legal under G3
    but the exact-tie baseline rather than a tau above the noise.
    """
    no_error = _band(0.0, 1e-9)

    assert no_error.display_tau() == 5e-10
    assert math.isnan(no_error.decades), "a ratio against zero has no logarithm"


def test_both_degeneracies_at_once_collapse_the_quoted_tau_to_zero() -> None:
    """No measured error and no positive gap together quote a tau of zero.

    The infinite-`g_min` branch is checked first, so the answer is the floor,
    which is zero. This is the one configuration where the quoted tau is the
    exact-tie baseline rather than a measurement.
    """
    both = _band(0.0, math.inf)

    assert both.is_valid, "vacuously: zero is below infinity"
    assert both.display_tau() == 0.0


def test_the_band_reports_its_endpoints_in_hex_as_well_as_decimal() -> None:
    """Both endpoints are recorded in hex as well as decimal.

    The decimal rendering of a subnormal loses bits a reader needs to reproduce
    the band exactly.
    """
    payload = _band(1e-16, 1e-9).as_dict()

    assert payload["tau_floor_hex"] == float.hex(1e-16)
    assert payload["g_min_hex"] == float.hex(1e-9)
    assert set(payload) >= {"is_valid", "is_invariant", "decades", "display_tau"}


# ---------------------------------------------------------------------------
# TransitionPoint: one sampled ratio on the A1 curve
# ---------------------------------------------------------------------------
def test_a_ratio_below_one_is_covered_by_the_certificate_and_at_one_is_not() -> None:
    """The theorem is strict, so `ratio == 1.0` sits on the boundary uncovered.

    The guarantee is `eps < m_k / 2`. The boundary point is still sampled: a
    flip there rather than beyond it is the failure of interest.
    """
    assert TransitionPoint(ratio=0.99, n_flips=0, n_trials=10).within_certificate
    assert not TransitionPoint(ratio=1.0, n_flips=0, n_trials=10).within_certificate
    assert not TransitionPoint(ratio=1.01, n_flips=0, n_trials=10).within_certificate


def test_a_ratio_of_zero_is_covered_because_no_perturbation_is_applied() -> None:
    """The degenerate left end of the sweep."""
    assert TransitionPoint(ratio=0.0, n_flips=0, n_trials=10).within_certificate


def test_a_flip_rate_over_no_trials_is_undefined_rather_than_zero() -> None:
    """An unsampled ratio is no evidence.

    A zero would plot as a point on the transition curve that no trial produced.
    """
    assert math.isnan(TransitionPoint(ratio=0.5, n_flips=0, n_trials=0).flip_rate)


@pytest.mark.parametrize(
    ("flips", "trials", "rate"), [(0, 10, 0.0), (5, 10, 0.5), (10, 10, 1.0), (1, 3, 1 / 3)]
)
def test_the_flip_rate_is_the_share_of_trials_that_flipped(
    flips: int, trials: int, rate: float
) -> None:
    """The rate is reported with its denominator.

    A rate over three trials and one over thirty thousand are different claims.
    """
    point = TransitionPoint(ratio=0.5, n_flips=flips, n_trials=trials)
    assert point.flip_rate == pytest.approx(rate)
    assert point.as_dict()["n_trials"] == trials


# ---------------------------------------------------------------------------
# CertificateAudit: soundness is not the same as having tested anything
# ---------------------------------------------------------------------------
def _audit(cu: int = 0, cc: int = 0, uu: int = 0, uc: int = 0, **kw: int) -> CertificateAudit:
    """The 2x2 table built directly, local by house convention."""
    return CertificateAudit(
        certified_unchanged=cu,
        certified_changed=cc,
        uncertified_unchanged=uu,
        uncertified_changed=uc,
        n_undefined=kw.get("n_undefined", 0),
        n_exact_tie=kw.get("n_exact_tie", 0),
    )


def test_a_single_certified_failure_makes_the_audit_unsound() -> None:
    """`certified_changed` must be zero.

    Any other value falsifies section 4.4: it is a bug or a broken proof, never
    a statistic.
    """
    assert _audit(cu=100, cc=0).is_sound
    assert not _audit(cu=100, cc=1).is_sound


def test_an_audit_that_drew_no_certified_perturbation_is_sound_and_says_nothing() -> None:
    """Soundness and conclusiveness are separate properties.

    `is_sound` is `certified_changed == 0`, which an audit with an empty
    certified cell satisfies having checked the theorem zero times, so
    soundness is read beside the certified count.
    """
    vacuous = _audit(uu=1000, uc=500)

    assert vacuous.is_sound, "vacuously"
    assert not vacuous.is_conclusive
    assert vacuous.n_certified == 0


def test_one_certified_perturbation_is_enough_to_make_the_audit_conclusive() -> None:
    """The threshold is one certified trial, not a sample-size rule.

    The theorem is a guarantee, so a single counterexample would refute it and
    a single confirmation is a real test of it.
    """
    assert _audit(cu=1).is_conclusive
    assert _audit(cc=1).is_conclusive, "even a failing one tested it"


def test_conservatism_is_reported_over_the_uncertified_cases_alone() -> None:
    """Conservatism is measured over the uncertified cases alone.

    The certificate is sufficient, not necessary, so the figure of interest is
    how often it declines to certify a stable case. Accuracy over the whole
    table would reward a certificate that always says no.
    """
    assert _audit(cu=10, uu=3, uc=1).conservatism == 0.75
    assert _audit(cu=10, uu=0, uc=4).conservatism == 0.0


def test_conservatism_over_no_uncertified_cases_is_undefined() -> None:
    """With no uncertified case, conservatism is NaN rather than 1.0.

    Every perturbation is certified, so there is no declined case. A value of
    1.0 would claim perfect conservatism from no evidence.
    """
    assert math.isnan(_audit(cu=10).conservatism)


def test_the_excluded_and_undefined_counts_travel_with_the_table() -> None:
    """The excluded and undefined counts travel with the 2x2 table.

    Exact-tie queries are excluded from the A1 curve because the tie-break
    decides membership there. They are counted rather than dropped, so the
    exclusion is visible in the published record.
    """
    audit = _audit(cu=5, uu=2, n_undefined=3, n_exact_tie=7)
    payload = audit.as_dict()

    assert payload["n_exact_tie"] == 7
    assert payload["n_undefined"] == 3
    assert payload["n_certified"] == 5
    assert set(payload) >= {"is_sound", "is_conclusive", "conservatism"}


# ---------------------------------------------------------------------------
# verify_band_invariance: probing a band whose endpoints may be degenerate
# ---------------------------------------------------------------------------

#: A corpus with one clean gap structure, so any change in the probed tie shape
#: is attributable to the tau rather than to the scores.
_SPREAD = [[1.0, 0.5, 0.0]]


def test_an_invalid_band_verifies_nothing_and_says_so() -> None:
    """An invalid band has no interval to probe, so the check returns `False`.

    A vacuous `True` would report the invariance upheld having checked it
    nowhere.
    """
    assert verify_band_invariance(_band(1e-9, 1e-9), _SPREAD) is False


def test_an_ordinary_band_holds_its_tie_structure_across_every_probe() -> None:
    """Piecewise constancy proves this when no gap lies inside the band.

    The probe runs the code anyway, so a tau comparison made with the wrong
    strictness surfaces here rather than in an argument.
    """
    assert verify_band_invariance(_band(1e-16, 1e-9), _SPREAD) is True


@pytest.mark.parametrize("probes", [0, -1, -(2**20)])
def test_a_probe_count_below_one_is_refused_rather_than_dividing_by_zero(probes: int) -> None:
    """A probe count below one is refused, and the guard names the value.

    A count below one samples no tau, so there is no invariance to check.
    """
    with pytest.raises(ValueError, match=f"probes must be at least 1, got {probes}"):
        verify_band_invariance(_band(1e-16, 1e-9), _SPREAD, probes=probes)


def test_a_single_probe_takes_the_lower_endpoint_rather_than_spacing_nothing() -> None:
    """One probe cannot be spaced across a band, so it is a single tau.

    It starts at the lower endpoint, the smallest tau the band admits. With
    `g_min` finite, the clamp then moves it just below `g_min`.
    """
    assert verify_band_invariance(_band(1e-16, 1e-9), _SPREAD, probes=1) is True


def test_a_band_wide_enough_to_swallow_a_real_gap_fails_verification() -> None:
    """A band spanning a real score gap fails verification.

    With the upper endpoint above a score gap, the tie structure changes
    across the band and the probe detects it.
    """
    assert verify_band_invariance(_band(1e-16, 10.0), _SPREAD) is False


def test_a_degenerate_lower_endpoint_is_substituted_rather_than_logged() -> None:
    """A zero lower endpoint is substituted so the band is still probed.

    `tau_floor` is zero when every reduction policy is correctly rounded, and
    `log10(0)` raises. A tiny positive value replaces it, and `tau = 0` is
    appended because it is admissible and is the exact-tie baseline.
    """
    assert verify_band_invariance(_band(0.0, 1e-9), _SPREAD) is True


def test_a_degenerate_upper_endpoint_is_substituted_and_then_fails_honestly() -> None:
    """An infinite upper endpoint is substituted, then fails on a gapped corpus.

    `g_min` is infinite when no strictly-positive gap exists, so a finite
    stand-in is probed. On a corpus that has gaps the stand-in reaches above
    them, the structure moves and the answer is `False`.
    """
    assert verify_band_invariance(_band(1e-16, math.inf), _SPREAD) is False


# ---------------------------------------------------------------------------
# certificate_audit: which queries the audit refuses to count, and why
# ---------------------------------------------------------------------------
def test_a_k_past_the_end_of_a_query_is_counted_as_undefined_not_dropped(a1_setup) -> None:
    """`k >= len(scores)` has no boundary to certify, so it is counted.

    The published record then shows how much of the query set the audit cannot
    speak for, rather than a rate over a denominator that shrank.
    """
    vectors, table = a1_setup
    audit = certificate_audit(vectors, table, k=999, seed=1, trials=2)

    assert audit.n_undefined == len(vectors)
    assert audit.n_certified == 0
    assert not audit.is_conclusive, "nothing was tested, so soundness says nothing"


def test_an_exact_tie_query_is_excluded_and_counted_separately(a1_setup) -> None:
    """An exact-tie query is excluded and counted, since it is A2's regime.

    At `m_k = 0` the radius is zero, so `eps` is zero, the perturbed scores
    equal the originals and "unchanged" is trivially true. Such a trial would
    land in (uncertified, unchanged) and inflate the published conservatism.
    """
    vectors, table = a1_setup
    tied = [[0.5] * len(vectors[0])]
    audit = certificate_audit(tied, table, k=1, seed=1, trials=5)

    assert audit.n_exact_tie == 1
    assert audit.n_certified == 0
    assert audit.uncertified_unchanged == 0, "the trials never ran at all"


def test_the_counted_categories_partition_every_query(a1_setup) -> None:
    """Each query is undefined, exact-tie or trialled, and in exactly one.

    The three counts reconcile against the query set. A query silently in none
    of them would make every rate above it wrong by an unknown amount.
    """
    vectors, table = a1_setup
    trials = 4
    audit = certificate_audit(vectors, table, k=2, seed=7, trials=trials)

    trialled = len(vectors) - audit.n_undefined - audit.n_exact_tie
    counted = (
        audit.certified_unchanged
        + audit.certified_changed
        + audit.uncertified_unchanged
        + audit.uncertified_changed
    )

    # Exact, not `counted // trials`: integer division rounds a miscount away.
    # With four cells and one of them reading the wrong key, the total can be
    # wrong by less than `trials` and still floor to the right answer.
    assert counted == trialled * trials
    assert trialled > 0, "every query was excluded, so the table below is empty"


def test_the_audit_is_reproducible_from_its_seed(a1_setup) -> None:
    """Two audits at one seed must agree exactly.

    The generator is local and seeded per call, so a published 2x2 table can be
    rerun.
    """
    vectors, table = a1_setup
    first = certificate_audit(vectors, table, k=2, seed=11, trials=6)
    second = certificate_audit(vectors, table, k=2, seed=11, trials=6)

    assert first.as_dict() == second.as_dict()


def test_the_audit_does_not_disturb_the_global_random_state(a1_setup) -> None:
    """The audit uses `random.Random(seed)` rather than `random.seed`.

    Reseeding the module generator would make every other seeded thing in the
    process depend on whether an audit has run.
    """
    import random as _random

    vectors, table = a1_setup
    _random.seed(1234)
    before = _random.random()

    _random.seed(1234)
    certificate_audit(vectors, table, k=2, seed=99, trials=3)
    assert _random.random() == before


def test_perturbations_straddle_the_radius_so_both_cells_are_populated(a1_setup) -> None:
    """The draw reaches `max_ratio` times the radius, so both rows fill.

    Drawing only tiny perturbations would make the certificate look trivially
    sound: every trial inside the radius and unchanged.
    """
    vectors, table = a1_setup
    audit = certificate_audit(vectors, table, k=2, seed=3, trials=40)

    assert audit.n_certified > 0, "the certified row was reached"
    assert audit.uncertified_unchanged + audit.uncertified_changed > 0, "and the uncertified one"
    assert audit.is_sound
    assert audit.is_conclusive


# ---------------------------------------------------------------------------
# summarise_values: the mean is summed exactly, and an empty sample says nothing
# ---------------------------------------------------------------------------
def test_the_mean_is_summed_exactly_rather_than_left_to_right() -> None:
    """The mean is computed with `math.fsum`, not a running accumulator.

    A margin sample spans many orders of magnitude, and a left-to-right
    accumulation over the sorted values cancels the small ones away. The
    contrast is written out because CPython 3.12's `sum` compensates floats.
    """
    sample = [1.0, 1e100, 1.0, -1e100]

    running = 0.0
    for value in sorted(sample):
        running += value
    assert running == 0.0, "the accumulation this guards against, on the sorted values"

    d = summarise_values("m", sample)
    assert d.mean == math.fsum(sample) / 4 == 0.5


def test_an_empty_sample_has_no_zero_share_rather_than_a_share_of_zero() -> None:
    """`n_zero / n` is 0/0 on an empty sample, so the share is NaN.

    No observations is no evidence about the exact-tie rate, whereas 0.0 would
    publish "no exact ties were seen" from a sample that saw nothing.
    """
    empty = summarise_values("m", [])
    assert empty.n == 0
    assert empty.n_zero == 0, "no observations, so none of them were ties"
    assert math.isnan(empty.share_zero)

    all_undefined = summarise_values("m", [math.nan, math.nan])
    assert math.isnan(all_undefined.share_zero), "an all-NaN sample is empty too"


def test_a_sample_of_one_zero_is_entirely_ties() -> None:
    """A single zero is a share of 1.0.

    This is the other end of the same ratio, so the NaN above is the empty
    case rather than the general one.
    """
    assert summarise_values("m", [0.0]).share_zero == 1.0


def test_the_published_record_names_the_percentile_method() -> None:
    """The published record names its percentile method.

    Nearest-rank and interpolated percentiles disagree wherever the distribution
    has an atom, and a margin distribution has one at exactly zero. A reader
    recomputing a p50 with numpy's default needs to know the convention.
    """
    recorded = summarise_values("m", [0.0, 1.0]).as_dict()
    assert recorded["percentile_method"] == "nearest-rank (no interpolation)"


def test_a_signed_zero_does_not_make_the_extremes_depend_on_arrival_order() -> None:
    """Signed zeros must not let arrival order set the reported extremes.

    `-0.0 == 0.0` and `sorted` is stable, so without a sign-aware key the first
    zero becomes `minimum` and the last `maximum`, changing the digest. `==`
    holds either way, so the assertions compare bits.
    """
    forward = summarise_values("m", [-0.0, 0.0])
    reversed_ = summarise_values("m", [0.0, -0.0])

    assert same_bits(forward.minimum, reversed_.minimum)
    assert same_bits(forward.maximum, reversed_.maximum)

    # Each end follows IEEE-754 `totalOrder`: `-0.0` is the minimum and `+0.0`
    # the maximum, matching IEEE 754-2019 `minimum`/`maximum`.
    assert same_bits(forward.minimum, -0.0)
    assert same_bits(forward.maximum, 0.0)

    # The order is total across the whole sample, not only at the ends, so a
    # percentile landing on zero is fixed too.
    sample = [1.0, 0.0, -1.0, -0.0]
    assert same_bits(
        summarise_values("m", sample).percentiles["p50"],
        summarise_values("m", list(reversed(sample))).percentiles["p50"],
    )

    # Both signs count as zero: the sort key must not change `n_zero`.
    assert forward.n_zero == 2


def test_the_record_carries_the_undefined_count_beside_the_statistics() -> None:
    """A summary over mostly undefined values has to be visibly thin.

    Without `n_nan` the statistics would present a p50 over three observations
    exactly as one over three hundred.
    """
    recorded = summarise_values("m", [1.0, math.nan, math.nan, math.nan]).as_dict()

    assert recorded["n"] == 1
    assert recorded["n_nan"] == 3
    assert recorded["share_zero"] == 0.0, "one observation, and it was not a tie"


def test_an_all_nan_summary_reports_every_percentile_as_undefined() -> None:
    """An all-NaN summary keeps every percentile key, each set to NaN.

    A reader diffing two runs' records needs the same shape from both.
    """
    recorded = summarise_values("m", [math.nan]).as_dict()

    assert set(recorded["percentiles"]) == set(
        summarise_values("m", [1.0]).as_dict()["percentiles"]
    )
    assert all(math.isnan(v) for v in recorded["percentiles"].values())


# ---------------------------------------------------------------------------
# percentile: a positive percentile whose rank underflows to zero
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [5e-324, 1e-320, 1e-300])
def test_a_positive_percentile_too_small_to_reach_rank_one_returns_the_minimum(
    p: float,
) -> None:
    """A positive percentile whose rank underflows to zero returns the minimum.

    The guards only reject `p <= 0`, so a subnormal positive percentile computes
    `ceil(p / 100 * n)`, which underflows to 0. Without `max(1, rank)` the
    index is `-1`, which Python reads as the sample maximum.
    """
    sample = [1.0, 2.0, 3.0, 4.0]
    assert percentile(sample, p) == 1.0
    assert percentile(sample, p) != sample[-1], "the wrong end is what the clamp prevents"


def test_the_rank_of_such_a_percentile_really_does_underflow() -> None:
    """The rank of a subnormal percentile underflows to zero.

    If `p / 100 * n` stayed positive the clamp above would guard nothing and
    the test above would pass for the wrong reason.
    """
    assert math.ceil(5e-324 / 100.0 * 4) == 0
    assert math.ceil(1.0 / 100.0 * 4) == 1, "an ordinary percentile reaches rank one"


def test_the_summary_reports_both_ends_of_the_sample() -> None:
    """`minimum` and `maximum` come off opposite ends of the sorted values.

    A summary whose maximum equals its minimum reads as a degenerate
    distribution, which is a finding this project reports, so it must not be
    an artefact of the summariser.
    """
    d = summarise_values("m", [3.0, 1.0, 2.0])

    assert d.minimum == 1.0
    assert d.maximum == 3.0
    assert d.minimum != d.maximum, "three distinct values are not a degenerate sample"


def test_the_default_percentiles_keep_both_tails() -> None:
    """The default percentile grid keeps p1 and p99 as well as p0 and p100.

    The mass of interest in a margin distribution is at the bottom, and a grid
    starting at p5 would not show it. p0 and p100 restate min and max under the
    same nearest-rank convention.
    """
    assert DEFAULT_PERCENTILES == (0, 1, 5, 25, 50, 75, 95, 99, 100)

    keys = summarise_values("m", [1.0, 2.0]).percentiles
    assert list(keys) == [f"p{p}" for p in DEFAULT_PERCENTILES]


def test_the_first_percent_of_a_large_sample_is_not_the_minimum() -> None:
    """`p <= 0.0` is a shortcut for the bottom of the scale, not of the sample.

    On 200 observations the first percentile is the second element, and a
    guard that swallowed everything up to p1 would report the minimum instead.
    """
    sample = [float(i) for i in range(200)]

    assert percentile(sample, 1.0) == 1.0
    assert percentile(sample, 0.0) == 0.0, "the scale's bottom is still the minimum"


def test_the_two_shortcut_guards_agree_with_the_general_case_at_their_boundaries() -> None:
    """The two shortcut guards agree with the rank arithmetic at the ends.

    `p <= 0` and `p >= 100` return an end directly: `ceil(0) = 0` clamps to
    rank one, and `ceil(100/100 * n) = n` is the last rank. The branches are
    shortcuts, not corrections, so a formula change that broke this is visible.
    """
    sample = [1.0, 2.0, 3.0, 4.0]
    n = len(sample)

    assert math.ceil(0.0 / 100.0 * n) == 0, "clamped to rank one by max(1, rank)"
    assert math.ceil(100.0 / 100.0 * n) == n
    assert percentile(sample, 0.0) == sample[max(1, 0) - 1]
    assert percentile(sample, 100.0) == sample[n - 1]


# ---------------------------------------------------------------------------
# The transition curve has to transition
# ---------------------------------------------------------------------------
def test_perturbations_well_outside_the_radius_do_flip_the_top_k(a1_setup) -> None:
    """Perturbations well outside the radius flip the top-k.

    Section 4.4 guarantees no flip inside the radius. A counter that never
    counted would satisfy that and the monotonicity check, since `0 >= 0`
    holds, so a curve flat at zero everywhere must fail here.
    """
    vectors, table = a1_setup
    points, n_used, _ = transition_curve(vectors, table, 2, seed=1, trials=25)

    assert n_used > 0, "the curve is over an empty query set"
    by_ratio = {p.ratio: p for p in points}

    assert by_ratio[20.0].n_flips > 0, "a twentyfold perturbation must move the top-k"
    assert by_ratio[5.0].n_flips > 0
    assert by_ratio[20.0].flip_rate > by_ratio[1.1].flip_rate, "and the curve rises"


def test_the_curve_is_flat_at_zero_across_the_whole_certified_region(a1_setup) -> None:
    """Every certified ratio is flat at zero, not merely the smallest.

    The certificate is an interval, so one probe below the radius would leave
    the rest of it untested.
    """
    vectors, table = a1_setup
    points, _, _ = transition_curve(vectors, table, 2, seed=1, trials=25)
    inside = [p for p in points if p.within_certificate]

    assert len(inside) >= 4, "the grid samples the certified region more than once"
    assert all(p.n_flips == 0 for p in inside)
    assert all(p.n_trials > 0 for p in inside), "and each of them actually ran"


def test_the_default_ratio_grid_straddles_the_certified_radius() -> None:
    """The default ratio grid samples both sides of 1.0 and the point itself.

    The grid is the x-axis of section 7.3's transition plot. The points at 0.99
    and 1.01 make the boundary visible rather than inferred from a gap between
    0.9 and 1.1.
    """
    assert 1.0 in DEFAULT_RATIOS, "the certified radius itself is sampled"
    assert min(DEFAULT_RATIOS) < 1.0 < max(DEFAULT_RATIOS)
    assert 0.99 in DEFAULT_RATIOS, "just inside, where the certificate still holds"
    assert 1.01 in DEFAULT_RATIOS, "and just outside, where it no longer does"
    assert list(DEFAULT_RATIOS) == sorted(DEFAULT_RATIOS), "an unsorted grid plots as a zigzag"
    assert len([r for r in DEFAULT_RATIOS if r < 1.0]) >= 3, "the certified side is sampled too"


def test_an_audit_that_ran_no_trials_reports_every_cell_as_zero(a1_setup) -> None:
    """The 2x2 table starts empty and is only ever incremented.

    When every query is excluded, here because `k` is past the end of all of
    them, all four cells read zero. A non-zero start would put trials that
    never ran into the published conservatism.
    """
    vectors, table = a1_setup
    audit = certificate_audit(vectors, table, k=999, seed=1, trials=5)

    assert audit.n_undefined == len(vectors), "nothing was trialled"
    assert audit.certified_unchanged == 0
    assert audit.certified_changed == 0
    assert audit.uncertified_unchanged == 0
    assert audit.uncertified_changed == 0
    assert sum(audit.as_dict()[k] for k in ("certified_unchanged", "certified_changed")) == 0


def test_an_audit_that_met_no_exact_ties_records_none() -> None:
    """`n_exact_tie` defaults to zero.

    A2's regime is the exception rather than the rule. Any other default would
    report exclusions that never happened on every audit that omits the field.
    """
    audit = CertificateAudit(
        certified_unchanged=3,
        certified_changed=0,
        uncertified_unchanged=1,
        uncertified_changed=2,
        n_undefined=0,
    )

    assert audit.n_exact_tie == 0
    assert audit.as_dict()["n_exact_tie"] == 0
    assert audit.is_sound, "no certified change, so section 4.4 holds here"


# ---------------------------------------------------------------------------
# measure_noise_floor: what an instrument reports when it found nothing
# ---------------------------------------------------------------------------
def _diverging_corpus() -> tuple[object, list[object]]:
    """A corpus long enough that the norm summation diverges.

    Local by house convention, and the same construction the divergence test
    above uses: short documents keep every policy in exact agreement, which
    would make the measurements below vacuous.
    """
    rng = random.Random(11)
    vocab = [f"t{i}" for i in range(150)]
    documents = [[rng.choice(vocab) for _ in range(80)] for _ in range(50)]
    model = TfidfVectoriser().fit(documents, [f"d{i}" for i in range(50)])
    queries = [
        TfidfVectoriser.transform_query([rng.choice(vocab) for _ in range(40)], model)
        for _ in range(5)
    ]
    return model, queries


def test_a_policy_that_never_strayed_reports_no_error_rather_than_an_inherited_one() -> None:
    """A policy that never strays reports exactly zero error.

    The accumulators start at zero and only ever rise. Neumaier is that policy
    here, and the one the project recommends, so a non-zero starting value
    would give it a fabricated error bar.
    """
    model, queries = _diverging_corpus()
    by_policy = {p.policy: p for p in measure_noise_floor(model, queries).per_policy}
    clean = by_policy["neumaier"]

    assert clean.n_differing == 0, "the premise: this policy strayed nowhere"
    assert clean.max_abs == 0.0
    assert clean.max_ulps == 0.0


def test_every_instrument_compares_every_score_it_was_given() -> None:
    """Each policy sees one comparison per (query, document) pair.

    `n_compared` is the denominator of every rate the noise floor publishes,
    so a counter that skipped would inflate each of them.
    """
    model, queries = _diverging_corpus()
    floor = measure_noise_floor(model, queries)
    expected = len(queries) * model.n_documents

    assert expected == 250
    for policy in floor.per_policy:
        assert policy.n_compared == expected, policy.policy
        assert policy.n_differing <= policy.n_compared


def test_the_recorded_error_is_a_difference_and_not_a_sum() -> None:
    """The recorded error is `abs(a - b)`, not a sum.

    Cosine scores live in [0, 1], so a sum of the two is near 1 rather than
    near zero, and still positive and still rising with corpus size. A genuine
    floor is many orders of magnitude below the scores it is measured on.
    """
    model, queries = _diverging_corpus()
    floor = measure_noise_floor(model, queries)
    naive = {p.policy: p for p in floor.per_policy}["naive"]

    assert naive.n_differing > 0, "the premise: something strayed"
    assert 0.0 < naive.max_abs < 1e-12, "a floor, not a score"
    assert floor.eta == naive.max_abs


def test_each_cell_of_the_audit_reads_its_own_category(a1_setup) -> None:
    """Each cell of the audit reads its own category.

    The table is indexed by `(certified, unchanged)`. A cell reading a
    neighbour's key still gives four non-negative counts, so each cell is pinned
    by the ratio it feeds: `conservatism` has `uncertified_unchanged` on top.
    """
    vectors, table = a1_setup
    audit = certificate_audit(vectors, table, k=2, seed=7, trials=4)

    assert (audit.certified_unchanged, audit.certified_changed) == (6, 0)
    assert (audit.uncertified_unchanged, audit.uncertified_changed) == (4, 6)

    assert audit.n_certified == 6, "the certified row is its two cells"
    assert audit.conservatism == 4 / 10, "and conservatism is the uncertified row's share"
    assert audit.is_sound, "no certified perturbation changed the top-k"


# ---------------------------------------------------------------------------
# measure_noise_floor: the ulp distance is a magnitude, not a signed offset
# ---------------------------------------------------------------------------
def _asymmetrically_straying_corpus() -> tuple[object, list[object]]:
    """A corpus whose naive ulp offsets are asymmetric: -6 at worst, +2 at most.

    Local by house convention. Seed 2 is load-bearing: on
    :func:`_diverging_corpus` the strays are symmetric, so signed and unsigned
    readings agree. Rationale: docs/engineering_notes.md#asymmetric-stray-seed
    """
    rng = random.Random(2)
    vocab = [f"t{i}" for i in range(120)]
    documents = [[rng.choice(vocab) for _ in range(60)] for _ in range(20)]
    model = TfidfVectoriser().fit(documents, [f"d{i}" for i in range(20)])
    queries = [
        TfidfVectoriser.transform_query([rng.choice(vocab) for _ in range(30)], model)
        for _ in range(3)
    ]
    return model, queries


def test_a_stray_below_the_exact_value_counts_as_far_as_one_above() -> None:
    """`ulps_between` is signed, so a `max` over its results is not a distance.

    On this corpus the naive offsets run from -6 to +2, and a signed `max` from
    zero keeps only the +2. The recorded `max_ulps` must be 6, checked against
    a recomputation rather than the literal so a corpus shift does not break it.
    """
    from tfidf_stability.utils.numerics import Reduction, bits_of, ulps_between

    model, queries = _asymmetrically_straying_corpus()
    documents = [model.document(i) for i in range(model.n_documents)]
    exact_norms = model.matrix.row_norms(Reduction.EXACT)
    policy_norms = model.matrix.row_norms(Reduction.NAIVE)

    signed: list[float] = []
    for query in queries:
        exact = cosine_against_corpus(query, documents, exact_norms, Reduction.EXACT)
        got = cosine_against_corpus(query, documents, policy_norms, Reduction.NAIVE)
        signed += [
            ulps_between(a, b) for a, b in zip(got, exact, strict=True) if bits_of(a) != bits_of(b)
        ]

    assert signed, "the premise: this corpus makes the naive policy stray at all"
    largest_signed = max(signed)
    true_distance = max(abs(offset) for offset in signed)
    assert true_distance > largest_signed, (
        "the premise this test exists for: the strays must be asymmetric, or the "
        "signed and unsigned readings agree and the assertion below is vacuous"
    )

    recorded = {p.policy: p for p in measure_noise_floor(model, queries).per_policy}
    assert recorded[str(Reduction.NAIVE)].max_ulps == true_distance


def test_a_policy_that_strayed_cannot_report_a_zero_ulp_distance() -> None:
    """The three fields describe one set of differences and agree on emptiness.

    Weaker than the test above and true on every corpus: a record showing
    `n_differing = 38` beside `max_ulps = 0.0` contradicts itself whatever the
    cause.
    """
    for build in (_diverging_corpus, _asymmetrically_straying_corpus):
        model, queries = build()
        for policy in measure_noise_floor(model, queries).per_policy:
            where = f"{build.__name__}/{policy.policy}"
            assert policy.max_ulps >= 0.0, f"{where}: a distance cannot be negative"
            if policy.n_differing:
                assert policy.max_ulps > 0.0, f"{where}: {policy.n_differing} values differed"
                assert policy.max_abs > 0.0, f"{where}: and they differed by something"
            else:
                assert policy.max_ulps == 0.0, f"{where}: nothing differed"
                assert policy.max_abs == 0.0, f"{where}: nor by any absolute amount"
