"""The Stage 11 experiment harness.

These tests check that the harness **computes what it claims**, not that the
results are interesting. A harness that always reported "no flips" would produce
a beautiful A1 figure and prove nothing, so the tests here mostly construct
inputs whose answer is known by hand and check the harness finds it -- including
the cases where the correct answer is a failure.
"""

from __future__ import annotations

import json
import math

import pytest

from tfidf_stability.analysis.noise_floor import (
    NoiseFloor,
    PolicyError,
    measure_noise_floor,
    tau_band,
    verify_band_invariance,
)
from tfidf_stability.analysis.stability_profile import certificate_audit, transition_curve
from tfidf_stability.analysis.summarise import (
    ExperimentResult,
    percentile,
    summarise_values,
)
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.io import canonical_json
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
    """Every reported percentile must be an observation that actually occurred.

    NumPy's default would return 2.5 for the median here. That value is not any
    observed margin, so it could not be looked up in the raw data or compared
    with `same_bits` -- see the module docstring in `analysis/summarise.py`.
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
    """NaN marks an *undefined* quantity (G16), never a measurement of zero."""
    d = summarise_values("m", [1.0, 2.0, math.nan, 3.0])
    assert d.n == 3
    assert d.n_nan == 1
    assert d.mean == 2.0


def test_the_exact_tie_share_is_reported_separately() -> None:
    """It is G3's headline statistic and vanishes into the percentiles otherwise."""
    d = summarise_values("m", [0.0, 0.0, 0.0, 1.0])
    assert d.share_zero == 0.75
    assert d.percentiles["p50"] == 0.0


def test_an_all_nan_sample_summarises_without_raising() -> None:
    d = summarise_values("m", [math.nan, math.nan])
    assert d.n == 0
    assert d.n_nan == 2
    assert math.isnan(d.mean)


# ---------------------------------------------------------------------------
# The tau band
# ---------------------------------------------------------------------------
def test_tau_floor_is_exactly_twice_eta() -> None:
    """The factor 2 is the margin error e_i + e_j, not a safety fudge."""
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

    That is a finding about the corpus -- A1's and A2's regimes are not separable
    on it -- and the code must say so instead of returning a plausible number.
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
    """The norm summation is the dominant error source and is easy to miss.

    `TfidfModel.norms` is precomputed under the model's own reduction, so passing
    a different policy to `cosine_against_corpus` alone would hold the norms
    fixed. A dot product runs over a handful of shared terms; a norm runs over
    the whole document vector. Measuring only the former understates the floor.
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
    """Section 4.4's guarantee. A single flip here falsifies the theorem."""
    vectors, table = a1_setup
    points, _, _ = transition_curve(vectors, table, 2, seed=1, trials=25)
    inside = [p for p in points if p.within_certificate]
    assert inside, "the ratio grid must sample below the certified radius"
    assert all(p.n_flips == 0 for p in inside)


def test_the_flip_rate_is_monotone_enough_to_be_a_transition(a1_setup) -> None:
    """Not strictly monotone -- it is sampled -- but it must end above it starts."""
    vectors, table = a1_setup
    points, _, _ = transition_curve(vectors, table, 2, seed=1, trials=25)
    assert points[-1].flip_rate >= points[0].flip_rate


def test_exact_tie_queries_are_excluded_from_the_a1_curve(a1_setup) -> None:
    """At m_k = 0 the outcome is decided by the tie-break: that is A2, not A1.

    Averaging those queries into an A1 curve would let a tie-break effect be read
    as a numerical-stability effect.
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
    """A local Random, so an experiment cannot be perturbed by unrelated code."""
    import random

    vectors, table = a1_setup
    random.seed(99)
    before = random.random()
    random.seed(99)
    transition_curve(vectors, table, 2, seed=1, trials=5)
    assert random.random() == before


def test_the_certificate_is_sound(a1_setup) -> None:
    """Certified-but-changed must be zero. Any other value falsifies section 4.4."""
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
    """`False` from the certificate means "not covered", never "will change"."""
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
    """NaN and Infinity are not JSON, and both occur here for good reasons.

    An undefined margin is reported as NaN rather than coerced to a number
    (G16), and ``g_min`` is infinite when a corpus has no strictly-positive
    score gap. Python's encoder emits bare ``NaN``/``Infinity`` tokens, which its
    own loader accepts and every strict parser -- JavaScript, jq, Go, Rust --
    rejects. Both real experiment result files were unparseable that way.
    """
    text = canonical_json({"undefined": math.nan, "unbounded": math.inf, "fine": 1.5})
    parsed = _strict_loads(text)
    assert parsed == {"undefined": None, "unbounded": None, "fine": 1.5}


def test_non_finite_values_are_sanitised_at_every_depth() -> None:
    """Nested, because a margin summary is several levels down in a payload."""
    text = canonical_json({"a": [{"b": [math.nan, {"c": -math.inf}]}]})
    assert _strict_loads(text) == {"a": [{"b": [None, {"c": None}]}]}


def test_a_full_experiment_result_serialises_to_strict_json() -> None:
    """The end-to-end property: a written result must be machine-readable."""
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
    """`tau_floor` is 0 whenever every reduction policy was exactly
    correctly-rounded, which is measured behaviour for Neumaier -- not a
    hypothetical. Taking log10 of it raised ValueError."""
    scores = [[1.0, 0.5, 0.25]]
    band = tau_band(_floor(0.0), scores)
    assert band.tau_floor == 0.0
    assert verify_band_invariance(band, scores)
    assert math.isfinite(band.display_tau())


def test_an_unbounded_band_never_yields_an_infinite_tau() -> None:
    """With every score identical there is no positive gap, so `g_min` is
    infinite. The geometric midpoint would be `inf`, and at `tau = inf` every
    tie ball swallows the corpus -- an unusable tolerance to hand a caller."""
    scores = [[0.5, 0.5, 0.5]]
    band = tau_band(_floor(1e-16), scores)
    assert math.isinf(band.g_min)
    assert math.isfinite(band.display_tau())
    assert band.display_tau() >= band.tau_floor
