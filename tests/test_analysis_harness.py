"""The Stage 11 experiment harness.

These check that the harness computes what it claims, saying nothing about
whether the results are interesting: one always reporting "no flips" would
produce a clean A1 figure and prove nothing. Most inputs here have an answer
known by hand, including the cases where the correct answer is a failure.
"""

from __future__ import annotations

import json
import math
import random

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
from tfidf_stability.perturbation.score_bounds import certified_radius
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
    assert math.isnan(d.mean)


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
    That is a finding about the corpus (A1's and A2's regimes do not separate on
    it), so the code must report it instead of returning a plausible number.
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
    """The norm summation dominates the error and is easy to miss.

    `TfidfModel.norms` is precomputed under the model's own reduction, so passing a
    different policy to `cosine_against_corpus` alone holds the norms fixed. A dot
    product runs over a handful of shared terms, a norm over the whole document
    vector, so measuring only the dot product understates the floor.
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
    """Sampled, so not strictly monotone; it must still end above where it starts."""
    vectors, table = a1_setup
    points, _, _ = transition_curve(vectors, table, 2, seed=1, trials=25)
    assert points[-1].flip_rate >= points[0].flip_rate


def test_exact_tie_queries_are_excluded_from_the_a1_curve(a1_setup) -> None:
    """At m_k = 0 the tie-break decides the outcome, which is A2's regime.
    Averaging those queries into an A1 curve reads a tie-break effect as one of
    numerical stability.
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
    """`False` from the certificate means "not covered"; it claims nothing about
    whether the ranking changes."""
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
    """NaN and Infinity are outside JSON, and both occur here: an undefined margin
    is reported as NaN rather than coerced to a number (G16), and ``g_min`` is
    infinite when a corpus has no strictly-positive score gap. Python's encoder
    emits bare ``NaN``/``Infinity`` tokens, which its own loader accepts and every
    strict parser (JavaScript, jq, Go, Rust) rejects. Both real experiment result
    files were unparseable that way.
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
    """`tau_floor` is 0 whenever every reduction policy came out correctly
    rounded, measured behaviour for Neumaier. Taking log10 of it raised
    ValueError."""
    scores = [[1.0, 0.5, 0.25]]
    band = tau_band(_floor(0.0), scores)
    assert band.tau_floor == 0.0
    assert verify_band_invariance(band, scores)
    assert math.isfinite(band.display_tau())


def test_an_unbounded_band_never_yields_an_infinite_tau() -> None:
    """With every score identical there is no positive gap, so `g_min` is
    infinite. The geometric midpoint would be `inf`, and at `tau = inf` every tie
    ball swallows the corpus."""
    scores = [[0.5, 0.5, 0.5]]
    band = tau_band(_floor(1e-16), scores)
    assert math.isinf(band.g_min)
    assert math.isfinite(band.display_tau())
    assert band.display_tau() >= band.tau_floor


# ---------------------------------------------------------------------------
# The reporting blocks, which are what a manifest actually carries
# ---------------------------------------------------------------------------
# Every as_dict below lands in a published JSON file. The tests assert what the
# block guarantees rather than a literal key list: that it survives the canonical
# writer, and that every float is accompanied by the hex form that makes a
# comparison across machines exact. Pinning the literal dict would calcify the
# schema against any future field.
def test_a_policy_error_block_survives_the_canonical_writer() -> None:
    block = PolicyError("naive", 100, 25, 1e-16, 8.0).as_dict()
    assert json.loads(canonical_json(block, indent=None)) == block
    assert block["policy"] == "naive"


def test_the_share_of_differing_scores_is_the_ratio_it_claims() -> None:
    assert PolicyError("naive", 100, 25, 1e-16, 8.0).share_differing == 0.25


def test_a_policy_that_compared_nothing_reports_zero_rather_than_dividing() -> None:
    """An empty comparison is not evidence of agreement, but it must not raise
    while a report is being assembled."""
    assert PolicyError("exact", 0, 0, 0.0, 0.0).share_differing == 0.0


def test_a_noise_floor_block_carries_the_hex_form_of_every_float() -> None:
    """Decimal rendering loses the last bit, and the last bit is the subject."""
    block = _floor(1e-16).as_dict()
    assert json.loads(canonical_json(block, indent=None)) == block
    assert float.fromhex(block["eta_hex"]) == block["eta"]


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
    """log10(0) is not a number of decades, and -inf in a manifest is not a
    measurement."""
    band = tau_band(_floor(0.0), [[1.0, 0.5]])
    assert band.tau_floor == 0.0
    assert math.isnan(band.decades)


# ---------------------------------------------------------------------------
# The band probe
# ---------------------------------------------------------------------------
def test_a_probe_count_below_one_is_refused_rather_than_dividing_by_zero() -> None:
    """`i / (probes - 1)` divided by zero at probes = 1 and said nothing useful.
    A count below one is a caller error and now says so."""
    band = tau_band(_floor(1e-16), [[1.0, 0.5, 0.25]])
    with pytest.raises(ValueError, match="probes must be at least 1"):
        verify_band_invariance(band, [[1.0, 0.5, 0.25]], probes=0)


def test_a_single_probe_samples_the_lower_endpoint_rather_than_failing() -> None:
    """One probe cannot be spaced across a band, so it is the floor itself."""
    band = tau_band(_floor(1e-16), [[1.0, 0.5, 0.25]])
    assert verify_band_invariance(band, [[1.0, 0.5, 0.25]], probes=1)


def test_an_unbounded_band_is_probed_over_a_substitute_upper_end() -> None:
    """`g_min` is infinite when no strictly-positive gap exists. Probing to
    infinity would sample nothing, so a finite substitute is used instead."""
    band = tau_band(_floor(1e-16), [[1.0, 1.0, 1.0]])
    assert math.isinf(band.g_min), "the premise: every gap is exactly zero"
    assert verify_band_invariance(band, [[1.0, 1.0, 1.0]]) is not None


# ---------------------------------------------------------------------------
# The certificate audit's own honesty
# ---------------------------------------------------------------------------
def test_an_audit_that_drew_no_certified_perturbation_is_not_conclusive() -> None:
    """is_sound is `certified_changed == 0`, so an audit that certified nothing
    reports the theorem upheld having checked it zero times. An earlier version
    of the section 4.4 attack did exactly that."""
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
    """Zero would claim every uncertified perturbation changed the ranking,
    which is the opposite of what no evidence means."""
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
    """Strict: at exactly the certified radius the two scores meet, so the
    guarantee is for radii strictly below it."""
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
    """A defined-but-NaN certificate is the trap the second half of the guard
    catches.

    A NaN among the scores gives a certificate that reports itself defined while
    its radius is NaN. Every comparison against NaN is false, so `realised <
    radius` would answer "not certified" for every trial and the audit would
    silently report a conclusive-looking run built on nothing.
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
    """The measurement arm, which the six-document fixture never reaches.

    On the mini corpus every policy agrees with EXACT bit for bit, so the whole
    error-accumulation branch never executed and eta was always zero. A floor
    measured only where nothing strays is not a floor. Documents here are long
    enough that the norm summation, which is where the error accumulates,
    actually diverges.
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
    A tau above g_min merges two genuinely separated scores, so the shape moves
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
