"""Cosine similarity: edge cases and the perturbation bounds of section 4.

The property tests hand the README section 4 inequalities to Hypothesis and ask
for a counterexample instead of spot-checking a few examples.
"""

from __future__ import annotations

import dataclasses
import math
import sys

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tfidf_stability.similarity.cosine import cosine, cosine_against_corpus, cosine_matrix
from tfidf_stability.similarity.geometry import (
    LipschitzBound,
    corpus_lipschitz_bound,
    difference_norm,
    lipschitz_constant,
    norm_lower_bound,
    three_term_bound,
    unit,
)
from tfidf_stability.utils.numerics import Reduction, same_bits
from tfidf_stability.vectorisation.sparse import SparseVector, dot, l2_norm

DIM = 24


def sv(mapping: dict[int, float], dim: int = DIM) -> SparseVector:
    return SparseVector.from_mapping(mapping, dim)


# ---------------------------------------------------------------------------
# The zero-vector convention (section 2.3)
# ---------------------------------------------------------------------------
def test_zero_vector_similarity_is_zero_not_nan() -> None:
    """0/0 must never escape as NaN: a NaN in a sort key is undefined behaviour."""
    z = SparseVector.zero(DIM)
    a = sv({0: 1.0, 3: 2.0})
    assert cosine(z, a) == 0.0
    assert cosine(a, z) == 0.0
    assert cosine(z, z) == 0.0
    assert not math.isnan(cosine(z, z))


def test_disjoint_supports_give_exactly_zero() -> None:
    assert cosine(sv({0: 1.0}), sv({5: 1.0})) == 0.0


def test_self_similarity_is_one_to_within_a_few_ulp() -> None:
    """dot/(n*n) rounds three times, so the result misses 1 by a few ulp."""
    a = sv({0: 1.0, 3: 2.0, 7: 0.5})
    assert abs(cosine(a, a) - 1.0) <= 4 * math.ulp(1.0)


def test_similarity_is_symmetric_bitwise() -> None:
    a, b = sv({0: 1.0, 2: 3.0}), sv({0: 2.0, 2: 1.0, 5: 4.0})
    assert cosine(a, b) == cosine(b, a)


def test_cosine_matrix_is_exactly_symmetric() -> None:
    vs = [sv({0: 1.0, 1: 2.0}), sv({1: 1.0}), SparseVector.zero(DIM)]
    m = cosine_matrix(vs)
    assert all(m[i][j] == m[j][i] for i in range(3) for j in range(3))


def test_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine(sv({0: 1.0}, 4), sv({0: 1.0}, 5))


def test_corpus_scoring_handles_zero_documents_and_zero_query() -> None:
    docs = [sv({0: 1.0}), SparseVector.zero(DIM)]
    norms = [l2_norm(docs[0]), 0.0]
    assert cosine_against_corpus(sv({0: 1.0}), docs, norms) == [pytest.approx(1.0), 0.0]
    # A zero query is degenerate but legitimate: every score is 0 (spec_addenda G3).
    assert cosine_against_corpus(SparseVector.zero(DIM), docs, norms) == [0.0, 0.0]


def test_precomputed_norms_change_nothing_numerically() -> None:
    a, b = sv({0: 1.0, 2: 3.0}), sv({0: 2.0, 2: 1.0})
    assert cosine(a, b) == cosine(a, b, u_norm=l2_norm(a), v_norm=l2_norm(b))


# ---------------------------------------------------------------------------
# Property tests over non-negative sparse vectors
# ---------------------------------------------------------------------------
nonneg = st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False)
sparse_map = st.dictionaries(st.integers(0, DIM - 1), nonneg, min_size=0, max_size=DIM)


@pytest.mark.property
@given(sparse_map, sparse_map)
def test_cosine_is_within_zero_and_one(a: dict[int, float], b: dict[int, float]) -> None:
    """Guaranteed only because TF-IDF coordinates are non-negative (section 2.3)."""
    c = cosine(sv(a), sv(b))
    assert -1e-12 <= c <= 1.0 + 4 * math.ulp(1.0)


@pytest.mark.property
@given(sparse_map)
def test_cosine_with_self_is_one_or_zero(a: dict[int, float]) -> None:
    v = sv(a)
    c = cosine(v, v)
    assert c == 0.0 if l2_norm(v) == 0.0 else abs(c - 1.0) <= 8 * math.ulp(1.0)


#: Below this magnitude a coordinate's square is subnormal, so
#: ``sqrt(sum of squares)`` loses precision while the coordinate itself stays
#: representable. The measured onset matches this threshold.
NORM_UNDERFLOW_THRESHOLD = math.sqrt(sys.float_info.min)  # ~1.49e-154


# A per-test `settings` REPLACES the profile's `suppress_health_check`, so naming
# only `filter_too_much` would drop the `too_slow` suppression the ci and nightly
# profiles set.
#
# The three `assume()` calls below exclude the low-norm regime (G18) and reject
# about 41% of examples (697 invalid of 1697 at max_examples=1000), close enough
# to Hypothesis's threshold that the check fires depending on machine, seed and
# library version: it passed locally and failed on the runner. Suppressing it
# cannot hide a counterexample, since rejections are counted before the test body
# runs.
@pytest.mark.property
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(sparse_map, sparse_map, st.floats(min_value=1e-6, max_value=1e6))
def test_cosine_is_invariant_under_positive_scaling(
    a: dict[int, float], b: dict[int, float], k: float
) -> None:
    """What makes the scikit-learn cross-check possible.

    The paper's tf normalisation rescales each document by one positive scalar,
    so it can move only the norms, and hence only the bounds of sections 4.2 and
    4.3.

    Scale invariance is a statement about the reals. In binary64 it holds only
    while the coordinates' squares stay normal, so vectors scaled below
    :data:`NORM_UNDERFLOW_THRESHOLD` are excluded; that regime is the low-norm
    instability of README section 6, tested below.
    """
    u, v = sv(a), sv(b)
    assume(l2_norm(u) > 0 and l2_norm(v) > 0)
    scaled = SparseVector(u.indices, tuple(x * k for x in u.values), u.dim)
    assume(math.isfinite(l2_norm(scaled)))
    # Both vectors have to clear the threshold: the assertion compares
    # cos(scaled, v) against cos(u, v), and scaling up can lift `scaled` clear
    # while `u` stays under. Hypothesis found u = 8.39e-160 with k = 177795, where
    # scaled lands at 1.49e-154 and the two cosines differ by 5e-7 against a
    # 1e-12 tolerance (the instability G18 measures).
    for vector in (u, scaled):
        assume(min(abs(x) for x in vector.values if x != 0.0) > NORM_UNDERFLOW_THRESHOLD)
    assert cosine(scaled, v) == pytest.approx(cosine(u, v), abs=1e-12)


def test_cosine_degrades_for_vectors_whose_squares_underflow() -> None:
    """Section 6's "cosine becomes unstable for low-norm vectors", made concrete.

    ``l2_norm`` is ``sqrt(sum of squares)``: a coordinate below ``sqrt(DBL_MIN)``
    squares into the subnormal range and loses bits, and further down the square
    flushes to zero, so a non-zero vector reports a norm of 0. A hypot-style
    rescaled norm would avoid it; section 6 forbids that class of stabilising
    transformation, so this pins where the specification's own limitation starts.
    """
    unit = sv({0: 1.0})
    assert pytest.approx(1.49e-154, rel=1e-2) == NORM_UNDERFLOW_THRESHOLD

    # Comfortably above the threshold: exact.
    assert cosine(sv({0: 1e-150}), unit) == 1.0
    assert cosine(sv({0: 1e-154}), unit) == 1.0

    # Below it: the norm is no longer exact, so self-similarity drifts.
    degraded = cosine(sv({0: 1e-155}), unit)
    assert degraded != 1.0
    assert abs(degraded - 1.0) < 1e-14, "degrades gradually at first"

    # Far below: the square flushes to zero, and a non-zero vector is reported
    # as orthogonal to everything.
    assert l2_norm(sv({0: 1e-170})) == 0.0
    assert cosine(sv({0: 1e-170}), unit) == 0.0


@pytest.mark.property
@given(sparse_map, sparse_map)
def test_dot_product_is_symmetric(a: dict[int, float], b: dict[int, float]) -> None:
    assert dot(sv(a), sv(b)) == dot(sv(b), sv(a))


# ---------------------------------------------------------------------------
# Section 4.3: the explicit Lipschitz bound (spec_addenda G4)
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(sparse_map, sparse_map, sparse_map, sparse_map)
def test_lipschitz_bound_is_never_violated(
    a: dict[int, float], b: dict[int, float], c: dict[int, float], d: dict[int, float]
) -> None:
    """Adversarial search for a counterexample to ``C = 1/L``.

    Tiny-norm vectors are excluded: evaluating the bound in binary64 there is
    dominated by its own rounding.
    """
    u, v, up, vp = sv(a), sv(b), sv(c), sv(d)
    norms = [l2_norm(x) for x in (u, v, up, vp)]
    assume(min(norms) > 1e-6)

    bound = lipschitz_constant(u, v, up, vp)
    assert bound.holds, (
        f"observed {bound.observed!r} exceeded uniform {bound.uniform!r} / "
        f"tight {bound.tight!r} (L={bound.min_norm!r})"
    )


def test_lipschitz_bound_is_close_to_attained() -> None:
    """Perturbing one vector along a direction orthogonal to the other nearly
    attains the tight form, so the bound is not vacuous."""
    u = sv({0: 1.0, 1: 1.0})
    v = sv({0: 1.0, 1: 1.0})
    best = 0.0
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        up = sv({0: 1.0, 1: 1.0 + eps})
        best = max(best, lipschitz_constant(u, v, up, v).tightness)
    assert 0.0 < best <= 1.0


def test_lipschitz_requires_nonzero_vectors() -> None:
    z, a = SparseVector.zero(DIM), sv({0: 1.0})
    with pytest.raises(ValueError, match="non-zero"):
        lipschitz_constant(z, a, a, a)


def test_norm_lower_bound_holds_for_tfidf_vectors(mini_model) -> None:  # type: ignore[no-untyped-def]
    """``||w_i|| >= 1/sqrt(nnz_i)``, from ``||tf||_1 == 1`` and ``idf >= 1``."""
    for i in range(mini_model.n_documents):
        nnz = mini_model.matrix.row(i).nnz
        if nnz == 0:
            continue
        assert mini_model.norms[i] >= norm_lower_bound(nnz) - 1e-12


def test_corpus_lipschitz_bound_dominates_every_pairwise_constant(mini_model) -> None:  # type: ignore[no-untyped-def]
    """``sqrt(max nnz)`` must bound the ``1/L`` of every non-degenerate pair."""
    nnzs = [mini_model.matrix.row(i).nnz for i in range(mini_model.n_documents)]
    corpus_c = corpus_lipschitz_bound(nnzs)
    live = [i for i in range(mini_model.n_documents) if mini_model.norms[i] > 0]
    worst = max(1.0 / mini_model.norms[i] for i in live)
    assert worst <= corpus_c + 1e-9


# ---------------------------------------------------------------------------
# Section 4.2: the three-term decomposition (spec_addenda G5)
# ---------------------------------------------------------------------------
#: Section 4.2 is an algebraic statement about Euclidean norms, and below
#: ``NORM_UNDERFLOW_THRESHOLD`` ``l2_norm`` fails to be one: the coordinate
#: squares are subnormal and a handful of bits survive (G18). Nightly found the
#: tight case ``tf = 0`` at ``1.5e-161``, where the inequality is an equality over
#: the reals and the computed sides differ by 0.5%, six significant bits
#: surviving the squaring. That is a fact about the norm, pinned by
#: ``test_cosine_degrades_for_vectors_whose_squares_underflow``; excluding the
#: band here keeps this test about section 4.2.
above_underflow = st.one_of(
    st.just(0.0),
    st.floats(NORM_UNDERFLOW_THRESHOLD, 1e3, allow_nan=False, allow_infinity=False),
)
normal_sparse_map = st.dictionaries(
    st.integers(0, DIM - 1), above_underflow, min_size=0, max_size=DIM
)


@pytest.mark.property
@given(normal_sparse_map, normal_sparse_map, st.floats(1.0, 10.0), st.floats(0.0, 2.0))
def test_three_term_bound_is_never_violated(
    tf_a: dict[int, float],
    tf_b: dict[int, float],
    idf_linf: float,
    didf_linf: float,
) -> None:
    """``||w' - w||`` must never exceed the section 4.2 decomposition.

    Constructed on a common index set with idf and idf' both bounded as given,
    the setting the inequality is stated in.

    Checking a real-arithmetic statement in binary64 needs three preconditions,
    each of which nightly falsified this test on before it was in place
    (``docs/spec_addenda.md#g27``): compute the bound from the realised idf
    perturbation, keep all four norms outside the subnormal-square band, and take
    the slack as the elementwise rounding of ``w`` rather than a fixed constant.
    """
    tf = sv(tf_a)
    tf_prime = sv(tf_b)
    delta_tf = SparseVector.from_mapping(
        {
            i: dict(zip(tf_prime.indices, tf_prime.values, strict=True)).get(i, 0.0)
            - dict(zip(tf.indices, tf.values, strict=True)).get(i, 0.0)
            for i in set(tf.indices) | set(tf_prime.indices)
        },
        DIM,
    )
    # Subtraction re-enters the band the strategy excludes: two coordinates both
    # above the threshold can differ by far less than it. Each norm must be zero
    # or faithfully representable.
    for vec in (tf, tf_prime, delta_tf):
        norm = l2_norm(vec)
        assume(norm == 0.0 or norm >= NORM_UNDERFLOW_THRESHOLD)

    # Realise the idf/idf' pair before computing the bound: the realised
    # perturbation can exceed the one asked for. Below half an ulp of
    # ``idf_linf`` the sum steps a whole ulp instead. Nightly found this at 100k
    # examples: idf_linf=10, didf_linf=1e-15, ulp(10)=1.78e-15, so the step was
    # 1.78x the request and the bound met a larger observation
    # (docs/perturbation_notes.md: assume on the realised delta).
    idf = dict.fromkeys(range(DIM), idf_linf)
    idf_p = dict.fromkeys(range(DIM), idf_linf + didf_linf)
    didf_realised = max(abs(idf_p[i] - idf[i]) for i in range(DIM))
    bound = three_term_bound(tf, delta_tf, idf_linf, didf_realised)

    w = SparseVector.from_mapping(
        {i: v * idf[i] for i, v in zip(tf.indices, tf.values, strict=True)}, DIM
    )
    w_p = SparseVector.from_mapping(
        {i: v * idf_p[i] for i, v in zip(tf_prime.indices, tf_prime.values, strict=True)}, DIM
    )
    observed = difference_norm(w, w_p)

    # ``w' - w`` is a fourth norm the comparison rests on, and the first to
    # underflow: it is smaller than ``tf`` by roughly the idf perturbation, so
    # ``tf`` can sit above the threshold while the difference does not. Nightly
    # found tf = sqrt(DBL_MIN) with didf = 1e-5, difference ~1.5e-159 with a
    # subnormal square though every input vector is faithful.
    assume(observed == 0.0 or observed >= NORM_UNDERFLOW_THRESHOLD)

    # ``observed`` is evaluated in binary64: every ``v * idf[i]`` rounds, so each
    # component of ``w' - w`` carries up to ``u*(|w_i| + |w'_i|)`` of error the
    # inequality does not model. The allowance is that quantity; a fixed absolute
    # slack is too loose for small vectors and, at the ulp-scale perturbations
    # above, unrelated to the magnitudes in play.
    u = 2.0**-53
    rounding = u * (idf_linf + didf_realised) * (l2_norm(tf) + l2_norm(tf_prime))
    assert observed <= bound * (1 + 1e-9) + rounding


# ---------------------------------------------------------------------------
# Reduction policies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(Reduction))
def test_every_reduction_policy_agrees_on_an_exact_case(policy: Reduction) -> None:
    """With exactly representable values every policy must give the same answer."""
    a = sv({0: 1.0, 1: 2.0, 2: 4.0})
    b = sv({0: 0.5, 1: 0.25, 2: 8.0})
    assert dot(a, b, policy) == 33.0


def test_naive_and_exact_reductions_can_differ() -> None:
    """The spread between policies is what measures the floating-point noise
    floor, so an equality here would leave the tau derivation of section 7.0
    measuring nothing.

    Each addend sits below half an ulp of the running total, so the naive
    left-fold discards every one, while their exact sum exceeds half an ulp and
    survives correct rounding. ``n_small`` addends of size ``e`` show this only
    when ``n_small * e > ulp(1)/2``; a smaller ``e`` makes both policies agree and
    the test vacuous.
    """
    n_small = DIM - 1
    small = 1e-17
    assert n_small * small > math.ulp(1.0) / 2, "addends too small to be observable"
    assert 1.0 + small == 1.0, "each addend must vanish individually"

    idx = tuple(range(DIM))
    a = SparseVector(idx, (1.0,) + (small,) * n_small, DIM)
    b = SparseVector(idx, (1.0,) * DIM, DIM)

    naive = dot(a, b, Reduction.NAIVE)
    exact = dot(a, b, Reduction.EXACT)
    assert naive == 1.0
    assert exact > naive
    assert naive != exact


def test_unit_vector_has_unit_norm() -> None:
    u = unit(sv({0: 3.0, 1: 4.0}))
    assert abs(l2_norm(u) - 1.0) <= 4 * math.ulp(1.0)
    assert unit(SparseVector.zero(DIM)).nnz == 0


def test_a_norm_supplied_for_every_document_is_required() -> None:
    """The norms are precomputed and passed in parallel to the documents, so a
    short list would silently pair each score with the wrong denominator rather
    than run out: `zip` stops at the shorter of the two."""
    docs = [sv({0: 1.0}), sv({1: 1.0}), sv({2: 1.0})]
    with pytest.raises(ValueError, match="3 documents but 2 norms"):
        cosine_against_corpus(sv({0: 1.0}), docs, [1.0, 1.0])


def test_a_difference_norm_across_dimensions_is_refused() -> None:
    """Two vectors of different dimension have no common space to subtract in.
    Padding the shorter one would return a number for a question with no answer.
    """
    with pytest.raises(ValueError, match="dimension mismatch"):
        difference_norm(sv({0: 1.0}, dim=DIM), sv({0: 1.0}, dim=DIM + 1))


def test_corpus_scoring_divides_by_both_norms_not_just_one() -> None:
    """Every existing corpus-scoring test uses unit vectors, where the whole
    denominator is 1.0 and the division is unobservable.

    Mutation testing found it: turning `dot / (q_norm * dn)` into either
    `dot * (q_norm * dn)` or `dot / (q_norm / dn)` left the suite green. On unit
    input all three agree, so the normalisation -- the one thing cosine does that
    a dot product does not -- was covered but never asserted.

    Pythagorean triples, so every norm here is exact in binary64 and the expected
    values can be written down rather than recomputed by the code under test.
    """
    query = sv({0: 3.0, 1: 4.0})  # norm 5
    docs = [sv({0: 5.0, 1: 12.0}), sv({0: 8.0, 1: 6.0})]  # norms 13 and 10
    norms = [13.0, 10.0]
    assert [l2_norm(d) for d in docs] == norms, "the premise: neither norm is 1"
    assert l2_norm(query) == 5.0

    scores = cosine_against_corpus(query, docs, norms)

    # dot = 3*5 + 4*12 = 63, over 5*13; and 3*8 + 4*6 = 48, over 5*10.
    assert scores == [63.0 / 65.0, 48.0 / 50.0]
    assert scores[0] != 63.0 * 65.0, "multiplying by the norms would pass on unit input"
    assert scores[0] != 63.0 / (5.0 / 13.0), "so would dividing one norm by the other"


def test_corpus_scoring_agrees_bitwise_with_scoring_each_document_alone() -> None:
    """The batched path and the pairwise one must not be two different formulas.
    Asserted on raw bit patterns: a denominator assembled in the other order
    would agree to within rounding and disagree here."""
    query = sv({0: 3.0, 1: 4.0, 2: 1.0})
    docs = [sv({0: 5.0, 1: 12.0}), sv({0: 8.0, 1: 6.0, 2: 2.0}), sv({2: 7.0})]
    norms = [l2_norm(d) for d in docs]

    batched = cosine_against_corpus(query, docs, norms)
    assert len(batched) == 3, "an empty comparison would assert nothing at all"
    for score, doc in zip(batched, docs, strict=True):
        assert same_bits(score, cosine(query, doc))


# ---------------------------------------------------------------------------
# The section 4.3 bound, by value rather than by inequality
# ---------------------------------------------------------------------------
# Everything above asserts that a bound holds. That is the property the paper
# claims, and it is the reason mutation testing scored this module at 49%: an
# inequality with slack is satisfied by a great many wrong constants, so
# `C = 1/L` became `C = 1*L`, the Dunkl-Williams denominators inverted, and
# `1/sqrt(nnz)` turned into `0/sqrt(nnz)` with the suite still green.
#
# These pin the arithmetic instead. Norms are powers of two and small integers
# throughout, so every expected value below is exact in binary64 and is written
# down rather than recomputed by the code under test.
def test_the_lipschitz_bound_reports_each_of_its_pieces_by_value() -> None:
    """One perturbation, worked by hand.

    ``u`` and ``u'`` are 4e0 and 8e0; ``v`` and ``v'`` are 16e0 and 32e0. So
    ``L = min(4, 16, 8, 32) = 4``, ``du = 4``, ``dv = 16``.
    """
    u, u_prime = sv({0: 4.0}), sv({0: 8.0})
    v, v_prime = sv({0: 16.0}), sv({0: 32.0})

    bound = lipschitz_constant(u, v, u_prime, v_prime)

    assert bound.min_norm == 4.0
    assert bound.constant == 0.25, "C = 1/L, not L and not 1"
    assert bound.uniform == 0.25 * 20.0, "C * (du + dv)"
    # Dunkl-Williams per vector: 2*du/(nu + nu') + 2*dv/(nv + nv').
    assert bound.tight == 8.0 / 12.0 + 32.0 / 48.0
    # Every vector here is a positive multiple of e0, so all four cosines are
    # exactly 1 and the perturbation moves the similarity not at all.
    assert bound.observed == 0.0
    assert bound.holds


def test_the_bound_object_cannot_be_edited_after_it_is_reported() -> None:
    """It is a measurement, and it travels into a run manifest. A caller that
    could adjust `observed` after the fact could make any bound hold."""
    bound = lipschitz_constant(sv({0: 4.0}), sv({0: 16.0}), sv({0: 8.0}), sv({0: 32.0}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        bound.observed = 0.0  # type: ignore[misc]
    assert not hasattr(bound, "__dict__"), "slots=True: no ad-hoc attributes either"


# ---------------------------------------------------------------------------
# `holds` has to be able to say no
# ---------------------------------------------------------------------------
def _bound(uniform: float, tight: float, observed: float) -> LipschitzBound:
    return LipschitzBound(
        constant=1.0, min_norm=1.0, uniform=uniform, tight=tight, observed=observed
    )


def test_a_violation_of_either_form_is_a_violation() -> None:
    """Both bounds are claimed, so satisfying one is not enough. With `or` in
    place of `and` every assertion of `holds` in this file still passed."""
    assert not _bound(uniform=10.0, tight=0.5, observed=1.0).holds, "the tight form fails"
    assert not _bound(uniform=0.5, tight=10.0, observed=1.0).holds, "the uniform form fails"
    assert not _bound(uniform=0.5, tight=0.5, observed=1.0).holds, "both fail"
    assert _bound(uniform=10.0, tight=10.0, observed=1.0).holds, "and neither, here"


@pytest.mark.parametrize("scale", [0.5, 4.0])
def test_the_rounding_slack_is_admitted_exactly_and_not_a_hair_further(scale: float) -> None:
    """The slack covers rounding incurred while evaluating the bound, so an
    observed value sitting exactly on it is inside.

    Two scales because the slack is `1e-12 * max(1.0, uniform) + 1e-15`: below 1
    the `max` decides it and above 1 the multiplication does, and a mutation of
    either is invisible at the scale that does not exercise it.
    """
    slack = 1e-12 * max(1.0, scale) + 1e-15
    assert _bound(uniform=scale, tight=scale, observed=scale + slack).holds

    beyond = scale + slack + max(1e-9, 1e-9 * scale)
    assert not _bound(uniform=scale, tight=scale, observed=beyond).holds


def test_tightness_is_the_ratio_and_is_zero_where_there_is_no_bound() -> None:
    """`observed / tight`, not the product: on the near-attained case both sit
    close to 1 and the two are hard to tell apart."""
    assert _bound(uniform=9.0, tight=4.0, observed=3.0).tightness == 0.75
    # A zero tight form has no ratio. Guarding on `> 0` rather than `>= 0` is
    # what keeps this from dividing by zero.
    assert _bound(uniform=1.0, tight=0.0, observed=1.0).tightness == 0.0


# ---------------------------------------------------------------------------
# The two closed-form bounds
# ---------------------------------------------------------------------------
def test_the_norm_lower_bound_is_one_over_root_nnz() -> None:
    """`||tf||_1 == 1` and `idf >= 1` termwise give `||w|| >= 1/sqrt(nnz)`.

    The existing test asserts a model's norms clear this with slack, which
    `0.0 / sqrt(nnz)` also satisfies. These are the values themselves.
    """
    assert norm_lower_bound(1) == 1.0
    assert norm_lower_bound(4) == 0.5
    assert norm_lower_bound(16) == 0.25
    assert norm_lower_bound(2) == 1.0 / math.sqrt(2)


def test_an_empty_support_has_no_lower_bound_rather_than_an_infinite_one() -> None:
    """The vector is zero there. Returning 0.0 keeps the bound true; dividing by
    sqrt(0) would raise, and the guard is what stops it."""
    assert norm_lower_bound(0) == 0.0
    assert norm_lower_bound(-1) == 0.0, "a negative support is degenerate, not an error"


def test_the_corpus_bound_is_the_root_of_the_largest_support() -> None:
    assert corpus_lipschitz_bound([1]) == 1.0
    assert corpus_lipschitz_bound([4, 1, 2]) == 2.0
    assert corpus_lipschitz_bound([9, 16]) == 4.0


def test_a_corpus_with_no_live_document_is_unbounded_rather_than_zero() -> None:
    """`C = 1/L` blows up as norms shrink, so the limit of an empty corpus is
    infinity. Zero would claim the best possible conditioning for the case with
    no evidence at all -- and section 6 reads this number as a warning."""
    assert corpus_lipschitz_bound([]) == math.inf
    assert corpus_lipschitz_bound([0, 0]) == math.inf
    assert corpus_lipschitz_bound([0, 4]) == 2.0, "the zeros are skipped, not counted"


def test_normalising_the_zero_vector_returns_it_rather_than_dividing_by_zero() -> None:
    """The one input `unit` cannot normalise. Returning it unchanged keeps the
    function total, which is what lets callers apply it across a corpus that
    contains an all-stopword document."""
    zero = SparseVector.zero(DIM)
    assert unit(zero) is zero

    scaled = unit(sv({0: 3.0, 1: 4.0}))
    assert scaled.values == (3.0 / 5.0, 4.0 / 5.0)


# ---------------------------------------------------------------------------
# unit: the two vectors it cannot normalise
# ---------------------------------------------------------------------------
def test_a_vector_whose_norm_underflows_is_returned_unnormalised() -> None:
    """The zero-vector branch is triggered by the norm, not by the support.

    A vector of a single smallest subnormal has stored entries and a norm of
    zero, so `unit` returns it unchanged -- still of "length" 5e-324 rather than
    length one. Silent, and the G18 regime reaching a second function.
    """
    tiny = sv({0: 5e-324})
    assert l2_norm(tiny) == 0.0
    assert tiny.nnz == 1

    normalised = unit(tiny)
    assert normalised.values == (5e-324,), "returned as-is rather than scaled"


def test_a_vector_whose_norm_overflows_normalises_to_all_zeros() -> None:
    """The norm is infinite, so every coordinate divided by it is zero. Not an
    error, and not a unit vector either -- the other end of the same problem."""
    huge = sv({0: 1e200, 1: 1e200})
    assert l2_norm(huge) == math.inf
    assert unit(huge).values == (0.0, 0.0)


def test_an_ordinary_vector_normalises_to_length_one() -> None:
    """The reference point the two degenerate cases are read against."""
    assert unit(sv({0: 3.0, 1: 4.0})).values == (0.6, 0.8)


# ---------------------------------------------------------------------------
# difference_norm
# ---------------------------------------------------------------------------
def test_a_vector_is_no_distance_from_itself() -> None:
    """Exactly zero, not merely small: every coordinate difference is an exact
    zero before the sum."""
    same = sv({0: 3.0, 1: 4.0})
    assert same_bits(difference_norm(same, same), 0.0)


def test_the_two_zeros_are_no_distance_apart() -> None:
    """`-0.0 - 0.0` is `-0.0`, and squaring it gives `+0.0`, so the sign does
    not survive into the norm."""
    assert same_bits(difference_norm(sv({0: -0.0}), sv({0: 0.0})), 0.0)


def test_disjoint_supports_are_the_norm_of_the_two_together() -> None:
    """The union is iterated in ascending index order, which is what makes this
    agree bit for bit with the native implementation."""
    assert same_bits(difference_norm(sv({0: 3.0}), sv({1: 4.0})), 5.0)


# ---------------------------------------------------------------------------
# lipschitz_constant: each of the four vectors in turn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_a_zero_vector_in_any_of_the_four_positions_is_refused(position: int) -> None:
    """The bound is `C = 1 / L` with `L` the minimum of the four norms, so a
    single zero makes it vacuous. Each position separately: a guard written
    against only the first pair would pass three of these.
    """
    vectors = [sv({0: 1.0}), sv({0: 1.0}), sv({0: 2.0}), sv({0: 2.0})]
    vectors[position] = SparseVector.zero(DIM)

    with pytest.raises(ValueError, match="requires four non-zero vectors"):
        lipschitz_constant(*vectors)


def test_the_zero_vector_guard_is_what_keeps_the_constant_finite() -> None:
    """`C = 1 / L`, so an infinite constant would need `L` below about
    `5.6e-309`. No such norm exists that is not zero: the sum of squares
    underflows first, and the smallest representable non-zero norm is around
    `1e-161`, whose reciprocal is a comfortable `1e161`.

    So the guard rejecting zero vectors is not one of several things standing
    between the bound and an infinity -- it is the only one, and the bound is
    finite for every input that gets past it.
    """
    smallest_live = sv({0: 1e-161})
    assert l2_norm(smallest_live) > 0.0

    bound = lipschitz_constant(smallest_live, smallest_live, smallest_live, smallest_live)
    assert math.isfinite(bound.constant)
    assert math.isfinite(bound.uniform)


def test_the_uniform_and_tight_forms_are_reported_separately() -> None:
    """Both travel on the result so a caller can say which one it quoted -- the
    same reason the certificate carries two radii."""
    bound = lipschitz_constant(sv({0: 4.0}), sv({0: 16.0}), sv({0: 8.0}), sv({0: 32.0}))
    assert bound.uniform != bound.tight
    assert bound.observed <= min(bound.uniform, bound.tight)


# ---------------------------------------------------------------------------
# norm_lower_bound at the large end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nnz", [2**20, 2**40, 2**53])
def test_the_lower_bound_stays_positive_for_any_support_a_corpus_could_have(nnz: int) -> None:
    """`1 / sqrt(nnz)`. Even at 2**53 distinct terms -- far beyond any real
    vocabulary -- it is a small positive number rather than underflowing, so the
    bound never degenerates into claiming nothing."""
    bound = norm_lower_bound(nnz)
    assert 0.0 < bound < 1.0
    assert bound == 1.0 / math.sqrt(nnz)
