"""Cosine similarity: edge cases and the perturbation bounds of section 4.

The property tests hand the README section 4 inequalities to Hypothesis and ask
for a counterexample instead of spot-checking a few examples.
"""

from __future__ import annotations

import math
import sys

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tfidf_stability.similarity.cosine import cosine, cosine_against_corpus, cosine_matrix
from tfidf_stability.similarity.geometry import (
    corpus_lipschitz_bound,
    difference_norm,
    lipschitz_constant,
    norm_lower_bound,
    three_term_bound,
    unit,
)
from tfidf_stability.utils.numerics import Reduction
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


@given(sparse_map, sparse_map)
def test_cosine_is_within_zero_and_one(a: dict[int, float], b: dict[int, float]) -> None:
    """Guaranteed only because TF-IDF coordinates are non-negative (section 2.3)."""
    c = cosine(sv(a), sv(b))
    assert -1e-12 <= c <= 1.0 + 4 * math.ulp(1.0)


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


@given(sparse_map, sparse_map)
def test_dot_product_is_symmetric(a: dict[int, float], b: dict[int, float]) -> None:
    assert dot(sv(a), sv(b)) == dot(sv(b), sv(a))


# ---------------------------------------------------------------------------
# Section 4.3: the explicit Lipschitz bound (spec_addenda G4)
# ---------------------------------------------------------------------------
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
