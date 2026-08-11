"""Cosine similarity: edge cases and the perturbation bounds of section 4.

The property tests here do not merely confirm the inequalities of README section
4 on a few examples -- they hand the inequality to Hypothesis and ask it to find
a counterexample. A bound that has survived an adversarial search is worth
considerably more than one that has been spot-checked.
"""

from __future__ import annotations

import math
import sys

import pytest
from hypothesis import assume, given
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
    """Not exactly 1: dot/(n*n) rounds three times. The tolerance is derived."""
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


#: Below this magnitude a coordinate's *square* falls into the subnormal range,
#: so ``sqrt(sum of squares)`` loses precision even though the value itself is
#: perfectly representable. Measured onset agrees with the theory exactly.
NORM_UNDERFLOW_THRESHOLD = math.sqrt(sys.float_info.min)  # ~1.49e-154


@given(sparse_map, sparse_map, st.floats(min_value=1e-6, max_value=1e6))
def test_cosine_is_invariant_under_positive_scaling(
    a: dict[int, float], b: dict[int, float], k: float
) -> None:
    """The observation that makes the scikit-learn cross-check possible.

    The paper's tf normalisation rescales each document by a single positive
    scalar, so it cannot change any similarity -- only the norms, and hence only
    the bounds of sections 4.2 and 4.3.

    Scale invariance is a statement about the reals. In binary64 it holds only
    while the coordinates' *squares* stay normal, so vectors scaled below
    :data:`NORM_UNDERFLOW_THRESHOLD` are excluded -- not because the
    implementation is wrong there, but because it is *correctly* exhibiting the
    low-norm instability README section 6 describes. That regime has its own
    test below.
    """
    u, v = sv(a), sv(b)
    assume(l2_norm(u) > 0 and l2_norm(v) > 0)
    scaled = SparseVector(u.indices, tuple(x * k for x in u.values), u.dim)
    assume(math.isfinite(l2_norm(scaled)))
    assume(min(abs(x) for x in scaled.values if x != 0.0) > NORM_UNDERFLOW_THRESHOLD)
    assert cosine(scaled, v) == pytest.approx(cosine(u, v), abs=1e-12)


def test_cosine_degrades_for_vectors_whose_squares_underflow() -> None:
    """Section 6's "cosine becomes unstable for low-norm vectors", made concrete.

    ``l2_norm`` is ``sqrt(sum of squares)``, so a coordinate below
    ``sqrt(DBL_MIN)`` squares into the subnormal range and loses precision --
    catastrophically so further down, where the square flushes to zero entirely
    and a perfectly good vector reports a norm of 0.

    A hypot-style rescaled norm would avoid this, and section 6 explicitly
    forbids exactly that kind of stabilising transformation. So the behaviour is
    correct with respect to the specification, and this test pins *where* the
    specification's own limitation begins rather than pretending it does not
    exist.
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
# Section 4.3 -- the explicit Lipschitz bound (spec_addenda G4)
# ---------------------------------------------------------------------------
@given(sparse_map, sparse_map, sparse_map, sparse_map)
def test_lipschitz_bound_is_never_violated(
    a: dict[int, float], b: dict[int, float], c: dict[int, float], d: dict[int, float]
) -> None:
    """Adversarial search for a counterexample to ``C = 1/L``.

    Vectors with a tiny norm are excluded, not because the bound fails there but
    because evaluating it in binary64 becomes dominated by its own rounding.
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
    """A bound that is never tight would be useless; this one very nearly is.

    Perturbing one vector slightly along a direction orthogonal to the other
    makes the tight form near-attained.
    """
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
# Section 4.2 -- the three-term decomposition (spec_addenda G5)
# ---------------------------------------------------------------------------
@given(sparse_map, sparse_map, st.floats(1.0, 10.0), st.floats(0.0, 2.0))
def test_three_term_bound_is_never_violated(
    tf_a: dict[int, float],
    tf_b: dict[int, float],
    idf_linf: float,
    didf_linf: float,
) -> None:
    """``||w' - w||`` must never exceed the section 4.2 decomposition.

    Constructed on a common index set with idf and idf' both bounded as given,
    which is the setting the inequality is stated in.
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
    bound = three_term_bound(tf, delta_tf, idf_linf, didf_linf)

    # Realise a concrete idf/idf' pair respecting the supplied sup-norms.
    idf = dict.fromkeys(range(DIM), idf_linf)
    idf_p = dict.fromkeys(range(DIM), idf_linf + didf_linf)
    w = SparseVector.from_mapping(
        {i: v * idf[i] for i, v in zip(tf.indices, tf.values, strict=True)}, DIM
    )
    w_p = SparseVector.from_mapping(
        {i: v * idf_p[i] for i, v in zip(tf_prime.indices, tf_prime.values, strict=True)}, DIM
    )
    observed = difference_norm(w, w_p)
    assert observed <= bound * (1 + 1e-9) + 1e-12


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
    """The spread between policies is what measures the floating-point noise floor.

    If this ever became an equality the tau-derivation experiment of section 7.0
    would be measuring nothing, so the difference is asserted, not tolerated.

    The construction is deliberate: each addend is individually below half an ulp
    of the running total, so the naive left-fold discards every one of them, but
    their *exact* sum exceeds half an ulp and so survives correct rounding. Note
    that ``n_small`` addends of size ``e`` only demonstrate this when
    ``n_small * e > ulp(1)/2``; a smaller ``e`` would make both policies agree
    and the test would be vacuous.
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
