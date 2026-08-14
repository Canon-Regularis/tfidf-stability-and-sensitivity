"""Geometric quantities and the perturbation bounds of README section 4.

Section 4.3 states that

    |cos(u', v') - cos(u, v)| <= C (||u' - u|| + ||v' - v||)

"for a constant C depending on lower and upper bounds on the norms" and leaves
``C`` unspecified, which cannot be computed, tested or used. This module pins it
and proves it. See ``docs/spec_addenda.md#g4``.

**Theorem.** For non-zero ``u, v, u', v'`` and ``L = min(||u||, ||v||, ||u'||, ||v'||)``,

    |cos(u', v') - cos(u, v)| <= (1/L) (||u' - u|| + ||v' - v||),   i.e. C = 1/L.

*Proof.* Writing ``u_hat = u / ||u||``:

1. ``|<u_hat, v_hat> - <u_hat', v_hat'>| <= |<u_hat - u_hat', v_hat>| +
   |<u_hat', v_hat - v_hat'>| <= ||u_hat - u_hat'|| + ||v_hat - v_hat'||``, by
   the triangle inequality and Cauchy-Schwarz with unit vectors.
2. In an inner-product space the Dunkl-Williams inequality holds with the sharp
   Hilbert-space constant 2:
   ``||u/||u|| - u'/||u'|| || <= 2 ||u - u'|| / (||u|| + ||u'||)``.
3. ``||u|| + ||u'|| >= 2L``, so ``||u_hat - u_hat'|| <= ||u - u'|| / L``; likewise
   for ``v``. []

A tighter non-uniform form is also provided, and a bound depending only on the
corpus, with no reference to the perturbation:

    C <= sqrt(max nnz)

which follows because ``idf >= 1`` and ``||tf||_1 = 1`` force
``||w_i|| >= 1/sqrt(nnz_i)``. That turns section 6's remark that "cosine
similarity becomes unstable for low-norm vectors" into a quantitative statement
and localises the instability to short documents.

Every bound here is checked by an adversarial property test that searches for a
violation rather than confirming a few examples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tfidf_stability.utils.numerics import Reduction, reduce_sum, sqrt
from tfidf_stability.vectorisation.sparse import SparseVector, l2_norm

__all__ = [
    "LipschitzBound",
    "corpus_lipschitz_bound",
    "difference_norm",
    "lipschitz_constant",
    "norm_lower_bound",
    "three_term_bound",
    "unit",
]


def unit(v: SparseVector, policy: Reduction = Reduction.NAIVE) -> SparseVector:
    """Normalise to unit length; the zero vector is returned unchanged."""
    n = l2_norm(v, policy)
    if n == 0.0:
        return v
    return SparseVector(indices=v.indices, values=tuple(x / n for x in v.values), dim=v.dim)


def difference_norm(a: SparseVector, b: SparseVector, policy: Reduction = Reduction.NAIVE) -> float:
    """``||a - b||_2``, computed over the union of the two supports.

    Iterating the union in ascending index order keeps the summation order
    canonical, so this agrees bit-for-bit with the native implementation.
    """
    if a.dim != b.dim:
        raise ValueError(f"dimension mismatch: {a.dim} vs {b.dim}")
    da = dict(zip(a.indices, a.values, strict=True))
    db = dict(zip(b.indices, b.values, strict=True))
    squares = []
    for i in sorted(da.keys() | db.keys()):
        d = da.get(i, 0.0) - db.get(i, 0.0)
        squares.append(d * d)
    return sqrt(reduce_sum(squares, policy))


@dataclass(frozen=True, slots=True)
class LipschitzBound:
    """The section 4.3 bound, with the pieces that produced it."""

    #: The uniform constant ``C = 1 / L``.
    constant: float
    #: ``L = min`` of the four norms.
    min_norm: float
    #: ``C * (||du|| + ||dv||)``: the bound as section 4.3 writes it.
    uniform: float
    #: The tighter per-vector form.
    tight: float
    #: The quantity being bounded, for reporting tightness.
    observed: float

    @property
    def holds(self) -> bool:
        """Whether both bounds are respected, allowing for the rounding incurred
        while evaluating the bound itself."""
        slack = 1e-12 * max(1.0, self.uniform) + 1e-15
        return self.observed <= self.uniform + slack and self.observed <= self.tight + slack

    @property
    def tightness(self) -> float:
        """``observed / tight``: how close the bound comes to being attained."""
        return self.observed / self.tight if self.tight > 0.0 else 0.0


def lipschitz_constant(
    u: SparseVector,
    v: SparseVector,
    u_prime: SparseVector,
    v_prime: SparseVector,
    policy: Reduction = Reduction.NAIVE,
) -> LipschitzBound:
    """Evaluate the section 4.3 bound for a concrete perturbation.

    Returns a :class:`LipschitzBound` carrying the uniform constant, both bound
    values and the observed change, so a caller can assert the inequality and
    report how tight it was.

    Raises:
        ValueError: If any of the four vectors is zero, where the bound is
            vacuous: cosine is defined there by convention rather than geometry.
    """
    nu, nv = l2_norm(u, policy), l2_norm(v, policy)
    nup, nvp = l2_norm(u_prime, policy), l2_norm(v_prime, policy)
    if min(nu, nv, nup, nvp) <= 0.0:
        raise ValueError("the Lipschitz bound requires four non-zero vectors")

    from tfidf_stability.similarity.cosine import cosine

    observed = abs(
        cosine(u_prime, v_prime, policy, u_norm=nup, v_norm=nvp)
        - cosine(u, v, policy, u_norm=nu, v_norm=nv)
    )
    du = difference_norm(u, u_prime, policy)
    dv = difference_norm(v, v_prime, policy)

    L = min(nu, nv, nup, nvp)
    C = 1.0 / L
    return LipschitzBound(
        constant=C,
        min_norm=L,
        uniform=C * (du + dv),
        # Dunkl-Williams applied per vector, without collapsing to a single L.
        tight=2.0 * du / (nu + nup) + 2.0 * dv / (nv + nvp),
        observed=observed,
    )


def norm_lower_bound(nnz: int) -> float:
    """``||w_i||_2 >= 1 / sqrt(nnz_i)`` for a TF-IDF vector.

    ``||tf_i||_1 = 1`` exactly, so Cauchy-Schwarz gives
    ``||tf_i||_2 >= 1/sqrt(nnz_i)``, and ``idf >= 1`` termwise can only increase
    the norm. Returns ``0.0`` for an empty support, where the vector is zero.
    """
    return 0.0 if nnz <= 0 else 1.0 / math.sqrt(nnz)


def corpus_lipschitz_bound(nnz_values: Sequence[int]) -> float:
    """``C <= sqrt(max nnz)``: a Lipschitz constant computable from the corpus alone.

    Needs no perturbation, so it serves as an a priori conditioning number for
    the whole pipeline. It grows with the largest support and ``C = 1/L`` blows
    up as norms shrink, so it identifies short documents as the source of section
    6's instability.
    """
    m = max((n for n in nnz_values if n > 0), default=0)
    return math.sqrt(m) if m > 0 else math.inf


def three_term_bound(
    tf: SparseVector,
    delta_tf: SparseVector,
    idf_linf: float,
    delta_idf_linf: float,
    policy: Reduction = Reduction.NAIVE,
) -> float:
    """The section 4.2 decomposition:

        ||w' - w|| <= ||dtf|| ||idf||_inf + ||tf|| ||didf||_inf
                      + ||dtf|| ||didf||_inf

    separating a local document edit, a global corpus change, and their
    interaction.

    Presupposes a common index set. A corpus perturbation changes the vocabulary
    and the paper does not say how ``didf`` is defined then; this project
    resolves it on the union vocabulary. See ``docs/spec_addenda.md#g5``.
    """
    n_dtf = l2_norm(delta_tf, policy)
    n_tf = l2_norm(tf, policy)
    return n_dtf * idf_linf + n_tf * delta_idf_linf + n_dtf * delta_idf_linf
