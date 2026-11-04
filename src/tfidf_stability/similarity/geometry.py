"""Geometric quantities and the perturbation bounds of README section 4.

The section 4.3 constant is ``C = 1/L``, ``L = min(||u||, ||v||, ||u'||, ||v'||)``,
by Dunkl-Williams; ``idf >= 1`` and ``||tf||_1 = 1`` give ``C <= sqrt(max nnz)``.
See ``docs/spec_addenda.md#g4``. Rationale: docs/engineering_notes.md#cosine-lipschitz-constant
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
        """Whether both bounds hold, with slack for rounding in evaluating them."""
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

    Returns a :class:`LipschitzBound` with the uniform constant, both bounds and
    the observed change. Raises ``ValueError`` if any norm is not strictly positive
    (NaN included): cosine at zero is a convention, and a NaN minimum gives ``C = NaN``.
    """
    nu, nv = l2_norm(u, policy), l2_norm(v, policy)
    nup, nvp = l2_norm(u_prime, policy), l2_norm(v_prime, policy)
    # Each norm is tested for positivity rather than `min(...) <= 0.0`: every comparison
    # with NaN is false, so a NaN norm passes that test, and `min` returns whichever
    # operand it saw first, making the guard depend on argument order. The `tau` guards
    # in ranking/tie_groups.py and analysis/stratify.py negate for the same reason.
    if not all(norm > 0.0 for norm in (nu, nv, nup, nvp)):
        raise ValueError(
            "the Lipschitz bound requires four non-zero vectors with a defined norm, "
            f"got {(nu, nv, nup, nvp)}"
        )

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

    It needs no perturbation, so it is an a priori conditioning number for the
    pipeline. It grows with the largest support, so the instability section 6
    describes is worst for documents with many distinct terms, not short ones.
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
    """Bound ``||w' - w||`` by the section 4.2 three-term decomposition.

    ``||dtf|| ||idf||_inf + ||tf|| ||didf||_inf + ||dtf|| ||didf||_inf``: a document
    edit, a corpus change and their interaction. It presupposes a common index set;
    ``didf`` is taken on the union vocabulary. See ``docs/spec_addenda.md#g5``.
    """
    n_dtf = l2_norm(delta_tf, policy)
    n_tf = l2_norm(tf, policy)
    return n_dtf * idf_linf + n_tf * delta_idf_linf + n_dtf * delta_idf_linf
