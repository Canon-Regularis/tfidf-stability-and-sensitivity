"""Cosine similarity (README section 2.3).

    cos(u, v) = (u . v) / (||u||_2 ||v||_2),  and 0 if either vector is zero.

TF-IDF coordinates are non-negative, so ``cos(u, v)`` lies in ``[0, 1]``. A
negative coordinate anywhere upstream would void that silently, so
:func:`~tfidf_stability.utils.validation.check_non_negative` exists to catch it.

The zero-vector convention drives the study. A document with no in-vocabulary
tokens is common in short-text corpora (around 17% of documents on a
MovieLens-shaped corpus) and scores exactly 0 against every query, so they form a
large block of exact ties at the bottom of every ranking, which is the regime
section 4.5's tie-break analysis is about.
"""

from __future__ import annotations

from collections.abc import Sequence

from tfidf_stability.utils.numerics import Reduction
from tfidf_stability.vectorisation.sparse import SparseVector, dot, l2_norm

__all__ = ["cosine", "cosine_against_corpus", "cosine_matrix"]


def cosine(
    u: SparseVector,
    v: SparseVector,
    policy: Reduction = Reduction.NAIVE,
    *,
    u_norm: float | None = None,
    v_norm: float | None = None,
) -> float:
    """Cosine similarity of two sparse non-negative vectors.

    Args:
        u, v: The vectors. Must share an ambient dimension.
        policy: Summation policy for the dot product and, if they are not
            supplied, the norms.
        u_norm, v_norm: Precomputed norms. Supplying them is purely a
            performance choice and changes nothing numerically, provided they
            were computed under the same policy.

    Returns:
        The similarity, in ``[0, 1]`` for non-negative inputs, or ``0.0`` if
        either vector is zero.

    Note:
        The expression is ``dot / (nu * nv)``. The groupings ``(dot / nu) / nv``
        and ``dot * (1 / (nu * nv))`` are algebraically equal and numerically
        different, and the native backend reproduces this one bit for bit.
    """
    if u.dim != v.dim:
        raise ValueError(f"dimension mismatch: {u.dim} vs {v.dim}")

    nu = l2_norm(u, policy) if u_norm is None else u_norm
    nv = l2_norm(v, policy) if v_norm is None else v_norm
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot(u, v, policy) / (nu * nv)


def cosine_against_corpus(
    query: SparseVector,
    documents: Sequence[SparseVector],
    doc_norms: Sequence[float],
    policy: Reduction = Reduction.NAIVE,
) -> list[float]:
    """Score one query against every document: ``s_i = cos(q, w_i)``.

    The query norm is computed once and the document norms are supplied
    precomputed, so the dot products dominate the cost.

    A zero query vector yields all-zero scores: degenerate but legitimate, since
    the ranking then depends entirely on the tie-break attributes. Flagged rather
    than excluded; see ``docs/spec_addenda.md#g3``.
    """
    if len(documents) != len(doc_norms):
        raise ValueError(f"{len(documents)} documents but {len(doc_norms)} norms were supplied")
    q_norm = l2_norm(query, policy)
    if q_norm == 0.0:
        return [0.0] * len(documents)
    return [
        0.0 if dn == 0.0 else dot(query, d, policy) / (q_norm * dn)
        for d, dn in zip(documents, doc_norms, strict=True)
    ]


def cosine_matrix(
    vectors: Sequence[SparseVector],
    policy: Reduction = Reduction.NAIVE,
    norms: Sequence[float] | None = None,
) -> list[list[float]]:
    """Full pairwise similarity matrix.

    Only the upper triangle is computed and the result mirrored, so
    ``S[i][j] == S[j][i]`` holds bit for bit rather than to within rounding.
    """
    n = len(vectors)
    ns = [l2_norm(v, policy) for v in vectors] if norms is None else list(norms)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = cosine(vectors[i], vectors[j], policy, u_norm=ns[i], v_norm=ns[j])
            out[i][j] = s
            out[j][i] = s
    return out
