"""Cosine similarity of sparse vectors (README section 2.3).

cos(u, v) = (u . v) / (||u||_2 ||v||_2), and 0 if either vector is zero. Non-negative
input (:func:`~tfidf_stability.utils.validation.check_non_negative`) keeps it in ``[0, 1]``.
Zero vectors tie at 0 (section 4.5). Rationale: docs/engineering_notes.md#zero-vector-convention
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
    """Cosine similarity of two sparse non-negative vectors of equal dimension.

    Returns ``dot / (nu * nv)`` under ``policy``, in ``[0, 1]`` for non-negative input, or 0.0
    if either vector is zero. The native backend uses this grouping and matches it bit for bit.
    Precomputed ``u_norm`` and ``v_norm`` change nothing numerically if computed under ``policy``.
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

    The query norm is computed once and ``doc_norms`` are precomputed. A zero query yields
    all-zero scores, so the ranking depends only on the tie-break attributes; the ranker
    flags it as ``query_degenerate`` rather than excluding it (``docs/spec_addenda.md#g3``).
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
