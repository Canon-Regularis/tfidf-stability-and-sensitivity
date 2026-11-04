"""Inverted-index query scoring, ``s_i = cos(q, w_i)`` (README section 2.3).

Normative reference for ``cpp/include/tfidf/similarity/scoring.hpp``. TAAT, DAAT and
:func:`~tfidf_stability.similarity.cosine.cosine_against_corpus` give identical bits: each adds
products in ascending term id, TAAT in :meth:`InvertedIndex.from_csr` order. Do not reorder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

from tfidf_stability.utils.numerics import Reduction, reduce_sum
from tfidf_stability.vectorisation.sparse import CsrMatrix, SparseVector, dot, l2_norm

__all__ = [
    "InvertedIndex",
    "ScoringAlgorithm",
    "ScoringScratch",
    "daat_scores",
    "score",
    "taat_scores",
]


class ScoringAlgorithm(str, Enum):
    """Which traversal to score with.

    The traversals agree bit for bit (:func:`~tfidf_stability.utils.numerics.same_bits` in
    ``tests/test_scoring_kernels.py``), so the choice affects speed only. TAAT costs
    ``O(sum of df(t) over query terms)`` and suits the rare-term profile queries of section 7.1.
    """

    #: Term at a time over the inverted index. The default.
    TAAT = "taat"
    #: Document at a time; an independent merge per document.
    DAAT = "daat"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class InvertedIndex:
    """Compressed sparse column form of the corpus matrix: the postings lists.

    Column ``t`` lists, in ascending document order, the documents containing term ``t`` and their
    weights. Fields: ``colptr`` (length ``n_cols + 1``), ``rowidx`` and ``values`` (parallel),
    ``n_rows`` (documents) and ``n_cols`` (vocabulary size). Mirrors ``tfidf::Csc``.
    """

    colptr: tuple[int, ...]
    rowidx: tuple[int, ...]
    values: tuple[float, ...]
    n_rows: int
    n_cols: int

    @property
    def nnz(self) -> int:
        return len(self.values)

    def postings_begin(self, t: int) -> int:
        return self.colptr[t]

    def postings_end(self, t: int) -> int:
        return self.colptr[t + 1]

    def df(self, t: int) -> int:
        """Document frequency of term ``t``: the length of its postings list."""
        return self.colptr[t + 1] - self.colptr[t]

    def is_canonical(self) -> bool:
        """Whether every postings list is strictly ascending and in range.

        The scoring loops rely on this rather than re-establishing it, so it is
        asserted in the tests instead of being paid for on every query.
        """
        well_formed = (
            len(self.colptr) == self.n_cols + 1
            and self.colptr[0] == 0
            and self.colptr[-1] == len(self.values)
            and len(self.rowidx) == len(self.values)
        )
        if not well_formed:
            return False
        for t in range(self.n_cols):
            lo, hi = self.colptr[t], self.colptr[t + 1]
            seg = self.rowidx[lo:hi]
            if lo > hi or any(a >= b for a, b in pairwise(seg)):
                return False
            if any(not 0 <= d < self.n_rows for d in seg):
                return False
        return True

    @classmethod
    def from_csr(cls, matrix: CsrMatrix) -> InvertedIndex:
        """Transpose a CSR corpus matrix by counting sort: ``O(nnz + n_cols)``.

        Rows are visited in ascending document order, so each postings list is ascending
        without a sort and no sort's tie-breaking affects the layout. The layout fixes the
        accumulation order, which fixes the digits.
        """
        colptr = [0] * (matrix.n_cols + 1)
        for t in matrix.indices:
            colptr[t + 1] += 1
        for c in range(matrix.n_cols):
            colptr[c + 1] += colptr[c]

        rowidx = [0] * matrix.nnz
        values = [0.0] * matrix.nnz
        cursor = colptr[:-1]
        for d in range(matrix.n_rows):
            for k in range(matrix.indptr[d], matrix.indptr[d + 1]):
                t = matrix.indices[k]
                pos = cursor[t]
                cursor[t] = pos + 1
                rowidx[pos] = d
                values[pos] = matrix.values[k]

        return cls(
            colptr=tuple(colptr),
            rowidx=tuple(rowidx),
            values=tuple(values),
            n_rows=matrix.n_rows,
            n_cols=matrix.n_cols,
        )


@dataclass(slots=True)
class ScoringScratch:
    """Reusable working state for TAAT, allocated once and reused per query.

    Mirrors ``tfidf::ScoringScratch``. Reuse must be unobservable, since a stale accumulator
    entry would contaminate the next query, so the reference reproduces the reuse for comparison.
    Fields: ``accumulator`` (dense, one slot per document) and ``touched`` (live documents).
    """

    accumulator: list[float] = field(default_factory=list)
    touched: list[int] = field(default_factory=list)

    def reset(self, n_docs: int) -> None:
        self.accumulator = [0.0] * n_docs
        self.touched = []

    def clear_touched(self) -> None:
        """Zero only the entries the previous query touched.

        ``O(|touched|)`` rather than ``O(n_docs)``, so scoring a sparse query
        against a large corpus costs nothing per untouched document.
        """
        for d in self.touched:
            self.accumulator[d] = 0.0
        self.touched = []


def _accumulate(
    query: SparseVector,
    index: InvertedIndex,
    policy: Reduction,
    scratch: ScoringScratch,
) -> None:
    """Fill ``scratch`` with the unnormalised dot products of the touched documents.

    Called by :func:`taat_scores`, which normalises the result. Every
    :class:`~tfidf_stability.utils.numerics.Reduction` policy is a pure function of the ordered
    addends, so equal addend sequences give equal bits.
    """
    if policy is Reduction.NAIVE:
        # The normative fold; its whole state is the running sum, so it lives in the dense array.
        # ASCENDING term id makes the result bit-identical to the merge-based dot product.
        # Do not reorder. Rationale: docs/engineering_notes.md#scoring-term-order
        for t, qv in zip(query.indices, query.values, strict=True):
            for p in range(index.colptr[t], index.colptr[t + 1]):
                d = index.rowidx[p]
                # A slot still 0.0 after an addition is recorded again. TF-IDF weights are
                # non-negative and a canonical structure stores no zeros, so that happens only if
                # every product so far underflowed; the duplicate rewrites the same quotient.
                if scratch.accumulator[d] == 0.0:
                    scratch.touched.append(d)
                scratch.accumulator[d] += qv * index.values[p]
        return

    # A compensated or tree-shaped reduction carries per-document state, so each touched
    # document gets its own addend sequence, reduced once at the end. The C++ accumulates
    # incrementally; each policy consumes addends strictly in order, so buffering changes no bit.
    products: list[list[float]] = [[] for _ in range(index.n_rows)]
    seen = bytearray(index.n_rows)
    for t, qv in zip(query.indices, query.values, strict=True):
        for p in range(index.colptr[t], index.colptr[t + 1]):
            d = index.rowidx[p]
            if not seen[d]:
                seen[d] = 1
                scratch.touched.append(d)
            products[d].append(qv * index.values[p])
    for d in scratch.touched:
        scratch.accumulator[d] = reduce_sum(products[d], policy)


def taat_scores(
    query: SparseVector,
    index: InvertedIndex,
    doc_norms: Sequence[float],
    policy: Reduction = Reduction.NAIVE,
    *,
    query_norm: float | None = None,
    scratch: ScoringScratch | None = None,
) -> list[float]:
    """Score a query against every document, term at a time.

    ``query`` is canonical; ``index`` is from :meth:`InvertedIndex.from_csr`; reusing ``scratch``
    changes no bit. ``policy`` sums the products and an omitted ``query_norm``; ``doc_norms`` and a
    supplied ``query_norm`` must come from it too. Returns ``s_i``; zero vectors per section 2.3.
    """
    if query.dim != index.n_cols:
        raise ValueError(f"dimension mismatch: query {query.dim} vs index {index.n_cols}")
    if len(doc_norms) != index.n_rows:
        raise ValueError(f"{index.n_rows} documents but {len(doc_norms)} norms were supplied")

    n_docs = index.n_rows
    out = [0.0] * n_docs
    qn = l2_norm(query, policy) if query_norm is None else query_norm
    if qn == 0.0 or query.nnz == 0:
        return out  # a zero query scores 0 everywhere (spec_addenda G3)

    if scratch is None:
        scratch = ScoringScratch()
    if len(scratch.accumulator) != n_docs:
        scratch.reset(n_docs)
    else:
        scratch.clear_touched()

    _accumulate(query, index, policy, scratch)

    for d in scratch.touched:
        dn = doc_norms[d]
        # dot / (qn * dn). The groupings (dot / qn) / dn and dot * (1/(qn * dn))
        # round differently; cosine.py pins this one.
        out[d] = 0.0 if dn == 0.0 else scratch.accumulator[d] / (qn * dn)
    return out


def daat_scores(
    query: SparseVector,
    corpus: CsrMatrix,
    doc_norms: Sequence[float],
    policy: Reduction = Reduction.NAIVE,
    *,
    query_norm: float | None = None,
) -> list[float]:
    """Score a query against every document, document at a time.

    Merges each row with :func:`~tfidf_stability.vectorisation.sparse.dot`; it shares no data
    structure or loop nesting with :func:`taat_scores`, so bit-for-bit agreement checks both.
    Arguments and return value match :func:`taat_scores`, with the corpus in CSR form.
    """
    if query.dim != corpus.n_cols:
        raise ValueError(f"dimension mismatch: query {query.dim} vs corpus {corpus.n_cols}")
    if len(doc_norms) != corpus.n_rows:
        raise ValueError(f"{corpus.n_rows} documents but {len(doc_norms)} norms were supplied")

    qn = l2_norm(query, policy) if query_norm is None else query_norm
    if qn == 0.0 or query.nnz == 0:
        return [0.0] * corpus.n_rows
    return [
        0.0 if doc_norms[d] == 0.0 else dot(query, corpus.row(d), policy) / (qn * doc_norms[d])
        for d in range(corpus.n_rows)
    ]


def score(
    query: SparseVector,
    corpus: CsrMatrix,
    index: InvertedIndex,
    doc_norms: Sequence[float],
    policy: Reduction = Reduction.NAIVE,
    *,
    # Keyword-only: the two traversals must agree to the last bit, so which one
    # ran is a performance choice and never a numerical one. Naming it at the
    # call site keeps that visible, and stops `policy` and `algorithm` (adjacent
    # enums that both default) from being swapped positionally.
    algorithm: ScoringAlgorithm = ScoringAlgorithm.TAAT,
    query_norm: float | None = None,
    scratch: ScoringScratch | None = None,
) -> list[float]:
    """Score by the requested traversal, mirroring ``tfidf::score``.

    Anything that fails ``is ScoringAlgorithm.DAAT``, including the string ``"daat"``, runs TAAT,
    as in the C++. The traversals agree to the last bit, so a wrong branch costs only time;
    raising would make the two implementations disagree about an input neither is asked for.
    """
    if algorithm is ScoringAlgorithm.DAAT:
        return daat_scores(query, corpus, doc_norms, policy, query_norm=query_norm)
    return taat_scores(query, index, doc_norms, policy, query_norm=query_norm, scratch=scratch)
