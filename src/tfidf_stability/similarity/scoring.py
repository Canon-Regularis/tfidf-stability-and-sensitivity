"""Query scoring over an inverted index: ``s_i = cos(q, w_i)`` (README section 2.3).

This module is the normative reference for the two scoring kernels of
``cpp/include/tfidf/similarity/scoring.hpp``. Until it existed those kernels had
no Python counterpart to be measured against -- only
:func:`~tfidf_stability.similarity.cosine.cosine_against_corpus`, which scores a
dense list of document vectors and shares neither loop nesting nor data
structure with either of them.

Three implementations, one number
---------------------------------

**TAAT** (term at a time) walks the postings list of each query term into a
dense accumulator. **DAAT** (document at a time) merges each document's row
against the query independently and never builds an inverted index at all.
``cosine_against_corpus`` does the same merge over a materialised sequence of
rows. All three must produce *identical bit patterns*, not merely values that
agree to within rounding, and ``tests/test_scoring_kernels.py`` asserts exactly
that via :func:`~tfidf_stability.utils.numerics.same_bits`.

*Why they agree.* The TAAT outer loop runs over query terms in **ascending term
identifier**, and a canonical query stores each term once, so a given term
contributes at most one addition to any one accumulator. The addend sequence
seen by document ``d`` is therefore

    ascending term id over supp(q) INTERSECT supp(w_d), starting from 0.0

which is precisely the sequence the merge in
:func:`~tfidf_stability.vectorisation.sparse.dot` produces. Every reduction
policy in :class:`~tfidf_stability.utils.numerics.Reduction` is a pure function
of that *ordered* sequence, so equal sequences give equal bits.

The property depends on the ascent. Blocking the term loop, or visiting postings
in any order other than the one :meth:`InvertedIndex.from_csr` lays down, would
reassociate the sum and change the digits -- which is the whole subject of this
project rather than an incidental detail. No such reordering is applied here,
and none may be applied on the normative path.

*Where this deviates in form from the C++.* The native kernels use incremental
accumulator objects (one ``add`` call per product); Python's reducers are batch
functions over a sequence. The non-naive TAAT branch therefore buffers each
document's products and reduces once at the end. That is a difference in
plumbing only: each policy consumes its addends strictly in order and its result
depends on nothing else, so buffering cannot change a bit.

Cost
----

TAAT costs ``O(sum of df(t) over query terms)`` multiply-adds plus ``O(|touched|)``
divisions -- not ``O(N)`` divisions, since an untouched document scores exactly
0. DAAT costs ``O(sum of nnz(d) over all documents)``. TAAT wins when the query's
terms are individually rare, which is the usual case for the TF-IDF profile
queries of section 7.1.
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

    The two are required to agree bit for bit, so the choice is a performance
    decision only. It is still explicit and recorded in run manifests, because a
    claim of agreement is worth nothing unless which one ran is knowable.
    """

    #: Term at a time over the inverted index. The default.
    TAAT = "taat"
    #: Document at a time; an independent merge per document.
    DAAT = "daat"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class InvertedIndex:
    """Compressed sparse column form of the corpus matrix -- the postings lists.

    Column ``t`` holds the documents containing term ``t``, in ascending document
    order, alongside their weights. Mirrors ``tfidf::Csc``.

    Attributes:
        colptr: Column boundaries, length ``n_cols + 1``.
        rowidx: Document indices, ascending within each column.
        values: Weights, parallel to ``rowidx``.
        n_rows: Document count.
        n_cols: Vocabulary size.
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

        Because the source rows are visited in ascending document order and each
        column's entries are appended in that order, every postings list comes
        out ascending in document identifier for free. No sort runs, so no
        sort's tie-breaking can influence the layout -- and the layout fixes the
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

    Mirrors ``tfidf::ScoringScratch``. It exists so that repeated scoring does no
    per-query allocation, and it is part of the reference rather than an
    optimisation detail because *reuse must be unobservable*: a stale accumulator
    entry would silently contaminate the next query's scores, and the only way to
    show it does not is to reproduce the reuse here and compare.

    Attributes:
        accumulator: Dense, one slot per document.
        touched: The documents that are live for the current query.
    """

    accumulator: list[float] = field(default_factory=list)
    touched: list[int] = field(default_factory=list)

    def reset(self, n_docs: int) -> None:
        self.accumulator = [0.0] * n_docs
        self.touched = []

    def clear_touched(self) -> None:
        """Zero only the entries the previous query touched.

        ``O(|touched|)`` rather than ``O(n_docs)``, which is what makes scoring a
        sparse query against a large corpus cost nothing per untouched document.
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

    Split out of :func:`taat_scores` only so the two accumulation strategies sit
    side by side and can be read against each other; the normalisation that
    follows is common to both.
    """
    if policy is Reduction.NAIVE:
        # The normative fold, and the only policy whose entire state is the
        # running sum -- so it can live directly in the dense array.
        #
        # ASCENDING term id. This is what makes the result bit-identical to the
        # merge-based dot product. Do not reorder.
        for t, qv in zip(query.indices, query.values, strict=True):
            for p in range(index.colptr[t], index.colptr[t + 1]):
                d = index.rowidx[p]
                # A slot can in principle accumulate back to 0.0 and be recorded
                # twice. It cannot here -- TF-IDF weights are non-negative, so a
                # sum of products of stored values is zero only if some factor is
                # zero, which a canonical sparse structure never stores -- and a
                # duplicate would merely rewrite the same quotient anyway.
                if scratch.accumulator[d] == 0.0:
                    scratch.touched.append(d)
                scratch.accumulator[d] += qv * index.values[p]
        return

    # A compensated or tree-shaped reduction carries per-document state, so each
    # touched document needs its own addend sequence.
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

    Args:
        query: The query vector, canonical (strictly ascending indices).
        index: The inverted index, from :meth:`InvertedIndex.from_csr`.
        doc_norms: Precomputed ``||w_i||_2``, one per document, under ``policy``.
        policy: Summation policy for the accumulated dot products and, when it is
            not supplied, the query norm.
        query_norm: Precomputed ``||q||_2``. Supplying it is a performance choice
            and changes nothing numerically, provided the same policy produced it.
        scratch: Working state to reuse across queries. A fresh one is allocated
            when omitted; passing the same object repeatedly is numerically
            indistinguishable from allocating each time.

    Returns:
        ``s_i = cos(q, w_i)`` for every document, with the zero-vector convention
        of section 2.3 applied.
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
        # The expression is dot / (qn * dn). Not (dot / qn) / dn, and not
        # dot * (1 / (qn * dn)); those round differently, and cosine.py pins
        # this exact form.
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

    An independent merge per document. It builds no inverted index and uses no
    dense accumulator, which is exactly why agreeing with :func:`taat_scores` to
    the last bit says something: the two share no data structure and no loop
    nesting, so a common indexing or accumulation bug has almost nowhere to hide.

    Arguments and return value match :func:`taat_scores`, with the corpus given
    in CSR rather than inverted form.
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
    algorithm: ScoringAlgorithm = ScoringAlgorithm.TAAT,
    *,
    query_norm: float | None = None,
    scratch: ScoringScratch | None = None,
) -> list[float]:
    """Score by the requested traversal. Mirrors ``tfidf::score``."""
    if algorithm is ScoringAlgorithm.DAAT:
        return daat_scores(query, corpus, doc_norms, policy, query_norm=query_norm)
    return taat_scores(query, index, doc_norms, policy, query_norm=query_norm, scratch=scratch)
