"""The deterministic ranking operator (README section 2.3.1).

Produces a :class:`Ranking`: a total order over documents, together with the
sorted score array that margins and tie groups are computed from.

The sort depends on neither ``k`` nor ``tau``: a ranking is a function of
``(scores, table, priority)`` alone. Section 7's grid is therefore
"queries x 3 operators" rather than "queries x k-values x tau-values", with every
``k`` and ``tau`` read off the result afterwards, worth roughly three orders of
magnitude. So :func:`rank` takes no ``k``, and the margin and tie-group modules
take a sorted score array rather than a ``Ranking``.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.sort_keys import (
    OPERATORS,
    PI,
    SortKeySpec,
    build_keys,
)
from tfidf_stability.utils.validation import (
    EmptyCorpusError,
    StrictMode,
    check_finite,
    resolve_k,
)

__all__ = [
    "Ranking",
    "Selection",
    "rank",
    "rank_all_operators",
    "rank_top_k",
    "sorted_scores_desc",
]


class Selection(str, Enum):
    """Which selection algorithm produced an order.

    All of these must yield the identical permutation, since the comparator is a
    strict total order. The agreement test over them is the operational content
    of "sort stability is irrelevant here", and the ranking analogue of the
    ``TAAT == DAAT`` check in the scoring layer.
    """

    FULL_SORT = "full_sort"
    STABLE_SORT = "stable_sort"
    HEAP_ALL = "heap_all"
    HEAP_TOP_K = "heap_top_k"
    INSERTION = "insertion"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Ranking:
    """A total order over documents, plus everything derived from the scores.

    ``order`` may be truncated by a top-k selection; ``sorted_scores`` never is.
    Truncating the document order is what makes partial selection worth doing,
    and keeping the full score array keeps margins and tie groups answerable at
    every ``k`` and ``tau`` without re-ranking.

    Attributes:
        order: Document indices, best first. Length ``n_selected``.
        sorted_scores: All scores, non-increasing. Always length ``n_documents``.
        scores: The raw score vector, index-aligned to documents.
        operator: The operator's name.
        key_digest: Identity of (operator, attribute table), for the manifest.
        query_degenerate: Every score is exactly zero, so the attributes decide
            the order alone. Defined observationally, since "the query vector was
            zero" is not checkable from the ranker's inputs.
        n_zero_norm_docs: Passed in rather than derived; this module has no
            dependency on ``vectorisation``.
        k_effective: The clamped ``k`` when lenient mode clamped one; ``None``
            for a full ranking.
    """

    order: tuple[int, ...]
    sorted_scores: tuple[float, ...]
    scores: tuple[float, ...]
    operator: str
    key_digest: str
    n_documents: int
    n_selected: int
    query_degenerate: bool
    n_zero_norm_docs: int
    n_excluded: int
    strict_mode: StrictMode
    k_effective: int | None = None

    @property
    def is_complete(self) -> bool:
        """Whether every document was selected."""
        return self.n_selected == self.n_documents

    def require_complete(self, what: str) -> None:
        """Raise if a truncated ranking is asked a whole-corpus question."""
        if not self.is_complete:
            raise ValueError(
                f"{what} needs the complete ranking, but only {self.n_selected} of "
                f"{self.n_documents} documents were selected. Use rank() rather than "
                f"rank_top_k()."
            )

    def top_k(self, k: int) -> tuple[int, ...]:
        """The first ``k`` document indices."""
        if k > self.n_selected:
            raise ValueError(f"top_k({k}) but only {self.n_selected} documents were selected")
        return self.order[:k]

    def score_at_rank(self, j: int) -> float:
        """``score(r_j)`` for the paper's 1-indexed rank ``j``.

        Ranks are 1-indexed in the public API to match ``r_1 ... r_n`` and
        0-indexed in the arrays beneath it.
        """
        if not 1 <= j <= self.n_documents:
            raise IndexError(f"rank {j} out of range 1..{self.n_documents}")
        return self.sorted_scores[j - 1]

    def rank_of(self, doc: int) -> int:
        """The 1-indexed rank of a document, which must have been selected."""
        try:
            return self.order.index(doc) + 1
        except ValueError:
            raise KeyError(
                f"document {doc} is not in the selected order"
                + ("" if self.is_complete else " (the ranking is truncated)")
            ) from None

    def order_within(self, documents: Sequence[int]) -> tuple[int, ...]:
        """``documents`` restricted to this ranking's order.

        Used by the stage 5 ordering distances, which compare two operators over
        a tie group or a top-k set.
        """
        wanted = set(documents)
        return tuple(d for d in self.order if d in wanted)


def sorted_scores_desc(scores: Sequence[float]) -> tuple[float, ...]:
    """Scores in non-increasing order.

    Sorted as raw doubles, independently of the ranking. All three operators and
    every ``k`` and ``tau`` share this array, and it is what makes margins
    provably tie-break independent.
    """
    return tuple(sorted(scores, reverse=True))


def _select(keys: Sequence[tuple[float, ...]], m: int, how: Selection) -> tuple[int, ...]:
    """Return the indices of the ``m`` smallest keys, in ascending key order."""
    n = len(keys)
    idx = range(n)
    if how is Selection.FULL_SORT:
        return tuple(sorted(idx, key=keys.__getitem__))[:m]
    if how is Selection.STABLE_SORT:
        # Python's sort is always stable; enumerated separately so the agreement
        # test can show stability cannot matter here.
        return tuple(sorted(idx, key=lambda i: keys[i]))[:m]
    if how is Selection.HEAP_ALL:
        return tuple(heapq.nsmallest(n, idx, key=keys.__getitem__))[:m]
    if how is Selection.HEAP_TOP_K:
        return tuple(heapq.nsmallest(m, idx, key=keys.__getitem__))
    # Insertion sort: naive, as an independent implementation.
    out: list[int] = []
    for i in idx:
        lo, hi = 0, len(out)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[out[mid]] < keys[i]:
                lo = mid + 1
            else:
                hi = mid
        out.insert(lo, i)
    return tuple(out[:m])


def rank(
    scores: Sequence[float],
    table: AttributeTable,
    spec: SortKeySpec = PI,
    *,
    mode: StrictMode = StrictMode.STRICT,
    n_zero_norm_docs: int = 0,
    selection: Selection = Selection.FULL_SORT,
    _sorted_scores: tuple[float, ...] | None = None,
) -> Ranking:
    """Rank every document under one operator.

    Args:
        scores: Similarity scores, index-aligned to the attribute table.
        table: The rank-encoded attribute table.
        spec: Which operator (:data:`~tfidf_stability.ranking.sort_keys.PI` by default).
        mode: Strict or lenient handling of degenerate inputs.
        n_zero_norm_docs: Reported on the result; supplied by the caller.
        selection: Which algorithm to use. Every value must give the identical
            permutation; the parameter exists so tests can prove it.
        _sorted_scores: Internal. Lets :func:`rank_all_operators` share one
            array object across operators.

    Returns:
        The :class:`Ranking`.

    Raises:
        EmptyCorpusError: If there are no documents.
        TfidfStabilityError: If any score is NaN or infinite, in lenient mode
            too: ``spec_addenda.md#g3`` lists non-finite scores among the
            rejected inputs rather than the legitimate degenerate grid points,
            and a NaN in a sort key is undefined behaviour in the native backend
            rather than merely a wrong answer.
    """
    n = len(scores)
    if n == 0:
        raise EmptyCorpusError("cannot rank an empty corpus")
    # One O(N) pass guarding an O(N log N) sort: under 1% overhead, and the only
    # thing between a corrupt score vector and undefined behaviour.
    check_finite(scores, "scores")

    keys = build_keys(scores, table, spec)
    order = _select(keys, n, selection)

    return Ranking(
        order=order,
        sorted_scores=_sorted_scores if _sorted_scores is not None else sorted_scores_desc(scores),
        scores=tuple(scores),
        operator=spec.name,
        key_digest=spec.digest(table),
        n_documents=n,
        n_selected=n,
        query_degenerate=all(s == 0.0 for s in scores),
        n_zero_norm_docs=n_zero_norm_docs,
        n_excluded=0,
        strict_mode=mode,
    )


def rank_top_k(
    scores: Sequence[float],
    table: AttributeTable,
    spec: SortKeySpec = PI,
    *,
    k: int,
    mode: StrictMode = StrictMode.STRICT,
    n_zero_norm_docs: int = 0,
    selection: Selection = Selection.HEAP_TOP_K,
) -> Ranking:
    """Rank only the top ``min(k + 1, N)`` documents.

    ``k + 1`` because the boundary margin ``m_k = score(r_k) - score(r_{k+1})``
    needs the first document outside the top-k, and section 7.3 stratifies every
    disagreement rate by ``m_k``.

    ``sorted_scores`` is still complete, so margins and tie groups remain
    answerable.
    """
    n = len(scores)
    if n == 0:
        raise EmptyCorpusError("cannot rank an empty corpus")
    check_finite(scores, "scores")

    k_eff = resolve_k(k, n, mode)
    m = min(k_eff + 1, n)

    keys = build_keys(scores, table, spec)
    order = _select(keys, m, selection)

    return Ranking(
        order=order,
        sorted_scores=sorted_scores_desc(scores),
        scores=tuple(scores),
        operator=spec.name,
        key_digest=spec.digest(table),
        n_documents=n,
        n_selected=m,
        query_degenerate=all(s == 0.0 for s in scores),
        n_zero_norm_docs=n_zero_norm_docs,
        n_excluded=0,
        strict_mode=mode,
        k_effective=k_eff,
    )


def rank_all_operators(
    scores: Sequence[float],
    table: AttributeTable,
    specs: Sequence[SortKeySpec] = OPERATORS,
    **kwargs: object,
) -> dict[str, Ranking]:
    """Rank under several operators, sharing one ``sorted_scores`` object.

    Margins depend only on the sorted score multiset, so they are identical under
    every operator. Sharing the array object makes that structural rather than
    coincidental, and turns the corresponding test into a regression guard on the
    sharing itself.
    """
    shared = sorted_scores_desc(scores)
    return {
        spec.name: rank(scores, table, spec, _sorted_scores=shared, **kwargs)  # type: ignore[arg-type]
        for spec in specs
    }
