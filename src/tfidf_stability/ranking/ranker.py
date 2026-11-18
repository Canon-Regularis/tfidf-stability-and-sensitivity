"""The deterministic ranking operator (README section 2.3.1).

Produces a :class:`Ranking`. :func:`rank` takes no ``k`` or ``tau``: a ranking is a
function of ``(scores, table, priority)`` alone, so section 7's grid is queries x 3 operators.
Rationale: docs/engineering_notes.md#ranking-independent-of-k-and-tau
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
    check_prefix_k,
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

    Every member yields the identical permutation, since the comparator is a strict total
    order, so sort stability is irrelevant; the agreement test mirrors ``TAAT == DAAT``.
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

    ``order`` holds ``n_selected`` document indices, best first, and may be truncated.
    ``sorted_scores`` holds all ``n_documents`` scores, non-increasing, so margins and tie
    groups stay answerable at every ``k`` and ``tau`` without re-ranking.
    """

    order: tuple[int, ...]
    sorted_scores: tuple[float, ...]
    #: The raw score vector, index-aligned to documents.
    scores: tuple[float, ...]
    operator: str
    #: Identity of (operator, attribute table), recorded in the manifest.
    key_digest: str
    n_documents: int
    n_selected: int
    #: Every score is exactly zero, so the attributes alone decide the order. This is
    #: observational: the query vector is not among the ranker's inputs.
    query_degenerate: bool
    #: Supplied by the caller; this module has no dependency on ``vectorisation``.
    n_zero_norm_docs: int
    n_excluded: int
    strict_mode: StrictMode
    #: The ``k`` used by :func:`rank_top_k`, clamped to ``n_documents`` in lenient mode;
    #: ``None`` for a full ranking.
    k_effective: int | None = None

    def __post_init__(self) -> None:
        """Normalise ``strict_mode`` to its member; ``resolve_k`` compares it."""
        object.__setattr__(self, "strict_mode", StrictMode(self.strict_mode))

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
        """The first ``k`` document indices.

        Both bounds are checked: ``order[:k]`` is a legal slice for a negative
        ``k`` and drops from the end rather than taking a prefix.
        """
        check_prefix_k(k, "top_k")
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
    """Rank every document under one operator and return a :class:`Ranking`.

    ``scores`` aligns to ``table``; ``_sorted_scores`` lets :func:`rank_all_operators` share one
    array; ``spec`` defaults to :data:`~tfidf_stability.ranking.sort_keys.PI`. Raises
    ``EmptyCorpusError``; ``NonFiniteValueError`` on NaN or inf in any mode (spec_addenda.md#g3).
    """
    n = len(scores)
    if n == 0:
        raise EmptyCorpusError("cannot rank an empty corpus")
    # Lenient mode rejects non-finite scores too: the addenda list them as rejected
    # inputs, not degenerate grid points, and a NaN sort key is undefined behaviour in the
    # native backend. One O(N) pass before an O(N log N) sort costs under 1%.
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

    The extra document gives the boundary margin ``m_k = score(r_k) - score(r_{k+1})``,
    by which section 7.3 stratifies every disagreement rate. ``sorted_scores`` stays
    complete, so margins and tie groups remain answerable.
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

    Margins depend only on the sorted score multiset, so they are identical under every
    operator; sharing the object makes that structural, and a test guards the sharing.
    """
    shared = sorted_scores_desc(scores)
    return {
        spec.name: rank(scores, table, spec, _sorted_scores=shared, **kwargs)  # type: ignore[arg-type]
        for spec in specs
    }
