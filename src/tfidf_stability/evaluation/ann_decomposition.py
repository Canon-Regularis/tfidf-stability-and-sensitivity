"""ANN top-k differences split into approximation, numerics and ties.

The exact top ``k`` ranks the whole corpus by the reference scores under ``id_asc``;
an ANN list keeps its file order and its own scores. Order-only differences are classed too.
Rationale: docs/engineering_notes.md#ann-decomposition
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.external import ExternalDataError
from tfidf_stability.evaluation.score_store import ScoreStore
from tfidf_stability.evaluation.tie_orders import TieOrder, with_tie_order
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.margins import boundary_margin
from tfidf_stability.ranking.ranker import rank_top_k
from tfidf_stability.ranking.sort_keys import PI_SCORE
from tfidf_stability.utils.numerics import same_bits
from tfidf_stability.utils.validation import StrictMode

__all__ = [
    "MISSING_RUNS",
    "AnnDecomposition",
    "Difference",
    "QueryDecomposition",
    "classify_query",
    "decompose",
    "run_ann",
]

#: The payload's note while no ANN run exists.
MISSING_RUNS = "HNSW builds arrive with #16"
#: The group every run joins when the caller names none.
_DEFAULT_GROUP = "all"


class Difference(str, Enum):
    """Why an ANN top-k list differs from the exact one."""

    APPROXIMATION = "approximation"
    NUMERICS = "numerics"
    TIE_BREAKING = "tie_breaking"
    ORDER_TIES = "order_ties"
    ORDER_NUMERICS = "order_numerics"

    def __str__(self) -> str:
        return self.value


#: The classes of a missed document, and of a same-set list in another order.
_MISS_KINDS = (Difference.APPROXIMATION, Difference.NUMERICS, Difference.TIE_BREAKING)
_ORDER_KINDS = (Difference.ORDER_TIES, Difference.ORDER_NUMERICS)


@dataclass(frozen=True, slots=True)
class QueryDecomposition:
    """One query at cutoff ``k``: the class of each missed exact document.

    ``order`` classes a list holding the exact set in another order, else ``None``;
    ``n_scores_differ`` counts listed documents whose ANN score's bits differ from the exact;
    ``overlap`` is the share of the exact top ``k`` the ANN top ``k`` keeps.
    """

    query_id: str
    k: int
    missed: Mapping[str, Difference]
    order: Difference | None
    n_listed: int
    n_scores_differ: int
    boundary_tie: bool
    overlap: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "missed", {d: Difference(c) for d, c in self.missed.items()})
        if self.order is not None:
            object.__setattr__(self, "order", Difference(self.order))

    @property
    def differs(self) -> bool:
        return bool(self.missed) or self.order is not None

    def count(self, kind: Difference) -> int:
        return sum(1 for c in self.missed.values() if c is kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "k": self.k,
            "missed": {doc: str(kind) for doc, kind in self.missed.items()},
            "counts": {str(kind): self.count(kind) for kind in _MISS_KINDS},
            "order": None if self.order is None else str(self.order),
            "n_listed": self.n_listed,
            "n_scores_differ": self.n_scores_differ,
            "boundary_tie": self.boundary_tie,
            "overlap": self.overlap,
        }


@dataclass(frozen=True, slots=True)
class AnnDecomposition:
    """One run against the exact reference at cutoff ``k``.

    ``queries`` holds only the queries whose top ``k`` differs; ``mean_overlap`` averages
    the share of the exact top ``k`` each ANN top ``k`` keeps, a quantity without qrels.
    """

    label: str
    k: int
    n_queries: int
    n_sets_differ: int
    n_boundary_ties: int
    mean_overlap: float
    queries: tuple[QueryDecomposition, ...]

    @property
    def totals(self) -> dict[str, int]:
        """Missed documents per miss class, then queries per order class."""
        out = {str(kind): sum(q.count(kind) for q in self.queries) for kind in _MISS_KINDS}
        out.update({str(kind): sum(q.order is kind for q in self.queries) for kind in _ORDER_KINDS})
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "k": self.k,
            "n_queries": self.n_queries,
            "n_sets_differ": self.n_sets_differ,
            "n_boundary_ties": self.n_boundary_ties,
            "mean_overlap": self.mean_overlap,
            "totals": self.totals,
            "queries": [q.as_dict() for q in self.queries],
        }


def _view(reference: EvaluatedQuery) -> tuple[AttributeTable, dict[str, int]]:
    """The reference's ``id_asc`` table and each candidate's index."""
    index = {doc: i for i, doc in enumerate(reference.candidate_ids)}
    return with_tie_order(reference.table, TieOrder.ID_ASC), index


def _listed_order(
    query_id: str, listed: Mapping[str, float], index: Mapping[str, int]
) -> list[str]:
    """``listed``'s documents in file order, once each one is checked."""
    previous = math.inf
    for doc, score in listed.items():
        if doc not in index:
            raise ExternalDataError(f"query {query_id!r} lists {doc!r}, outside the corpus")
        if not math.isfinite(score):
            raise ExternalDataError(f"query {query_id!r} gives {doc!r} the score {score!r}")
        if score > previous:
            raise ExternalDataError(f"query {query_id!r} is not sorted descending at {doc!r}")
        previous = score
    return list(listed)


def _exact_top(reference: EvaluatedQuery, table: AttributeTable, k: int) -> tuple[list[str], bool]:
    """The exact top ``k`` under ``table``, and whether ``k`` cuts a tie."""
    ranking = rank_top_k(reference.scores, table, PI_SCORE, k=k, mode=StrictMode.LENIENT)
    margin = boundary_margin(ranking.sorted_scores, k, mode=StrictMode.LENIENT)
    top = [reference.candidate_ids[i] for i in ranking.order[:k]]
    return top, margin.defined and margin.value == 0.0


def _missed(
    exact_top: Sequence[str],
    ann_top: Sequence[str],
    rescored_top: Sequence[str],
    listed: Mapping[str, float],
    exact_score: Mapping[str, float],
) -> dict[str, Difference]:
    """Class each exact top-k document absent from the ANN top ``k``.

    A listed one is numerics if exact re-ranking keeps it, else tie breaking. An unlisted
    one is tie breaking while a re-ranked document outside the exact top ``k`` with an
    equal exact score is left to stand in for it, each used once; else approximation.
    """
    exact_set, kept, rescored = set(exact_top), set(ann_top), set(rescored_top)
    substitutes = Counter(exact_score[doc] for doc in rescored_top if doc not in exact_set)
    missed: dict[str, Difference] = {}
    for doc in exact_top:
        if doc in kept:
            continue
        if doc in listed:
            missed[doc] = Difference.NUMERICS if doc in rescored else Difference.TIE_BREAKING
        elif substitutes[exact_score[doc]] > 0:
            substitutes[exact_score[doc]] -= 1
            missed[doc] = Difference.TIE_BREAKING
        else:
            missed[doc] = Difference.APPROXIMATION
    return missed


def _classify(
    reference: EvaluatedQuery,
    view: tuple[AttributeTable, dict[str, int]],
    listed: Mapping[str, float],
    k: int,
) -> QueryDecomposition:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    table, index = view
    docs = _listed_order(reference.query_id, listed, index)
    exact_top, boundary_tie = _exact_top(reference, table, k)
    exact = {doc: reference.scores[index[doc]] for doc in (*docs, *exact_top)}
    ann_top = docs[:k]
    # Re-ranked by exact score, a tie keeps the list's own order.
    rescored = sorted(range(len(docs)), key=lambda i: (-exact[docs[i]], i))[:k]
    rescored_top = [docs[i] for i in rescored]
    missed = _missed(exact_top, ann_top, rescored_top, listed, exact)
    order = None
    if not missed and ann_top != exact_top:
        order = Difference.ORDER_TIES if ann_top == rescored_top else Difference.ORDER_NUMERICS
    return QueryDecomposition(
        query_id=reference.query_id,
        k=k,
        missed=missed,
        order=order,
        n_listed=len(docs),
        n_scores_differ=sum(1 for doc in docs if not same_bits(listed[doc], exact[doc])),
        boundary_tie=boundary_tie,
        overlap=len(set(ann_top) & set(exact_top)) / len(exact_top),
    )


def classify_query(
    reference: EvaluatedQuery, listed: Mapping[str, float], *, k: int
) -> QueryDecomposition:
    """Class each difference of ``listed``'s top ``k`` from the exact one.

    ``listed`` maps documents to ANN scores in file order. Raises ``ExternalDataError``
    for a document outside the corpus, a non-finite score or a rising score, ``ValueError``
    for ``k < 1``.
    """
    return _classify(reference, _view(reference), listed, k)


def decompose(
    reference: QueryGrid,
    run: Mapping[str, Mapping[str, float]],
    *,
    k: int,
    label: str,
) -> AnnDecomposition:
    """Class run ``label``'s top-``k`` differences on every reference query.

    A reference query the run omits lists nothing. Raises ``ExternalDataError`` for a run
    query the reference does not score, ``ValueError`` for an empty reference or ``k < 1``.
    """
    if not reference.queries:
        raise ValueError("the reference scores no queries")
    unknown = sorted(set(run) - {q.query_id for q in reference.queries})
    if unknown:
        raise ExternalDataError(f"run {label!r} lists {unknown[0]!r}, a query the reference lacks")
    views: dict[tuple[int, int], tuple[AttributeTable, dict[str, int]]] = {}
    rows: list[QueryDecomposition] = []
    for query in reference.queries:
        key = (id(query.table), id(query.candidate_ids))
        if key not in views:
            views[key] = _view(query)
        rows.append(_classify(query, views[key], run.get(query.query_id, {}), k))
    return AnnDecomposition(
        label=label,
        k=k,
        n_queries=len(rows),
        n_sets_differ=sum(1 for row in rows if row.missed),
        n_boundary_ties=sum(1 for row in rows if row.boundary_tie),
        mean_overlap=math.fsum(row.overlap for row in rows) / len(rows),
        queries=tuple(row for row in rows if row.differs),
    )


def _spreads(
    results: Mapping[str, Sequence[AnnDecomposition]],
    membership: Mapping[str, str],
    cutoffs: Sequence[int],
) -> list[dict[str, Any]]:
    """Per group of two or more runs and per cutoff, the range of mean overlap."""
    spreads: list[dict[str, Any]] = []
    for group in sorted(set(membership.values())):
        members = [name for name in results if membership[name] == group]
        if len(members) < 2:
            continue
        for i, k in enumerate(cutoffs):
            overlap = {name: results[name][i].mean_overlap for name in members}
            low = min(members, key=overlap.__getitem__)
            high = max(members, key=overlap.__getitem__)
            spreads.append(
                {
                    "experiment": "e9",
                    "metric": f"overlap@{k}",
                    "spread": overlap[high] - overlap[low],
                    "source": {"group": group, "min_run": low, "max_run": high},
                }
            )
    return spreads


def run_ann(
    store: ScoreStore,
    reference_label: str,
    runs: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    cutoffs: Sequence[int],
    groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """E9: each run decomposed at each cutoff, and ``overlap@k`` spreads per group.

    ``groups`` maps each run to a group, one group by default. Raises ``ValueError`` for
    no, repeated or non-positive cutoffs, or groups naming other than the runs.
    """
    cuts = tuple(cutoffs)
    if not cuts or len(set(cuts)) != len(cuts) or min(cuts) < 1:
        raise ValueError(f"cutoffs must be distinct positive integers, got {list(cuts)}")
    membership = dict.fromkeys(runs, _DEFAULT_GROUP) if groups is None else dict(groups)
    if set(membership) != set(runs):
        raise ValueError("groups must name every run and nothing else")
    reference = store.grid(reference_label)
    results = {
        name: [decompose(reference, runs[name], k=k, label=name) for k in cuts]
        for name in sorted(runs, key=lambda name: name.encode("utf-8"))
    }
    return {
        "reference": reference_label,
        "cutoffs": list(cuts),
        "runs": [
            {
                "label": name,
                "group": membership[name],
                "decompositions": [d.as_dict() for d in decompositions],
            }
            for name, decompositions in results.items()
        ],
        "missing": None if runs else MISSING_RUNS,
        "spreads": _spreads(results, membership, cuts),
    }
