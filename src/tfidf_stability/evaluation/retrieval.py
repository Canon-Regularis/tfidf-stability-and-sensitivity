"""Score a grid of text queries against qrels under each identifier tie order.

Every variant ranks the same scores, so a metric that moves between variants moves through
the tie-break alone. A degenerate query ranks nothing and scores 0 where a document is relevant.
Rationale: docs/engineering_notes.md#identifier-tie-orders
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.metrics import (
    MeanMetric,
    mean_over_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from tfidf_stability.evaluation.tie_orders import TieOrder, tie_order_label, with_tie_order
from tfidf_stability.profiles.query_modes import QueryMode, QuerySet
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.margins import boundary_margin
from tfidf_stability.ranking.ranker import rank_top_k
from tfidf_stability.ranking.sort_keys import PI_SCORE
from tfidf_stability.ranking.tie_groups import tie_ball_interval
from tfidf_stability.similarity.scoring import InvertedIndex, ScoringScratch, taat_scores
from tfidf_stability.utils.validation import StrictMode
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

__all__ = [
    "QueryOutcome",
    "RetrievalReport",
    "Variant",
    "evaluate_retrieval",
    "score_text_queries",
]

#: The two variants every per-query row reports in full.
_ASC: str = "id_asc"
_DESC: str = "id_desc"


@dataclass(frozen=True, slots=True)
class Variant:
    """One identifier tie order; ``seed`` only for ``SHUFFLE``."""

    order: TieOrder
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", TieOrder(self.order))

    @property
    def label(self) -> str:
        return tie_order_label(self.order, self.seed)


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One judged query: its ``id_asc`` and ``id_desc`` metrics and its cutoff tie.

    ``margin`` is ``r_k - r_{k+1}`` (``None`` when undefined); ``tie_group`` counts the
    documents scoring exactly ``r_k``, so a group above 1 ending past ``k`` crosses it.
    """

    query_id: str
    degenerate: bool
    zero_vector: bool
    n_nonzero: int
    margin: float | None
    tie_group: int
    top_k_set_differs: bool
    top_k_order_differs: bool
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Mean metrics per variant, and one :class:`QueryOutcome` per judged query."""

    k: int
    depth: int
    means: dict[str, dict[str, MeanMetric]]
    queries: tuple[QueryOutcome, ...]

    @property
    def metric_names(self) -> tuple[str, str, str]:
        return (f"ndcg@{self.k}", f"recall@{self.depth}", f"rr@{self.k}")

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready: means, the spread over shuffles, counts and per-query rows."""
        shuffles = [label for label in self.means if label not in (_ASC, _DESC)]
        spread: dict[str, dict[str, float]] = {}
        for name in self.metric_names if shuffles else ():
            values = [self.means[label][name].mean for label in shuffles]
            spread[name] = {
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
            }
        return {
            "k": self.k,
            "depth": self.depth,
            "n_queries": len(self.queries),
            "n_degenerate": sum(q.degenerate for q in self.queries),
            "n_zero_vector": sum(q.zero_vector for q in self.queries),
            "n_tie_at_cutoff": sum(q.margin == 0.0 for q in self.queries),
            "n_top_k_set_differs": sum(q.top_k_set_differs for q in self.queries),
            "n_top_k_order_differs": sum(q.top_k_order_differs for q in self.queries),
            "means": {
                label: {
                    name: {"mean": m.mean, "n_queries": m.n_queries, "n_undefined": m.n_undefined}
                    for name, m in by_metric.items()
                }
                for label, by_metric in self.means.items()
            },
            "shuffle_spread": spread,
            "queries": [
                {
                    "query_id": q.query_id,
                    "degenerate": q.degenerate,
                    "zero_vector": q.zero_vector,
                    "n_nonzero": q.n_nonzero,
                    "margin": q.margin,
                    "tie_group": q.tie_group,
                    "top_k_set_differs": q.top_k_set_differs,
                    "top_k_order_differs": q.top_k_order_differs,
                    "metrics": q.metrics,
                }
                for q in self.queries
            ],
        }


def score_text_queries(
    query_set: QuerySet,
    model: TfidfModel,
    records: Sequence[Mapping[str, Any]],
    doc_ids: Sequence[str],
) -> QueryGrid:
    """Score text queries term at a time, as ``query_grid.evaluate`` would.

    The grid equals :func:`~tfidf_stability.analysis.query_grid.evaluate`'s: TAAT gives the bits
    ``cosine_against_corpus`` gives, and the attribute table is built once, since a text query
    keeps every document.
    """
    if query_set.mode is not QueryMode.TEXT:
        raise ValueError(f"expected {QueryMode.TEXT} queries, got {query_set.mode}")
    index = InvertedIndex.from_csr(model.matrix)
    table = AttributeTable.from_records(records)
    candidates = tuple(doc_ids)
    scratch = ScoringScratch()
    evaluated: list[EvaluatedQuery] = []
    for query in query_set.queries:
        if query.excluded:
            raise ValueError(f"text query {query.query_id!r} excludes documents")
        if query.is_degenerate:
            continue
        vector = TfidfVectoriser.transform_query(list(query.features), model)
        scores = taat_scores(vector, index, model.norms, model.reduction, scratch=scratch)
        evaluated.append(
            EvaluatedQuery(
                query_id=query.query_id,
                mode=query.mode,
                scores=tuple(scores),
                table=table,
                candidate_ids=candidates,
                target=None,
                n_excluded=0,
            )
        )
    return QueryGrid(
        queries=tuple(evaluated),
        mode=query_set.mode,
        aggregation=query_set.aggregation,
        min_interactions=query_set.min_interactions,
        exclude_profile_items=query_set.exclude_profile_items,
        n_users=query_set.n_users,
        n_degenerate=len(query_set.queries) - len(evaluated),
        n_zero_vector=sum(1 for q in evaluated if q.is_zero_vector),
    )


def _metrics(
    ranked: Sequence[str], judgements: Mapping[str, int], k: int, depth: int
) -> dict[str, float | None]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked, judgements, k),
        f"recall@{depth}": recall_at_k(ranked, judgements, depth),
        f"rr@{k}": reciprocal_rank_at_k(ranked, judgements, k),
    }


def _ranked(evaluated: EvaluatedQuery, table: AttributeTable, depth: int) -> list[str]:
    ranking = rank_top_k(evaluated.scores, table, PI_SCORE, k=depth, mode=StrictMode.LENIENT)
    return [evaluated.candidate_ids[i] for i in ranking.order[:depth]]


def _cutoff(sorted_scores: Sequence[float], k: int) -> tuple[float | None, int]:
    """The boundary margin at ``k`` and the size of the exact tie at position ``k``."""
    margin = boundary_margin(sorted_scores, k, mode=StrictMode.LENIENT)
    lo, hi = tie_ball_interval(sorted_scores, min(k, len(sorted_scores)) - 1, 0.0)
    return (margin.value if margin.defined else None), hi - lo


def evaluate_retrieval(
    grid: QueryGrid,
    qrels: Mapping[str, Mapping[str, int]],
    doc_ids: Sequence[str],
    *,
    k: int,
    depth: int,
    seeds: int,
) -> RetrievalReport:
    """Rank each judged query to ``depth`` under every tie-order variant.

    ``grid`` must hold text queries over all of ``doc_ids``; a judged query missing from it
    is degenerate. Raises ``ValueError`` for another query mode, a candidate set that is not
    the corpus, a query without judgements, ``k > depth`` or a negative ``seeds``.
    """
    if grid.mode is not QueryMode.TEXT:
        raise ValueError(f"expected a grid of {QueryMode.TEXT} queries, got {grid.mode}")
    if not 1 <= k <= depth:
        raise ValueError(f"need 1 <= k <= depth, got k={k}, depth={depth}")
    if seeds < 0:
        raise ValueError(f"seeds must be non-negative, got {seeds}")
    corpus = tuple(doc_ids)
    evaluated = {q.query_id: q for q in grid.queries}
    for scored in grid.queries:
        if scored.candidate_ids != corpus:
            raise ValueError(f"query {scored.query_id!r} does not rank the whole corpus")
        if scored.query_id not in qrels:
            raise ValueError(f"query {scored.query_id!r} has no judgements")

    variants = [Variant(TieOrder.ID_ASC), Variant(TieOrder.ID_DESC)]
    variants += [Variant(TieOrder.SHUFFLE, seed) for seed in range(seeds)]
    tables: dict[str, AttributeTable] = {}
    if grid.queries:
        base = grid.queries[0].table
        tables = {v.label: with_tie_order(base, v.order, seed=v.seed) for v in variants}

    per_variant: dict[str, dict[str, dict[str, float | None]]] = {v.label: {} for v in variants}
    outcomes: list[QueryOutcome] = []
    for query_id in sorted(qrels, key=lambda q: q.encode("utf-8")):
        judgements = qrels[query_id]
        query = evaluated.get(query_id)
        ranked = {
            v.label: [] if query is None else _ranked(query, tables[v.label], depth)
            for v in variants
        }
        for v in variants:
            per_variant[v.label][query_id] = _metrics(ranked[v.label], judgements, k, depth)
        asc, desc = ranked[_ASC][:k], ranked[_DESC][:k]
        margin, tie_group = (
            (None, 0) if query is None else _cutoff(sorted(query.scores, reverse=True), k)
        )
        outcomes.append(
            QueryOutcome(
                query_id=query_id,
                degenerate=query is None,
                zero_vector=query is not None and query.is_zero_vector,
                n_nonzero=0 if query is None else sum(1 for s in query.scores if s != 0.0),
                margin=margin,
                tie_group=tie_group,
                top_k_set_differs=set(asc) != set(desc),
                top_k_order_differs=asc != desc,
                metrics={_ASC: per_variant[_ASC][query_id], _DESC: per_variant[_DESC][query_id]},
            )
        )

    names = (f"ndcg@{k}", f"recall@{depth}", f"rr@{k}")
    means = {
        label: {
            name: mean_over_queries({q: row[name] for q, row in rows.items()}) for name in names
        }
        for label, rows in per_variant.items()
    }
    return RetrievalReport(k=k, depth=depth, means=means, queries=tuple(outcomes))
