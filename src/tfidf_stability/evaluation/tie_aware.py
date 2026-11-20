"""Tie-aware expected nDCG, recall and RR (McSherry and Najork, 2008).

Each metric is averaged over every order inside each group of equal scores, so no
tie-break enters; a query with no tied scores scores as ``metrics`` does, bit for bit.
Rationale: docs/engineering_notes.md#tie-aware-expected-metrics
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from tfidf_stability.analysis.query_grid import QueryGrid
from tfidf_stability.evaluation.metrics import MeanMetric, discount, mean_over_queries
from tfidf_stability.utils.validation import check_unique_ids

__all__ = [
    "ExactExpectation",
    "exact_expectation",
    "expected_means",
    "expected_metrics",
    "score_groups",
]


def score_groups(scores: Sequence[float]) -> list[tuple[int, ...]]:
    """Indices of equal scores, one group per value, highest value first.

    Equality is ``==``, so ``-0.0`` and ``0.0`` share a group; inside a group the
    indices ascend. Raises ``ValueError`` for a NaN, which no ranking can place.
    """
    if any(math.isnan(s) for s in scores):
        raise ValueError("a NaN score has no place in a ranking")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    groups: list[tuple[int, ...]] = []
    start = 0
    for end in range(1, len(order) + 1):
        if end == len(order) or scores[order[end]] != scores[order[start]]:
            groups.append(tuple(sorted(order[start:end])))
            start = end
    return groups


@dataclass(frozen=True, slots=True)
class ExactExpectation:
    """One query's expectations over its tie orders, held as exact fractions.

    ``dcg_terms`` holds one term per group, each discount taken as its float value;
    ``hits`` is the expected count of relevant documents in the first ``depth``.
    """

    dcg_terms: tuple[Fraction, ...]
    hits: Fraction
    rr: Fraction
    n_relevant: int


def _gain(grade: int) -> int:
    return grade if grade > 0 else 0


def _first_relevant(position: int, size: int, n_relevant: int, k: int) -> Fraction:
    """Expected ``1/rank`` of the first relevant document of a group, cut at ``k``.

    The group fills 0-indexed positions ``position`` onwards; the first relevant one
    sits at offset ``i`` with probability ``C(size-1-i, n_relevant-1) / C(size, n_relevant)``.
    """
    total = math.comb(size, n_relevant)
    last = min(size - n_relevant, k - position - 1)
    return sum(
        (
            Fraction(math.comb(size - 1 - i, n_relevant - 1), total) / (position + i + 1)
            for i in range(last + 1)
        ),
        Fraction(0),
    )


def exact_expectation(
    scores: Sequence[float],
    doc_ids: Sequence[str],
    judgements: Mapping[str, int],
    *,
    k: int,
    depth: int,
) -> ExactExpectation:
    """The exact expected DCG terms at ``k``, hits at ``depth`` and RR at ``k``.

    ``doc_ids[i]`` names ``scores[i]``; a judged document outside them is relevant but
    never ranked. Raises ``ValueError`` unless ``1 <= k <= depth``, or for mismatched lengths.
    """
    if not 1 <= k <= depth:
        raise ValueError(f"need 1 <= k <= depth, got k={k}, depth={depth}")
    if len(scores) != len(doc_ids):
        raise ValueError(f"{len(scores)} scores for {len(doc_ids)} documents")
    check_unique_ids(doc_ids)
    terms: list[Fraction] = []
    hits = Fraction(0)
    rr: Fraction | None = None
    position = 0
    for group in score_groups(scores):
        if position >= depth:
            break
        size = len(group)
        grades = [judgements.get(doc_ids[i], 0) for i in group]
        n_relevant = sum(1 for grade in grades if grade > 0)
        if position < k:
            inside = range(position, min(position + size, k))
            weight = sum((Fraction(discount(p + 1)) for p in inside), Fraction(0))
            terms.append(Fraction(sum(map(_gain, grades)), size) * weight)
            if rr is None and n_relevant:
                rr = _first_relevant(position, size, n_relevant, k)
        hits += Fraction(n_relevant * (min(position + size, depth) - position), size)
        position += size
    return ExactExpectation(
        dcg_terms=tuple(terms),
        hits=hits,
        rr=Fraction(0) if rr is None else rr,
        n_relevant=sum(1 for grade in judgements.values() if grade > 0),
    )


def expected_metrics(
    scores: Sequence[float],
    doc_ids: Sequence[str],
    judgements: Mapping[str, int],
    *,
    k: int,
    depth: int,
) -> dict[str, float | None]:
    """Expected ``ndcg@k``, ``recall@depth`` and ``rr@k``, or ``None`` each.

    Each DCG term and each ratio is rounded once from its exact value, and the ideal
    DCG is summed as ``ndcg_at_k`` sums it, so a tie-free query matches it bit for bit.
    """
    exact = exact_expectation(scores, doc_ids, judgements, k=k, depth=depth)
    if not exact.n_relevant:
        return {f"ndcg@{k}": None, f"recall@{depth}": None, f"rr@{k}": None}
    ideal = sorted((grade for grade in judgements.values() if grade > 0), reverse=True)[:k]
    idcg = math.fsum(grade * discount(rank) for rank, grade in enumerate(ideal, start=1))
    dcg = math.fsum(float(term) for term in exact.dcg_terms)
    return {
        f"ndcg@{k}": dcg / idcg,
        f"recall@{depth}": float(exact.hits / exact.n_relevant),
        f"rr@{k}": float(exact.rr),
    }


def expected_means(
    grid: QueryGrid,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    k: int,
    depth: int,
) -> dict[str, MeanMetric]:
    """Tie-aware means over every judged query, named as in E5.

    A judged query missing from ``grid`` is degenerate: it ranks nothing, as there.
    """
    scored = {q.query_id: q for q in grid.queries}
    rows: dict[str, dict[str, float | None]] = {}
    for query_id in sorted(qrels, key=lambda q: q.encode("utf-8")):
        query = scored.get(query_id)
        rows[query_id] = expected_metrics(
            () if query is None else query.scores,
            () if query is None else query.candidate_ids,
            qrels[query_id],
            k=k,
            depth=depth,
        )
    names = (f"ndcg@{k}", f"recall@{depth}", f"rr@{k}")
    return {name: mean_over_queries({q: row[name] for q, row in rows.items()}) for name in names}
