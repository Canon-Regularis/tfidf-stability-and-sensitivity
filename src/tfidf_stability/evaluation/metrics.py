"""nDCG, recall and reciprocal rank at k for one ranked list.

The list is scored as given, never re-sorted, so a tie order is measured. As in trec_eval,
gain is the grade, the discount is ``1 / log2(i + 1)`` and a grade of 0 or below is not relevant.
Rationale: docs/engineering_notes.md#relevance-metric-discount
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from tfidf_stability.utils.numerics import correctly_rounded_log_ratio
from tfidf_stability.utils.validation import check_unique_ids

__all__ = [
    "MeanMetric",
    "discount",
    "mean_over_queries",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]


@cache
def discount(rank: int) -> float:
    """``1 / log2(rank + 1)`` at a 1-indexed ``rank``, the same on every platform.

    Both logarithms are correctly rounded and so is the division, so no libm enters.
    """
    if rank < 1:
        raise ValueError(f"rank must be at least 1, got {rank}")
    return correctly_rounded_log_ratio(2, 1) / correctly_rounded_log_ratio(rank + 1, 1)


def _prefix(ranked_ids: Sequence[str], k: int) -> Sequence[str]:
    """The first ``k`` identifiers; refuses ``k < 1`` and a repeated identifier."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    prefix = ranked_ids[:k]
    check_unique_ids(prefix)
    return prefix


def ndcg_at_k(ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int) -> float | None:
    """nDCG at ``k``, or ``None`` when no document is relevant.

    The ideal list is every positive grade, highest first; an unjudged document gains 0.
    """
    prefix = _prefix(ranked_ids, k)
    ideal = sorted((grade for grade in judgements.values() if grade > 0), reverse=True)[:k]
    if not ideal:
        return None
    gains = ((rank, judgements.get(doc, 0)) for rank, doc in enumerate(prefix, start=1))
    dcg = math.fsum(grade * discount(rank) for rank, grade in gains if grade > 0)
    idcg = math.fsum(grade * discount(rank) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg


def recall_at_k(ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int) -> float | None:
    """Share of the relevant documents in the first ``k``; ``None`` if none exist."""
    prefix = _prefix(ranked_ids, k)
    relevant = {doc for doc, grade in judgements.items() if grade > 0}
    if not relevant:
        return None
    return sum(1 for doc in prefix if doc in relevant) / len(relevant)


def reciprocal_rank_at_k(
    ranked_ids: Sequence[str], judgements: Mapping[str, int], k: int
) -> float | None:
    """``1 / rank`` of the first relevant document in the first ``k``, else 0.

    ``None`` when no document is relevant, matching the other two metrics.
    """
    prefix = _prefix(ranked_ids, k)
    if not any(grade > 0 for grade in judgements.values()):
        return None
    for rank, doc in enumerate(prefix, start=1):
        if judgements.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True, slots=True)
class MeanMetric:
    """A metric averaged over the queries where it is defined."""

    mean: float
    n_queries: int
    n_undefined: int


def mean_over_queries(values: Mapping[str, float | None]) -> MeanMetric:
    """Average the defined values; each ``None`` is counted and left out.

    Raises ``ValueError`` when no value is defined.
    """
    defined = [value for value in values.values() if value is not None]
    if not defined:
        raise ValueError(f"no defined value among {len(values)} queries")
    return MeanMetric(
        mean=math.fsum(defined) / len(defined),
        n_queries=len(defined),
        n_undefined=len(values) - len(defined),
    )
