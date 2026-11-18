"""Queries a dataset ships as text, run through the corpus's own pipeline.

Each query keeps every document as a candidate, so :attr:`Query.n_candidates` is the
corpus size throughout. The pipeline must be the one the corpus was fitted with, or a
query term cannot meet the document term it names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from tfidf_stability.profiles.query_modes import Query, QueryMode, QuerySet
from tfidf_stability.profiles.user_profile import ProfileAggregation

if TYPE_CHECKING:
    from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline

__all__ = ["text_queries"]


def text_queries(
    queries: Mapping[str, str], pipeline: PreprocessingPipeline, *, n_documents: int
) -> QuerySet:
    """One :class:`Query` per text, ordered by UTF-8 query id.

    A text whose features are all removed by ``pipeline`` gives a degenerate query,
    which :func:`~tfidf_stability.analysis.query_grid.evaluate` counts and skips.
    """
    if n_documents < 0:
        raise ValueError(f"n_documents must be non-negative, got {n_documents}")
    built = tuple(
        Query(
            query_id=query_id,
            mode=QueryMode.TEXT,
            features=tuple(pipeline.preprocess(queries[query_id])),
            n_candidates=n_documents,
        )
        for query_id in sorted(queries, key=lambda q: q.encode("utf-8"))
    )
    # The set's profile fields describe interaction-built queries; text queries
    # use no profile, so they take the values that build none.
    return QuerySet(
        queries=built,
        mode=QueryMode.TEXT,
        aggregation=ProfileAggregation.TEXT_CONCAT,
        min_interactions=0,
        exclude_profile_items=False,
        n_users=0,
    )
