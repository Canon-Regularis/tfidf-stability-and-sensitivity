"""The section 7.1 query grid, shared by every experiment runner.

Section 7.1 constructs queries as user-profile queries and leave-one-out folds,
with item-as-query implemented but not evaluated. The experiment runners
originally used truncated document prefixes, a different and much easier
protocol: a prefix always retrieves its own source document at high similarity,
whereas a fold has to find a held-out item from the rest of a user's history.
That moves the score distribution, the margin distribution and every A1 number,
so both runners now go through the grid built here.

Per-query candidate sets
------------------------
Each query carries its own exclusions, so the candidate set differs per query
(G19). A fold excludes the remaining profile items (they contributed the query
text and would otherwise retrieve themselves) and keeps the target scoreable.
Consequences:

* ``N`` differs between queries, so ``k`` may exceed the candidate count. Those
  queries are reported rather than silently clamped.
* Margins are computed over the candidate scores only. Over the full corpus they
  would include documents the query could never retrieve and would inflate the
  apparent separation.
* The attribute table must be restricted to the same subset, or the tie-break
  ranks over non-candidates.

Degenerate queries
------------------
A profile can be empty after preprocessing, and a non-empty query can embed to
the zero vector when every feature is out of vocabulary. Both are legitimate
section 7.1 outcomes and are counted separately: G3 excludes them from margin
distributions and keeps them in ablations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tfidf_stability.profiles.query_modes import (
    Query,
    QueryMode,
    QuerySet,
    leave_one_out_queries,
    user_profile_queries,
)
from tfidf_stability.profiles.user_profile import Interaction, group_interactions
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

__all__ = ["EvaluatedQuery", "QueryGrid", "build_query_grid", "evaluate"]


@dataclass(frozen=True, slots=True)
class EvaluatedQuery:
    """One query, scored against its own candidate set."""

    query_id: str
    mode: str
    scores: tuple[float, ...]
    #: Attribute table restricted to this query's candidates, so the tie-break
    #: only ranks documents the query was allowed to retrieve.
    table: AttributeTable
    candidate_ids: tuple[str, ...]
    target: str | None
    n_excluded: int

    @property
    def n_candidates(self) -> int:
        return len(self.scores)

    @property
    def is_zero_vector(self) -> bool:
        """Every score is zero, so attributes alone decide the ranking.

        Observational rather than structural: a query can be non-empty and still
        embed to the zero vector when every feature is out of vocabulary.

        The emptiness guard earns its place. ``all()`` over no scores is
        ``True``, so a query with no candidates once answered yes and entered
        the published ``n_zero_vector`` as a pure-attribute ranking, with
        nothing there to rank. No current dataset produces an empty candidate
        set, so no published number moves.
        """
        return bool(self.scores) and all(s == 0.0 for s in self.scores)


@dataclass(frozen=True, slots=True)
class QueryGrid:
    """The evaluated grid, with the protocol that produced it."""

    queries: tuple[EvaluatedQuery, ...]
    mode: str
    aggregation: str
    min_interactions: int
    exclude_profile_items: bool
    n_users: int
    n_degenerate: int
    n_zero_vector: int

    def __len__(self) -> int:
        return len(self.queries)

    def score_vectors(self) -> list[list[float]]:
        return [list(q.scores) for q in self.queries]

    def provenance(self) -> dict[str, Any]:
        """The protocol block for the run manifest."""
        return {
            "query_mode": self.mode,
            "aggregation": self.aggregation,
            "min_interactions": self.min_interactions,
            "exclude_profile_items": self.exclude_profile_items,
            "n_users": self.n_users,
            "n_queries": len(self.queries),
            "n_degenerate_profiles": self.n_degenerate,
            "n_zero_vector_queries": self.n_zero_vector,
            # Candidate sets vary per query (G19), so a single N would be a
            # fiction; report both ends.
            "n_candidates_min": min((q.n_candidates for q in self.queries), default=0),
            "n_candidates_max": max((q.n_candidates for q in self.queries), default=0),
        }


def build_query_grid(
    interactions: Sequence[tuple[str, str, float]],
    features_by_doc: Mapping[str, Sequence[str]],
    doc_ids: Sequence[str],
    *,
    mode: QueryMode = QueryMode.LEAVE_ONE_OUT,
    min_interactions: int = 5,
    exclude_profile_items: bool = True,
    limit: int | None = None,
) -> QuerySet:
    """Construct the section 7.1 query set.

    Args:
        interactions: ``(user_id, doc_id, weight)`` triples.
        features_by_doc: Preprocessed feature streams, keyed by document id.
        doc_ids: Corpus order, used to size the candidate sets.
        mode: Which construction. Item-as-query is implemented in
            :mod:`~tfidf_stability.profiles.query_modes` but section 7.1
            excludes it from the reported experiments, so it is rejected here.
        min_interactions: Eligibility threshold.
        exclude_profile_items: G10 decision 3.
        limit: Cap the number of queries, taking a deterministic prefix. Every
            fold of every eligible user is a lot of queries and the cap keeps CI
            fast. Capping is always reported.

    Raises:
        ValueError: If ``mode`` is item-as-query.
    """
    # Loaders hand back plain triples; group_interactions wants the typed record.
    # Its `weight` feeds eligibility only, since section 7.1 specifies an
    # unweighted aggregation.
    grouped = group_interactions(
        Interaction(user_id=u, doc_id=d, weight=w) for u, d, w in interactions
    )

    if mode is QueryMode.LEAVE_ONE_OUT:
        query_set = leave_one_out_queries(
            grouped,
            features_by_doc,
            min_interactions=min_interactions,
            exclude_profile_items=exclude_profile_items,
            doc_ids=doc_ids,
        )
    elif mode is QueryMode.USER_PROFILE:
        query_set = user_profile_queries(
            grouped,
            features_by_doc,
            min_interactions=min_interactions,
            exclude_profile_items=exclude_profile_items,
            doc_ids=doc_ids,
        )
    else:
        raise ValueError(
            f"{mode} is implemented but section 7.1 excludes it from the reported "
            f"experiments; use USER_PROFILE or LEAVE_ONE_OUT"
        )

    if limit is not None and len(query_set.queries) > limit:
        # A deterministic prefix rather than a sample: the grid must not move
        # when the cap changes, or two runs at different caps are incomparable.
        query_set = QuerySet(
            queries=query_set.queries[:limit],
            mode=query_set.mode,
            aggregation=query_set.aggregation,
            min_interactions=query_set.min_interactions,
            exclude_profile_items=query_set.exclude_profile_items,
            n_users=query_set.n_users,
        )
    return query_set


def _score_one(
    query: Query,
    model: TfidfModel,
    records: Sequence[Mapping[str, Any]],
    doc_ids: Sequence[str],
) -> EvaluatedQuery:
    """Score one query against its own candidate set."""
    candidates = query.candidate_indices(doc_ids)
    documents = [model.document(i) for i in candidates]
    norms = [model.norms[i] for i in candidates]

    embedded = TfidfVectoriser.transform_query(list(query.features), model)
    scores = cosine_against_corpus(embedded, documents, norms, model.reduction)

    # Same subset as the scores, so the tie-break cannot reach a non-candidate.
    table = AttributeTable.from_records([records[i] for i in candidates])

    return EvaluatedQuery(
        query_id=query.query_id,
        mode=str(query.mode),
        scores=tuple(scores),
        table=table,
        candidate_ids=tuple(doc_ids[i] for i in candidates),
        target=query.target,
        n_excluded=len(doc_ids) - len(candidates),
    )


def evaluate(
    query_set: QuerySet,
    model: TfidfModel,
    records: Sequence[Mapping[str, Any]],
    doc_ids: Sequence[str],
) -> QueryGrid:
    """Score every query in the set against its own candidate set."""
    evaluated: list[EvaluatedQuery] = []
    n_degenerate = 0
    for query in query_set.queries:
        if query.is_degenerate:
            # No features at all. Scoring it would give an all-zero vector
            # indistinguishable from an out-of-vocabulary query, and those are
            # different findings, so count it here and skip.
            n_degenerate += 1
            continue
        evaluated.append(_score_one(query, model, records, doc_ids))

    return QueryGrid(
        queries=tuple(evaluated),
        mode=str(query_set.mode),
        aggregation=str(query_set.aggregation),
        min_interactions=query_set.min_interactions,
        exclude_profile_items=query_set.exclude_profile_items,
        n_users=query_set.n_users,
        n_degenerate=n_degenerate,
        n_zero_vector=sum(1 for q in evaluated if q.is_zero_vector),
    )
