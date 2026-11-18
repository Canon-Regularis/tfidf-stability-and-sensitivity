"""Query construction, and the leave-one-out protocol (section 7.1, G10).

Each item of every user passing :func:`~tfidf_stability.profiles.user_profile.eligible_users` is a
:class:`Query` target and a candidate; by default the rest are not, so :attr:`Query.n_candidates`
varies (``spec_addenda.md#g10``). Rationale: docs/engineering_notes.md#leave-one-out-protocol
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from tfidf_stability.profiles.user_profile import (
    ProfileAggregation,
    UserProfile,
    build_profile,
    eligible_users,
)

__all__ = [
    "Query",
    "QueryMode",
    "QuerySet",
    "item_as_query",
    "leave_one_out_queries",
    "user_profile_queries",
]


class QueryMode(str, Enum):
    """Which construction produced a query (section 7.1)."""

    USER_PROFILE = "user_profile"
    LEAVE_ONE_OUT = "leave_one_out"
    ITEM_AS_QUERY = "item_as_query"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Query:
    """One evaluable query, with the protocol decisions that shaped it.

    ``query_id`` is unique and deterministic, ``{user}::{held_out}`` for a fold; ``profile`` is
    ``None`` for item-as-query. ``excluded`` documents leave the candidates (decision 3) and never
    include ``target``; ``n_candidates`` is corpus size minus exclusions, which varies per fold.
    """

    query_id: str
    mode: QueryMode
    features: tuple[str, ...]
    profile: UserProfile | None = None
    target: str | None = None
    excluded: frozenset[str] = field(default_factory=frozenset)
    n_candidates: int = 0

    def __post_init__(self) -> None:
        """Normalise ``mode`` to its member; the grid layer selects on ``is``."""
        object.__setattr__(self, "mode", QueryMode(self.mode))

    @property
    def is_degenerate(self) -> bool:
        """Whether the query has no features at all.

        Distinct from a zero-vector query, which also arises when every feature
        is out of vocabulary. Both are legitimate and both are reported; this
        one is visible before embedding.
        """
        return not self.features

    def candidate_indices(self, doc_ids: Sequence[str]) -> tuple[int, ...]:
        """Positions of the documents this query may retrieve.

        The exclusion is applied by identifier: indices are positional and a
        caller may hold a differently-ordered corpus.
        """
        return tuple(i for i, d in enumerate(doc_ids) if d not in self.excluded)


@dataclass(frozen=True, slots=True)
class QuerySet:
    """A reproducible collection of queries, with its protocol recorded."""

    queries: tuple[Query, ...]
    mode: QueryMode
    aggregation: ProfileAggregation
    min_interactions: int
    exclude_profile_items: bool
    #: G14: section 7.1 defers these counts to "the dataset configuration", so
    #: they are fixed here and travel into the run manifest.
    n_users: int

    def __post_init__(self) -> None:
        """Normalise both enums to their members.

        ``build_profile`` selects the divisor on ``aggregation`` with ``is``,
        so an uncoerced string sums where the value asks for a mean.
        """
        object.__setattr__(self, "mode", QueryMode(self.mode))
        object.__setattr__(self, "aggregation", ProfileAggregation(self.aggregation))

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.queries)

    @property
    def candidate_spread(self) -> tuple[int, int, int]:
        """``(min, median, max)`` candidate count across queries (G19).

        Excluding a user's profile makes ``N`` differ per query, so a margin
        distribution pools populations of different sizes and a disagreement
        rate has a varying denominator. The spread records that variation.
        """
        counts = sorted(q.n_candidates for q in self.queries)
        if not counts:
            return (0, 0, 0)
        return (counts[0], counts[len(counts) // 2], counts[-1])

    def provenance(self) -> dict[str, object]:
        """The protocol, for the run manifest."""
        low, median, high = self.candidate_spread
        return {
            "mode": str(self.mode),
            "aggregation": str(self.aggregation),
            "min_interactions": self.min_interactions,
            "exclude_profile_items": self.exclude_profile_items,
            "n_users": self.n_users,
            "n_queries": len(self.queries),
            # G19: not constant, so it is recorded rather than inferred.
            "candidates_min": low,
            "candidates_median": median,
            "candidates_max": high,
        }


def _require_feature_bearing(aggregation: ProfileAggregation) -> None:
    """Reject an aggregation the query grid cannot carry.

    Raises ValueError unless ``aggregation`` is TEXT_CONCAT: a `Query` travels as a feature
    stream, the vector-space aggregations leave `UserProfile.features` empty, and `evaluate`
    would skip every such query as degenerate. `build_query_grid` raises alike for item-as-query.
    """
    if ProfileAggregation(aggregation) is not ProfileAggregation.TEXT_CONCAT:
        raise ValueError(
            f"{aggregation} builds a profile vector rather than a feature stream, "
            f"and the query grid carries features; use "
            f"{ProfileAggregation.TEXT_CONCAT} here, or embed the profile vector "
            f"directly with profile_vector()"
        )


def user_profile_queries(
    grouped: Mapping[str, Sequence[str]],
    features_by_doc: Mapping[str, Sequence[str]],
    *,
    aggregation: ProfileAggregation = ProfileAggregation.TEXT_CONCAT,
    min_interactions: int = 1,
    exclude_profile_items: bool = True,
    doc_ids: Sequence[str] | None = None,
) -> QuerySet:
    """One query per user, from their whole interaction history.

    No item is held out, so there is no retrieval target. The user's own items
    still leave the candidate set by default, for the same reason as in a fold:
    they contributed the query text and would otherwise retrieve themselves.
    """
    _require_feature_bearing(aggregation)
    users = (
        eligible_users(grouped, max(min_interactions, 2))
        if min_interactions >= 2
        else tuple(sorted(grouped, key=lambda u: u.encode("utf-8")))
    )
    total_docs = len(doc_ids) if doc_ids is not None else len(features_by_doc)

    queries = []
    for user in users:
        items = tuple(grouped[user])
        profile = build_profile(user, items, features_by_doc, aggregation)
        excluded = frozenset(items) if exclude_profile_items else frozenset()
        queries.append(
            Query(
                query_id=user,
                mode=QueryMode.USER_PROFILE,
                features=profile.features,
                profile=profile,
                excluded=excluded,
                n_candidates=total_docs - len(excluded),
            )
        )

    return QuerySet(
        queries=tuple(queries),
        mode=QueryMode.USER_PROFILE,
        aggregation=aggregation,
        min_interactions=min_interactions,
        exclude_profile_items=exclude_profile_items,
        n_users=len(users),
    )


def leave_one_out_queries(
    grouped: Mapping[str, Sequence[str]],
    features_by_doc: Mapping[str, Sequence[str]],
    *,
    aggregation: ProfileAggregation = ProfileAggregation.TEXT_CONCAT,
    min_interactions: int = 5,
    exclude_profile_items: bool = True,
    doc_ids: Sequence[str] | None = None,
) -> QuerySet:
    """Every fold for every eligible user, as a :class:`QuerySet` (G10, G11).

    Each item in ``grouped`` (:func:`~tfidf_stability.profiles.user_profile.group_interactions`) of
    a user meeting ``min_interactions`` is one target; the rest leave the candidates unless
    ``exclude_profile_items`` is False. ``doc_ids``, else ``features_by_doc``, sizes the corpus.
    """
    _require_feature_bearing(aggregation)
    users = eligible_users(grouped, min_interactions)
    total_docs = len(doc_ids) if doc_ids is not None else len(features_by_doc)

    queries = []
    for user in users:
        items = tuple(grouped[user])
        for held_out in items:
            remaining = tuple(d for d in items if d != held_out)
            profile = build_profile(user, remaining, features_by_doc, aggregation)
            # The target stays in: it is what the fold retrieves.
            excluded = frozenset(remaining) if exclude_profile_items else frozenset()
            queries.append(
                Query(
                    query_id=f"{user}::{held_out}",
                    mode=QueryMode.LEAVE_ONE_OUT,
                    features=profile.features,
                    profile=profile,
                    target=held_out,
                    excluded=excluded,
                    n_candidates=total_docs - len(excluded),
                )
            )

    return QuerySet(
        queries=tuple(queries),
        mode=QueryMode.LEAVE_ONE_OUT,
        aggregation=aggregation,
        min_interactions=min_interactions,
        exclude_profile_items=exclude_profile_items,
        n_users=len(users),
    )


def item_as_query(
    doc_id: str,
    features_by_doc: Mapping[str, Sequence[str]],
    *,
    exclude_self: bool = True,
    doc_ids: Sequence[str] | None = None,
) -> Query:
    """A single item used as its own query.

    Section 7.1 lists this as "part of the implementation but not evaluated in
    the present experiments". ``exclude_self`` defaults to ``True``: an item is
    its own nearest neighbour at similarity 1, which is uninformative.
    """
    total_docs = len(doc_ids) if doc_ids is not None else len(features_by_doc)
    excluded = frozenset({doc_id}) if exclude_self else frozenset()
    return Query(
        query_id=f"item::{doc_id}",
        mode=QueryMode.ITEM_AS_QUERY,
        features=tuple(features_by_doc[doc_id]),
        excluded=excluded,
        n_candidates=total_docs - len(excluded),
    )
