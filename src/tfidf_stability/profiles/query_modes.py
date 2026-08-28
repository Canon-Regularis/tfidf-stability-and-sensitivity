"""Query construction, and the leave-one-out protocol (section 7.1, G10).

Section 7.1 names three query modes and evaluates two:

* **user-profile queries**: aggregate a user's interacted items;
* **leave-one-out**: hold one interacted item out as the target, build the
  profile from the rest;
* **item-as-query**: "part of the implementation but not evaluated in the
  present experiments".

The leave-one-out description is four sentences and leaves five decisions open,
each of which changes the reported margin distribution. They are pinned here and
recorded on the :class:`Query`, so a result traces back to the protocol that
produced it (``spec_addenda.md#g10``).

1. **Which item is held out.** Every interacted item in turn, all folds, no
   sampling. The query count is then a function of the data rather than of a
   seed.
2. **Does the held-out item stay in the corpus?** Yes. It is the retrieval
   target; removing it makes the fold unanswerable.
3. **Are the user's remaining profile items excluded from the candidate set?**
   Yes, the most consequential unstated choice in section 7.1. Those items
   contributed the query's text, so left in they score near the top, occupy the
   top-k and dominate the margin distribution with an artefact. Excluding them
   makes the measurement about retrieval rather than self-similarity.
4. **Eligibility.** Users with at least five qualifying interactions
   (:func:`~tfidf_stability.profiles.user_profile.eligible_users`).
5. **What counts as an interaction.** A weight threshold from the dataset
   configuration; for MovieLens, ``rating >= 4.0``.

Decision 3 has a knock-on the paper never mentions: the candidate set is smaller
than the corpus and differs per fold, so ``N`` differs per query and any
statistic aggregated across folds has a varying denominator.
:attr:`Query.n_candidates` carries it, so that is visible rather than assumed
constant.
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

    Attributes:
        query_id: Unique and deterministic: ``{user}::{held_out}`` for a fold.
        mode: Which construction.
        features: The query's preprocessed feature stream.
        profile: The profile it came from, or ``None`` for item-as-query.
        target: The held-out item a leave-one-out fold is trying to retrieve.
        excluded: Documents removed from the candidate set (decision 3). The
            target is absent from it and stays scoreable.
        n_candidates: Corpus size minus exclusions. Varies per fold, so it is
            carried rather than inferred.
    """

    query_id: str
    mode: QueryMode
    features: tuple[str, ...]
    profile: UserProfile | None = None
    target: str | None = None
    excluded: frozenset[str] = field(default_factory=frozenset)
    n_candidates: int = 0

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

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.queries)

    @property
    def candidate_spread(self) -> tuple[int, int, int]:
        """``(min, median, max)`` candidate count across queries (G19).

        Excluding a user's profile makes ``N`` differ from query to query, so a
        margin distribution pools populations of different sizes and a
        disagreement rate has a varying denominator. The spread makes that
        visible rather than assumed away.
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

    A `Query` travels as a feature stream, and `Profile.features` is documented
    as "Empty for the vector-space aggregations, which never build one" -- those
    carry the profile as a vector instead. Nothing in the grid layer embeds one,
    so a vector aggregation produced a query with no features, `evaluate` counted
    every one of them as a degenerate profile and skipped it, and the caller got
    an empty grid and an `n_degenerate` equal to its size. No error, no queries.

    Two of the three documented aggregations behaved that way. Raising here says
    so at the point of the choice, in the style `build_query_grid` already uses
    for the item-as-query mode it declines to run.
    """
    if aggregation is not ProfileAggregation.TEXT_CONCAT:
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
    """Every fold for every eligible user (G10).

    For each eligible user and each of their items in turn, the item becomes the
    target and the remainder becomes the profile. The target stays in the corpus
    (decision 2); the remaining profile items leave the candidate set
    (decision 3).

    Args:
        grouped: ``{user_id: (doc_id, ...)}`` from
            :func:`~tfidf_stability.profiles.user_profile.group_interactions`.
        features_by_doc: Preprocessed feature streams.
        aggregation: Profile aggregation mode (G11).
        min_interactions: Eligibility threshold (decision 4).
        exclude_profile_items: Decision 3, default ``True``. ``False`` is a
            declared ablation: it changes what the margin distribution measures.
        doc_ids: The corpus's identifiers, for the candidate count. Defaults to
            the keys of ``features_by_doc``.

    Returns:
        The :class:`QuerySet`. Query count is ``sum(len(items))`` over eligible
        users: a function of the data alone, no sampling and no seed.
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
    the present experiments", so it is provided and kept out of the reported
    results.

    ``exclude_self`` defaults to ``True``: an item is its own nearest neighbour
    at similarity 1, which tells you nothing and displaces a real result.
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
