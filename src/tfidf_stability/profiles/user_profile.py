"""User-profile documents (README section 3, ``spec_addenda.md#g11``).

Section 3: "User-specific documents are constructed from interactions such as
liked, viewed, or favourited items. These user-profile documents are embedded
using the **same vocabulary and IDF mapping as the corpus**."

Aggregation
-----------
Section 7.1 says a profile is built "by aggregating **text** from a user's
interacted items", which pins :attr:`ProfileAggregation.TEXT_CONCAT` as the
normative mode. It has a consequence the paper does not draw out, and it is not
a small one:

    concatenating token streams makes ``tf`` a **length-weighted** average, so a
    single verbose item can dominate a user's profile regardless of how many
    other items they interacted with.

That is itself a stability question -- a profile whose direction is decided by
one long document is fragile in exactly the way section 1.1 is interested in --
so :attr:`ProfileAggregation.VECTOR_MEAN` and
:attr:`ProfileAggregation.VECTOR_SUM` are implemented as ablations. They weight
each item equally in the embedding space rather than by its token count. The
default remains the paper's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from tfidf_stability.utils.numerics import Reduction, reduce_sum, sqrt
from tfidf_stability.vectorisation.sparse import SparseVector
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

__all__ = [
    "Interaction",
    "ProfileAggregation",
    "UserProfile",
    "build_profile",
    "eligible_users",
    "embed_profile",
    "group_interactions",
    "profile_norm",
]


class ProfileAggregation(str, Enum):
    """How a user's interacted items combine into one query."""

    #: Concatenate token streams, then embed. Section 7.1's wording, and the
    #: normative default. Length-weighted, per the module docstring.
    TEXT_CONCAT = "text_concat"
    #: Embed each item, then average the vectors. Equal weight per item.
    VECTOR_MEAN = "vector_mean"
    #: Embed each item, then sum. Same *direction* as the mean, so identical
    #: cosine similarities -- retained because the norm differs, and section
    #: 4.2's bounds are stated in terms of norms.
    VECTOR_SUM = "vector_sum"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class Interaction:
    """One user-item interaction.

    ``weight`` carries the rating or engagement value the interaction came with.
    It is *not* used to weight the aggregation: section 7.1 describes an
    unweighted aggregation of interacted text, and introducing a weighting the
    paper does not specify would change every published number. It is retained
    so an eligibility filter can be applied over it, and so a weighted variant
    can be added later as a declared ablation.
    """

    user_id: str
    doc_id: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class UserProfile:
    """A user's aggregated interaction history, ready to embed."""

    user_id: str
    #: The items that contributed, in a deterministic order.
    item_ids: tuple[str, ...]
    #: Concatenated feature stream. Empty for the vector-space aggregations,
    #: which never build one.
    features: tuple[str, ...]
    aggregation: ProfileAggregation

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def group_interactions(
    interactions: Iterable[Interaction],
    *,
    min_weight: float | None = None,
) -> dict[str, tuple[str, ...]]:
    """Group interactions by user, applying the eligibility filter of G10(5).

    Args:
        interactions: All interactions.
        min_weight: Keep only interactions at or above this weight. For
            MovieLens this is the ``rating >= 4.0`` threshold that decides what
            "interacted" means -- a decision section 7.1 leaves open and that
            materially changes the profile.

    Returns:
        ``{user_id: (doc_id, ...)}`` with items in **ascending identifier
        order**, not arrival order. Interaction files are not order-stable, and
        a profile built by concatenation is order-*sensitive* (the feature
        stream differs, and with n-grams so does the vocabulary), so the order
        is canonicalised here rather than inherited.
    """
    grouped: dict[str, list[str]] = {}
    for interaction in interactions:
        if min_weight is not None and interaction.weight < min_weight:
            continue
        grouped.setdefault(interaction.user_id, []).append(interaction.doc_id)
    return {
        user: tuple(sorted(set(items), key=lambda d: d.encode("utf-8")))
        for user, items in grouped.items()
    }


def eligible_users(
    grouped: Mapping[str, Sequence[str]], min_interactions: int = 5
) -> tuple[str, ...]:
    """Users with enough interactions for leave-one-out (G10(4)).

    Five is the pinned threshold. Below it, holding one item out leaves a
    profile too thin to be meaningful, and section 7.1's "users with multiple
    interactions" does not say how many. Returned in ascending identifier order
    so the evaluated query set is reproducible.
    """
    if min_interactions < 2:
        raise ValueError(
            f"leave-one-out needs at least 2 interactions per user, got "
            f"min_interactions={min_interactions}"
        )
    return tuple(
        sorted(
            (u for u, items in grouped.items() if len(items) >= min_interactions),
            key=lambda u: u.encode("utf-8"),
        )
    )


def build_profile(
    user_id: str,
    item_ids: Sequence[str],
    features_by_doc: Mapping[str, Sequence[str]],
    aggregation: ProfileAggregation = ProfileAggregation.TEXT_CONCAT,
    *,
    separate_items: bool = False,
) -> UserProfile:
    """Aggregate a user's items into a profile document.

    Items are concatenated in the order given, which
    :func:`group_interactions` has already canonicalised into ascending
    identifier byte order (``spec_addenda.md#g20``).

    What is concatenated is **feature streams, not text**
    -------------------------------------------------------
    ``features_by_doc`` holds streams that have already been through
    :meth:`PreprocessingPipeline.preprocess`, so the n-grams exist before this
    function sees them and no n-gram pass runs over the joined result. Three
    consequences follow, and each contradicts something previously written here
    or in ``spec_addenda.md#g20`` (see ``#g28``):

    * **No seam n-grams are ever produced.** Text aggregation of
      ``"quick brown fox"`` and ``"lazy sleeping dog"`` yields the bigram
      ``fox|lazi``; this function yields the union of the two streams and
      nothing spanning the join.
    * **Aggregation is therefore already order-insensitive.** Reordering the
      items permutes the feature tuple but leaves the multiset, and hence the
      embedding, bit-identical. G20's canonical ordering remains worth having --
      it makes the *tuple* reproducible, which the digest depends on -- but it
      is not what stops scores moving, because nothing was moving them.
    * **``separate_items`` is inert**, for the same reason.

    Whether to switch to genuine text aggregation is a research decision, not a
    tidy-up: it would add one bigram per item boundary, change every profile's
    ``L``, and so move every number in section 7.1.

    Args:
        separate_items: Insert a gap sentinel between consecutive items. Its
            intent is to stop n-grams spanning the seam between two documents,
            but there are no such n-grams to stop (above), and the sentinel is
            not in the vocabulary, so it is discarded before it reaches a
            count. Retained rather than removed because it becomes meaningful
            the moment aggregation moves to text; asserted inert by
            ``test_separate_items_is_inert_while_aggregation_joins_features``.

    Raises:
        KeyError: If an item has no feature stream. Silently skipping it would
            shrink the profile without record, and a profile's size is part of
            what the leave-one-out protocol reports.
    """
    missing = [d for d in item_ids if d not in features_by_doc]
    if missing:
        raise KeyError(f"no feature stream for {missing[:3]}")

    features: tuple[str, ...] = ()
    if aggregation is ProfileAggregation.TEXT_CONCAT:
        if separate_items:
            from tfidf_stability.preprocessing.tokenise import GAP

            parts: list[str] = []
            for i, doc_id in enumerate(item_ids):
                if i:
                    parts.append(GAP)
                parts.extend(features_by_doc[doc_id])
            features = tuple(parts)
        else:
            features = tuple(f for d in item_ids for f in features_by_doc[d])

    return UserProfile(
        user_id=user_id,
        item_ids=tuple(item_ids),
        features=features,
        aggregation=aggregation,
    )


def embed_profile(
    profile: UserProfile,
    model: TfidfModel,
    features_by_doc: Mapping[str, Sequence[str]] | None = None,
    policy: Reduction = Reduction.NAIVE,
) -> SparseVector:
    """Embed a profile into the corpus's TF-IDF space.

    Uses the model's own vocabulary and IDF mapping in every mode, as section 3
    requires (``spec_addenda.md#g12``) -- nothing is refitted and the vocabulary
    is never extended.

    Args:
        profile: The aggregated profile.
        model: The fitted corpus model.
        features_by_doc: Required for the vector-space aggregations, which embed
            each item separately.
        policy: Reduction policy for the vector combination.

    Returns:
        The profile's TF-IDF vector.
    """
    if profile.aggregation is ProfileAggregation.TEXT_CONCAT:
        return TfidfVectoriser.transform_query(profile.features, model)

    if features_by_doc is None:
        raise ValueError(
            f"{profile.aggregation} embeds each item separately and needs features_by_doc"
        )

    accumulated: dict[int, list[float]] = {}
    for doc_id in profile.item_ids:
        vector = TfidfVectoriser.transform_query(features_by_doc[doc_id], model)
        for index, value in zip(vector.indices, vector.values, strict=True):
            accumulated.setdefault(index, []).append(value)

    is_mean = profile.aggregation is ProfileAggregation.VECTOR_MEAN
    divisor = float(profile.n_items) if is_mean else 1.0
    if divisor == 0.0:
        return SparseVector.zero(len(model.vocabulary))

    # Summed in ascending term-identifier order, matching every other reduction
    # in this project, so the result is reproducible.
    return SparseVector.from_mapping(
        {index: reduce_sum(values, policy) / divisor for index, values in accumulated.items()},
        len(model.vocabulary),
    )


def profile_norm(vector: SparseVector, policy: Reduction = Reduction.NAIVE) -> float:
    """``||q||_2`` for a profile vector.

    Exposed because section 6 identifies low-norm vectors as the unstable
    regime, and a thin profile -- few items, few in-vocabulary tokens -- is
    exactly how a low-norm *query* arises in practice.
    """
    return sqrt(reduce_sum([v * v for v in vector.values], policy))
