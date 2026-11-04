"""User-profile documents (README section 3, ``spec_addenda.md#g11``).

Section 7.1 makes :attr:`ProfileAggregation.TEXT_CONCAT` normative (length-weighted);
:attr:`ProfileAggregation.VECTOR_MEAN` and :attr:`ProfileAggregation.VECTOR_SUM` weight
items equally as ablations. Rationale: docs/engineering_notes.md#profile-aggregation
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from tfidf_stability.preprocessing.tokenise import GAP
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
    #: Embed each item, then sum. Same direction as the mean, so cosine
    #: similarities are identical; kept because the norm differs and section
    #: 4.2's bounds are stated in terms of norms.
    VECTOR_SUM = "vector_sum"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Interaction:
    """One user-item interaction.

    ``weight`` holds the rating or engagement value. Only the eligibility filter reads
    it: section 7.1 specifies an unweighted aggregation, and a weighting would move
    every published number.
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

    def __post_init__(self) -> None:
        """Normalise the aggregation to the member, refusing an unknown value.

        ``"vector_mean"`` equals the ``str`` enum member without being it, and
        :func:`embed_profile` selects the divisor with ``is``; the string would sum
        instead of averaging and move the norm that section 4.2's bounds use.
        """
        object.__setattr__(self, "aggregation", ProfileAggregation(self.aggregation))

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def group_interactions(
    interactions: Iterable[Interaction],
    *,
    min_weight: float | None = None,
) -> dict[str, tuple[str, ...]]:
    """Group interactions by user, applying the eligibility filter of G10(5).

    Drops interactions with ``weight < min_weight``; ``None`` or a NaN threshold keeps all,
    and a negative one is legal. Returns ``{user_id: (doc_id, ...)}``, deduplicated, items in
    ascending UTF-8 byte order (G20). Rationale: docs/engineering_notes.md#profile-eligibility
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

    ``min_interactions`` defaults to the pinned five: section 7.1 gives no count, and
    fewer items leave a held-out profile too thin. Ascending identifier byte order keeps
    the evaluated query set reproducible.
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

    Joins :meth:`PreprocessingPipeline.preprocess` streams in the order :func:`group_interactions`
    gives (``spec_addenda.md#g20``); no seam n-gram forms, so ``separate_items`` is inert (G28,
    amending ``spec_addenda.md#g20``). Rationale: docs/engineering_notes.md#profile-streams
    """
    missing = [d for d in item_ids if d not in features_by_doc]
    if missing:
        raise KeyError(f"no feature stream for {missing[:3]}")

    # Coerced before the branch below, not left to `UserProfile.__post_init__`: a string
    # spelling of TEXT_CONCAT fails the `is` test and yields an empty feature stream,
    # which the later coercion would then hide.
    aggregation = ProfileAggregation(aggregation)

    features: tuple[str, ...] = ()
    if aggregation is ProfileAggregation.TEXT_CONCAT:
        if separate_items:
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

    Uses the model's vocabulary and IDF in every mode, never refitted or extended
    (section 3, ``spec_addenda.md#g12``). The vector aggregations raise ``ValueError`` without
    ``features_by_doc`` and combine the item vectors under the reduction ``policy``.
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

    # Each term's values are summed in `item_ids` order, and `from_mapping` sorts the
    # indices ascending, so the result is reproducible.
    return SparseVector.from_mapping(
        {index: reduce_sum(values, policy) / divisor for index, values in accumulated.items()},
        len(model.vocabulary),
    )


def profile_norm(vector: SparseVector, policy: Reduction = Reduction.NAIVE) -> float:
    """``||q||_2`` for a profile vector.

    Exposed because section 6 identifies low-norm vectors as the unstable
    regime, and a thin profile (few items, few in-vocabulary tokens) is how a
    low-norm query arises in practice.
    """
    return sqrt(reduce_sum([v * v for v in vector.values], policy))
