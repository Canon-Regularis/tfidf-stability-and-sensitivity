"""User profiles and the leave-one-out protocol (section 7.1, G10, G11).

Section 7.1 describes leave-one-out in four sentences and leaves five decisions
open. Each is pinned in ``profiles/query_modes.py`` and tested here, because
each changes the reported margin distribution and none of them is recoverable
from the paper.

The one that matters most is decision 3 -- whether the user's *remaining*
profile items stay in the candidate set.
:func:`test_leaving_profile_items_in_lets_them_retrieve_themselves` demonstrates
what goes wrong without it: those documents contributed the query's text, so they
occupy the top ranks by construction and the measurement stops being about
retrieval at all.
"""

from __future__ import annotations

import pytest

from tfidf_stability.profiles.query_modes import (
    QueryMode,
    item_as_query,
    leave_one_out_queries,
    user_profile_queries,
)
from tfidf_stability.profiles.user_profile import (
    Interaction,
    ProfileAggregation,
    build_profile,
    eligible_users,
    embed_profile,
    group_interactions,
    profile_norm,
)
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

FEATURES = {
    "m1": ("space", "opera", "robot"),
    "m2": ("space", "robot", "laser"),
    "m3": ("romance", "paris", "letter"),
    "m4": ("romance", "letter", "rain"),
    "m5": ("cooking", "bread", "flour"),
    "m6": ("cooking", "flour", "yeast"),
    "m7": ("space", "laser", "comet"),
}
DOC_IDS = tuple(sorted(FEATURES))


def model():  # type: ignore[no-untyped-def]
    return TfidfVectoriser().fit([list(FEATURES[d]) for d in DOC_IDS], list(DOC_IDS))


def interactions() -> list[Interaction]:
    """One sci-fi user, one romance user, one user below the threshold."""
    return [
        *(Interaction("u_scifi", d, 5.0) for d in ("m1", "m2", "m7")),
        *(Interaction("u_scifi", d, 4.5) for d in ("m5", "m6")),
        *(Interaction("u_romance", d, 5.0) for d in ("m3", "m4")),
        Interaction("u_thin", "m1", 5.0),
        Interaction("u_lowrated", "m1", 1.0),
        Interaction("u_lowrated", "m2", 2.0),
    ]


# ---------------------------------------------------------------------------
# Grouping and eligibility (G10 decisions 4 and 5)
# ---------------------------------------------------------------------------
def test_grouping_is_canonically_ordered_not_arrival_ordered() -> None:
    """Interaction files are not order-stable, and a concatenated profile is
    order-*sensitive*, so the order is canonicalised rather than inherited."""
    forward = group_interactions(interactions())
    backward = group_interactions(list(reversed(interactions())))
    assert forward == backward
    assert forward["u_scifi"] == tuple(sorted(forward["u_scifi"]))


def test_the_weight_threshold_decides_what_interacted_means() -> None:
    """G10(5). For MovieLens this is ``rating >= 4.0``."""
    unfiltered = group_interactions(interactions())
    filtered = group_interactions(interactions(), min_weight=4.0)
    assert "u_lowrated" in unfiltered
    assert "u_lowrated" not in filtered, "both its ratings are below the threshold"


def test_eligibility_threshold_excludes_thin_users() -> None:
    """G10(4): five interactions."""
    grouped = group_interactions(interactions(), min_weight=4.0)
    assert eligible_users(grouped, 5) == ("u_scifi",)
    assert set(eligible_users(grouped, 2)) == {"u_romance", "u_scifi"}


def test_eligible_users_is_deterministically_ordered() -> None:
    grouped = group_interactions(interactions(), min_weight=4.0)
    assert eligible_users(grouped, 2) == tuple(sorted(eligible_users(grouped, 2)))


def test_leave_one_out_needs_at_least_two_interactions() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        eligible_users({"u": ("a",)}, 1)


# ---------------------------------------------------------------------------
# Profile aggregation (G11)
# ---------------------------------------------------------------------------
def test_text_concat_is_length_weighted() -> None:
    """The consequence section 7.1 does not draw out.

    A verbose item contributes more tokens, so it pulls the profile towards
    itself even though it is one item among several. That is a real fragility,
    which is why the vector modes exist as ablations.
    """
    features = {"short": ("a",), "long": tuple(["b"] * 50)}
    profile = build_profile("u", ("short", "long"), features)
    assert profile.features.count("b") == 50
    assert profile.features.count("a") == 1


def test_vector_mean_weights_items_equally() -> None:
    """The ablation: each item contributes the same regardless of length."""
    m = model()
    equal = build_profile("u", ("m1", "m5"), FEATURES, ProfileAggregation.VECTOR_MEAN)
    vector = embed_profile(equal, m, FEATURES)
    assert vector.nnz > 0
    assert vector.is_canonical()


def test_vector_sum_and_mean_give_the_same_similarities() -> None:
    """They differ by a positive scalar, and cosine is scale-invariant.

    The sum is kept anyway because the *norm* differs, and sections 4.2-4.3
    state their bounds in terms of norms.
    """
    m = model()
    docs = [m.document(i) for i in range(m.n_documents)]
    scores = {}
    for aggregation in (ProfileAggregation.VECTOR_MEAN, ProfileAggregation.VECTOR_SUM):
        profile = build_profile("u", ("m1", "m2"), FEATURES, aggregation)
        vector = embed_profile(profile, m, FEATURES)
        scores[aggregation] = cosine_against_corpus(vector, docs, m.norms)

    mean, total = scores[ProfileAggregation.VECTOR_MEAN], scores[ProfileAggregation.VECTOR_SUM]
    assert mean == pytest.approx(total, abs=1e-12)

    n_mean = profile_norm(
        embed_profile(
            build_profile("u", ("m1", "m2"), FEATURES, ProfileAggregation.VECTOR_MEAN), m, FEATURES
        )
    )
    n_sum = profile_norm(
        embed_profile(
            build_profile("u", ("m1", "m2"), FEATURES, ProfileAggregation.VECTOR_SUM), m, FEATURES
        )
    )
    assert n_sum > n_mean, "the norms differ, which is why both are kept"


def test_vector_aggregation_requires_the_feature_map() -> None:
    profile = build_profile("u", ("m1",), FEATURES, ProfileAggregation.VECTOR_MEAN)
    with pytest.raises(ValueError, match="features_by_doc"):
        embed_profile(profile, model())


def test_a_missing_feature_stream_is_an_error_not_a_silent_skip() -> None:
    """Profile size is reported, so a silently shrunk profile would misreport it."""
    with pytest.raises(KeyError):
        build_profile("u", ("m1", "nonexistent"), FEATURES)


def test_profiles_embed_with_the_corpus_vocabulary(  # G12
) -> None:
    """Section 3: same vocabulary, same IDF mapping, no refitting."""
    m = model()
    profile = build_profile("u", ("m1", "m2"), FEATURES)
    vector = embed_profile(profile, m)
    assert vector.dim == len(m.vocabulary)
    assert all(0 <= i < len(m.vocabulary) for i in vector.indices)


# ---------------------------------------------------------------------------
# Leave-one-out (G10 decisions 1, 2, 3)
# ---------------------------------------------------------------------------
def test_every_item_is_held_out_in_turn() -> None:
    """Decision 1: all folds, no sampling, so the count is a function of data."""
    grouped = group_interactions(interactions(), min_weight=4.0)
    qs = leave_one_out_queries(grouped, FEATURES, min_interactions=5, doc_ids=DOC_IDS)
    assert len(qs) == len(grouped["u_scifi"]) == 5
    assert {q.target for q in qs} == set(grouped["u_scifi"])


def test_the_held_out_item_stays_in_the_candidate_set() -> None:
    """Decision 2: it is the retrieval target, so removing it would make the
    fold unanswerable."""
    grouped = group_interactions(interactions(), min_weight=4.0)
    for q in leave_one_out_queries(grouped, FEATURES, doc_ids=DOC_IDS):
        assert q.target is not None
        assert q.target not in q.excluded
        assert q.target in [DOC_IDS[i] for i in q.candidate_indices(DOC_IDS)]


def test_the_remaining_profile_items_are_excluded() -> None:
    """Decision 3, the most consequential of the five."""
    grouped = group_interactions(interactions(), min_weight=4.0)
    for q in leave_one_out_queries(grouped, FEATURES, doc_ids=DOC_IDS):
        assert q.profile is not None
        assert q.excluded == frozenset(q.profile.item_ids)
        assert q.target not in q.excluded


def test_leaving_profile_items_in_lets_them_retrieve_themselves() -> None:
    """Why decision 3 is not a detail.

    The remaining profile items literally contributed the query's tokens, so
    without the exclusion they occupy the top of the ranking by construction --
    and the margin distribution then describes self-similarity rather than
    retrieval.
    """
    m = model()
    docs = [m.document(i) for i in range(m.n_documents)]
    grouped = {"u": ("m1", "m2", "m7")}

    included = leave_one_out_queries(
        grouped, FEATURES, min_interactions=2, exclude_profile_items=False, doc_ids=DOC_IDS
    )
    fold = next(q for q in included if q.target == "m7")
    vector = TfidfVectoriser.transform_query(fold.features, m)
    scores = dict(zip(DOC_IDS, cosine_against_corpus(vector, docs, m.norms), strict=True))

    top_two = sorted(scores, key=lambda d: -scores[d])[:2]
    assert set(top_two) == {"m1", "m2"}, "the profile's own items win"

    excluded = leave_one_out_queries(grouped, FEATURES, min_interactions=2, doc_ids=DOC_IDS)
    fold = next(q for q in excluded if q.target == "m7")
    survivors = [DOC_IDS[i] for i in fold.candidate_indices(DOC_IDS)]
    assert "m1" not in survivors
    assert "m2" not in survivors
    assert "m7" in survivors, "the target must remain retrievable"


def test_candidate_count_varies_per_fold_and_is_recorded() -> None:
    """The knock-on the paper never mentions: N differs per query, so any
    statistic aggregated across folds has a varying denominator."""
    grouped = {"u_a": ("m1", "m2", "m3"), "u_b": ("m1", "m2", "m3", "m4", "m5")}
    qs = leave_one_out_queries(grouped, FEATURES, min_interactions=3, doc_ids=DOC_IDS)
    counts = {q.n_candidates for q in qs}
    assert len(counts) > 1, "different profile sizes give different candidate sets"
    for q in qs:
        assert q.n_candidates == len(DOC_IDS) - len(q.excluded)


def test_fold_ids_are_unique_and_deterministic() -> None:
    grouped = group_interactions(interactions(), min_weight=4.0)
    a = leave_one_out_queries(grouped, FEATURES, doc_ids=DOC_IDS)
    b = leave_one_out_queries(grouped, FEATURES, doc_ids=DOC_IDS)
    ids = [q.query_id for q in a]
    assert len(set(ids)) == len(ids)
    assert ids == [q.query_id for q in b]


def test_the_profile_shrinks_by_exactly_one_item() -> None:
    grouped = group_interactions(interactions(), min_weight=4.0)
    for q in leave_one_out_queries(grouped, FEATURES, doc_ids=DOC_IDS):
        assert q.profile is not None
        assert q.profile.n_items == len(grouped["u_scifi"]) - 1
        assert q.target not in q.profile.item_ids


# ---------------------------------------------------------------------------
# The other modes
# ---------------------------------------------------------------------------
def test_user_profile_queries_have_no_target() -> None:
    grouped = group_interactions(interactions(), min_weight=4.0)
    qs = user_profile_queries(grouped, FEATURES, min_interactions=2, doc_ids=DOC_IDS)
    assert qs.mode is QueryMode.USER_PROFILE
    assert all(q.target is None for q in qs)
    assert len(qs) == 2


def test_item_as_query_excludes_itself_by_default() -> None:
    """An item is its own nearest neighbour at similarity 1, which displaces a
    real result and tells you nothing."""
    q = item_as_query("m1", FEATURES, doc_ids=DOC_IDS)
    assert q.mode is QueryMode.ITEM_AS_QUERY
    assert q.excluded == frozenset({"m1"})
    assert "m1" not in [DOC_IDS[i] for i in q.candidate_indices(DOC_IDS)]
    kept = item_as_query("m1", FEATURES, exclude_self=False, doc_ids=DOC_IDS)
    assert kept.excluded == frozenset()


# ---------------------------------------------------------------------------
# Provenance (G14)
# ---------------------------------------------------------------------------
def test_the_query_set_records_its_protocol() -> None:
    """Section 7.1 defers the counts to "the dataset configuration"; G14 fixes
    them into the manifest."""
    grouped = group_interactions(interactions(), min_weight=4.0)
    qs = leave_one_out_queries(grouped, FEATURES, min_interactions=5, doc_ids=DOC_IDS)
    p = qs.provenance()
    assert p["mode"] == "leave_one_out"
    assert p["aggregation"] == "text_concat"
    assert p["min_interactions"] == 5
    assert p["exclude_profile_items"] is True
    assert p["n_users"] == 1
    assert p["n_queries"] == 5


def test_a_degenerate_query_is_flagged() -> None:
    empty = {"e": ()}
    q = leave_one_out_queries(
        {"u": ("e", "m1", "m2")}, {**FEATURES, **empty}, min_interactions=2, doc_ids=DOC_IDS
    )
    fold = next(x for x in q if x.target == "m1")
    assert fold.is_degenerate is False
    lone = leave_one_out_queries({"u": ("e", "e2")}, {"e": (), "e2": ()}, min_interactions=2)
    assert all(x.is_degenerate for x in lone)


# ---------------------------------------------------------------------------
# G19 / G20 -- consequences the paper does not draw out
# ---------------------------------------------------------------------------
def test_the_candidate_spread_is_recorded(  # G19
) -> None:
    """Excluding the profile makes N vary per query, so the spread is reported."""
    grouped = {"u_a": ("m1", "m2", "m3"), "u_b": ("m1", "m2", "m3", "m4", "m5")}
    qs = leave_one_out_queries(grouped, FEATURES, min_interactions=3, doc_ids=DOC_IDS)
    low, median, high = qs.candidate_spread
    assert low < high, "different profile sizes give different candidate counts"
    assert low <= median <= high
    p = qs.provenance()
    assert (p["candidates_min"], p["candidates_max"]) == (low, high)


def test_concatenation_is_order_sensitive_at_the_seams(  # G20
) -> None:
    """Why the item order has to be canonicalised.

    Concatenating in a different order changes the tokens that meet at the
    boundary between two items -- and with n-grams enabled that changes the
    feature set, hence df, hence every score.
    """
    features = {"a": ("x", "y"), "b": ("p", "q")}
    forward = build_profile("u", ("a", "b"), features).features
    backward = build_profile("u", ("b", "a"), features).features
    assert forward != backward
    assert forward == ("x", "y", "p", "q")
    assert backward == ("p", "q", "x", "y")


def test_the_gap_sentinel_ablation_blocks_seam_ngrams(  # G20
) -> None:
    """The available fix, offered as an ablation rather than adopted silently.

    With a sentinel between items, no n-gram can span the seam, so the profile
    becomes a function of the item *set* rather than of its order.
    """
    from tfidf_stability.preprocessing.ngrams import generate_ngrams
    from tfidf_stability.preprocessing.tokenise import GAP

    features = {"a": ("x", "y"), "b": ("p", "q")}
    joined = build_profile("u", ("a", "b"), features, separate_items=True).features
    assert GAP in joined

    bigrams = generate_ngrams(list(joined), 2, 2)
    assert not any(b.startswith("y") and b.endswith("p") for b in bigrams), (
        "no bigram may bridge the seam between two items"
    )
    plain = generate_ngrams(list(build_profile("u", ("a", "b"), features).features), 2, 2)
    assert any("y" in b and "p" in b for b in plain), "without the sentinel, one does"


def test_vector_sum_and_mean_differ_only_in_norm(  # G21
) -> None:
    """They differ by a positive scalar, so no similarity can tell them apart --
    but the norms differ, and sections 4.2-4.3 bound in terms of norms."""
    m = model()
    vectors = {
        aggregation: embed_profile(
            build_profile("u", ("m1", "m2", "m7"), FEATURES, aggregation), m, FEATURES
        )
        for aggregation in (ProfileAggregation.VECTOR_MEAN, ProfileAggregation.VECTOR_SUM)
    }
    mean_v = vectors[ProfileAggregation.VECTOR_MEAN]
    sum_v = vectors[ProfileAggregation.VECTOR_SUM]
    assert mean_v.indices == sum_v.indices
    ratios = [s / mv for s, mv in zip(sum_v.values, mean_v.values, strict=True)]
    assert all(r == pytest.approx(3.0) for r in ratios), "a single positive scalar"
    assert profile_norm(sum_v) == pytest.approx(3.0 * profile_norm(mean_v))
