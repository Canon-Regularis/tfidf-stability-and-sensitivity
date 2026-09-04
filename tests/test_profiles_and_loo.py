"""User profiles and the leave-one-out protocol (section 7.1, G10, G11).

Section 7.1 describes leave-one-out in four sentences and leaves five decisions
open. Each is pinned in ``profiles/query_modes.py`` and tested here: each moves
the reported margin distribution, and none is recoverable from the paper.

Decision 3 (whether the user's remaining profile items stay in the candidate
set) matters most. Those documents contributed the query's text, so without the
exclusion they take the top ranks and the measurement describes self-similarity
instead of retrieval. See
:func:`test_leaving_profile_items_in_lets_them_retrieve_themselves`.
"""

from __future__ import annotations

import pytest

from tfidf_stability.profiles.query_modes import (
    Query,
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
from tfidf_stability.utils.numerics import same_bits
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
    """Interaction files are not order-stable and a concatenated profile is
    order-sensitive, so the order is canonicalised rather than inherited."""
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

    A verbose item contributes more tokens and so pulls the profile towards
    itself while being one item among several. The vector modes exist as
    ablations for it.
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

    The sum is kept because the norm differs, and sections 4.2-4.3 state their
    bounds in terms of norms.
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
    with pytest.raises(KeyError, match=r"no feature stream for \['nonexistent'\]"):
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

    The remaining profile items contributed the query's tokens, so without the
    exclusion they take the top of the ranking and the margin distribution
    describes self-similarity rather than retrieval.
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
# G19 / G20: consequences the paper does not draw out
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

    A different concatenation order changes the tokens that meet at the boundary
    between two items; with n-grams enabled that changes the feature set, hence
    df, hence every score.
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

    With a sentinel between items no n-gram can span the seam, so the profile
    becomes a function of the item set rather than of its order.
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
    """They differ by a positive scalar, so no similarity can tell them apart;
    the norms differ, and sections 4.2-4.3 bound in terms of norms."""
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


# ---------------------------------------------------------------------------
# What aggregation actually joins (spec_addenda G28)
# ---------------------------------------------------------------------------
def test_aggregation_joins_feature_streams_and_so_makes_no_seam_ngrams() -> None:
    """``build_profile`` concatenates preprocessed streams rather than text.

    G20 and this function's own docstring both used to say n-grams are generated
    over the concatenated stream, so item order changes the features produced at
    the seams. No pass runs over the joined result, so no seam n-gram is ever
    produced, and the order-sensitivity that motivated the canonical ordering
    does not exist either. Pinned so the gap between the spec and the code
    cannot close in the wrong direction.
    """
    from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    a_text, b_text = "quick brown fox", "lazy sleeping dog"
    by_doc = {"a": pipeline.preprocess(a_text), "b": pipeline.preprocess(b_text)}

    joined = build_profile("u", ["a", "b"], by_doc).features
    as_text = tuple(pipeline.preprocess(f"{a_text} {b_text}"))

    seam = set(as_text) - set(joined)
    assert seam, "text aggregation must produce a seam bigram, or this test proves nothing"
    assert seam.isdisjoint(joined), "the implementation must not produce it"
    assert len(joined) == len(by_doc["a"]) + len(by_doc["b"]), "a plain union, nothing added"


def test_item_order_cannot_move_a_profile_embedding(mini_model) -> None:  # type: ignore[no-untyped-def]
    """The corollary: reordering permutes the tuple and nothing else."""
    from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    by_doc = {
        "a": pipeline.preprocess("quick brown fox"),
        "b": pipeline.preprocess("lazy sleeping dog"),
    }
    forward = build_profile("u", ["a", "b"], by_doc)
    backward = build_profile("u", ["b", "a"], by_doc)

    assert forward.features != backward.features, "the tuple order does differ"
    assert sorted(forward.features) == sorted(backward.features), "but only in order"

    one = embed_profile(forward, mini_model)
    other = embed_profile(backward, mini_model)
    assert one.indices == other.indices
    assert all(same_bits(x, y) for x, y in zip(one.values, other.values, strict=True))


def test_separate_items_is_inert_while_aggregation_joins_features(mini_model) -> None:  # type: ignore[no-untyped-def]
    """The advertised G20 ablation cannot block a seam that is never formed."""
    from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline
    from tfidf_stability.preprocessing.tokenise import GAP

    pipeline = PreprocessingPipeline()
    by_doc = {
        "a": pipeline.preprocess("quick brown fox"),
        "b": pipeline.preprocess("lazy sleeping dog"),
    }
    plain = build_profile("u", ["a", "b"], by_doc, separate_items=False)
    separated = build_profile("u", ["a", "b"], by_doc, separate_items=True)

    assert GAP in separated.features, "the sentinel is inserted"
    assert GAP not in plain.features
    assert len(separated.features) == len(plain.features) + 1

    # ...and is then discarded, so the embedding is untouched.
    one = embed_profile(plain, mini_model)
    other = embed_profile(separated, mini_model)
    assert one.indices == other.indices
    assert all(same_bits(x, y) for x, y in zip(one.values, other.values, strict=True))


def test_an_empty_query_set_reports_a_zero_spread_rather_than_raising() -> None:
    """A protocol that eliminated every user -- a minimum-interaction threshold
    above what anyone reached -- is a legitimate configuration whose provenance
    still has to be written to the manifest. The alternative is a run that
    crashes while recording why it had nothing to run.
    """
    grouped = {"u_a": ("m1", "m2"), "u_b": ("m3", "m4")}
    qs = leave_one_out_queries(grouped, FEATURES, min_interactions=99, doc_ids=DOC_IDS)
    assert len(qs) == 0
    assert qs.candidate_spread == (0, 0, 0)

    p = qs.provenance()
    assert (p["candidates_min"], p["candidates_median"], p["candidates_max"]) == (0, 0, 0)


def test_a_profile_with_no_items_embeds_to_the_zero_vector() -> None:
    """The mean over no items has no divisor. The zero vector is the honest
    answer and a legitimate one downstream: section 6 identifies low-norm
    queries as the unstable regime, and this is its limit, where the ranking
    falls entirely to the tie-break.
    """
    m = model()
    empty = build_profile("u", (), FEATURES, ProfileAggregation.VECTOR_MEAN)
    assert empty.n_items == 0

    vector = embed_profile(empty, m, FEATURES)
    assert vector.dim == len(m.vocabulary)
    assert vector.values == () or all(v == 0.0 for v in vector.values)
    assert profile_norm(vector) == 0.0


def test_an_empty_profile_summed_rather_than_averaged_is_also_zero() -> None:
    """The sum path has a divisor of one, so it reaches the same place by a
    different route; the two aggregations must not disagree about nothing."""
    m = model()
    empty = build_profile("u", (), FEATURES, ProfileAggregation.VECTOR_SUM)
    assert profile_norm(embed_profile(empty, m, FEATURES)) == 0.0


# ---------------------------------------------------------------------------
# The candidate count, and the defaults that decide it
# ---------------------------------------------------------------------------
def test_the_candidate_count_is_the_corpus_less_the_exclusions() -> None:
    """G19's whole point: N varies per query, so a margin distribution pools
    populations of different sizes. Adding the exclusions instead of subtracting
    them gives a plausible number that is wrong in the direction that makes
    every query look better conditioned than it is.
    """
    grouped = group_interactions(interactions(), min_weight=4.0)
    qs = user_profile_queries(grouped, FEATURES, doc_ids=DOC_IDS)
    assert len(qs) >= 2

    for q in qs:
        assert q.n_candidates == len(DOC_IDS) - len(q.excluded)
        assert q.n_candidates < len(DOC_IDS), "the profile items really were removed"


def test_an_item_query_scores_every_document_but_itself() -> None:
    q = item_as_query("m1", FEATURES, doc_ids=DOC_IDS)
    assert q.n_candidates == len(DOC_IDS) - 1

    kept = item_as_query("m1", FEATURES, exclude_self=False, doc_ids=DOC_IDS)
    assert kept.n_candidates == len(DOC_IDS)


def test_the_candidate_spread_is_the_min_median_and_max_of_the_counts() -> None:
    """Asserted as the values rather than as `low <= median <= high`, which
    holds for a great many wrong triples -- including one that reports the
    second-smallest count as the minimum."""
    grouped = {"few": ("m1", "m2"), "many": ("m1", "m2", "m3", "m4", "m5")}
    qs = user_profile_queries(grouped, FEATURES, doc_ids=DOC_IDS)
    counts = sorted(q.n_candidates for q in qs)
    assert counts == [len(DOC_IDS) - 5, len(DOC_IDS) - 2], "two different-sized profiles"
    assert qs.candidate_spread == (counts[0], counts[len(counts) // 2], counts[-1])
    assert qs.candidate_spread[0] == 2


def test_a_query_carries_no_candidate_count_until_one_is_supplied() -> None:
    """It is carried rather than inferred, so the unset value has to be
    distinguishable from a corpus of one document."""
    bare = Query(query_id="q", mode=QueryMode.ITEM_AS_QUERY, features=("a",))
    assert bare.n_candidates == 0
    assert bare.excluded == frozenset()


# ---------------------------------------------------------------------------
# The defaults, which are part of the protocol and reach the manifest
# ---------------------------------------------------------------------------
def test_profile_items_are_excluded_unless_the_protocol_says_otherwise() -> None:
    """The default is the decision G10 records; flipping it lets every user's
    own items retrieve themselves and inflates every retrieval number."""
    grouped = {"u": ("m1", "m2")}
    default = user_profile_queries(grouped, FEATURES, doc_ids=DOC_IDS)
    assert default.exclude_profile_items is True
    assert next(iter(default)).excluded == frozenset({"m1", "m2"})

    kept = user_profile_queries(grouped, FEATURES, exclude_profile_items=False, doc_ids=DOC_IDS)
    assert next(iter(kept)).excluded == frozenset()


def test_one_interaction_is_enough_for_a_profile_query_by_default() -> None:
    """Unlike a leave-one-out fold, which needs two. Raising the default to two
    would silently drop every single-interaction user from the query set, and
    the count is only visible in the provenance.
    """
    grouped = {"thin": ("m1",), "thick": ("m1", "m2", "m3")}
    qs = user_profile_queries(grouped, FEATURES, doc_ids=DOC_IDS)
    assert {q.query_id for q in qs} == {"thin", "thick"}
    assert qs.min_interactions == 1
    assert qs.provenance()["min_interactions"] == 1

    # And the threshold is honoured when it is raised.
    strict = user_profile_queries(grouped, FEATURES, min_interactions=2, doc_ids=DOC_IDS)
    assert {q.query_id for q in strict} == {"thick"}


def test_a_nan_threshold_keeps_everything_exactly_as_no_threshold_does() -> None:
    """`weight < min_weight` is false for every weight when the threshold is
    NaN, so a NaN filters nothing.

    Four guards in this package spell the same shape `not (x >= 0)` precisely so
    a NaN is refused rather than ignored. This one does not, and should not: it
    has no non-negative domain, a negative threshold is a legal way to keep
    everything, and no caller constructs a NaN. Written down because the
    difference from those four is deliberate and otherwise looks like an
    oversight.
    """
    interactions = [
        Interaction(user_id="u", doc_id="m1", weight=5.0),
        Interaction(user_id="u", doc_id="m2", weight=1.0),
    ]
    everything = group_interactions(interactions)
    assert group_interactions(interactions, min_weight=float("nan")) == everything
    assert group_interactions(interactions, min_weight=-1.0) == everything

    # Contrast: a real threshold does filter, so the equality above is not
    # simply the filter never running.
    assert group_interactions(interactions, min_weight=4.0) == {"u": ("m1",)}


def test_an_interaction_exactly_at_the_threshold_counts_as_one() -> None:
    """G10 decision 5 sets the threshold as `rating >= 4.0`, so the filter drops
    what is strictly below it. At exactly 4.0 the interaction is kept.

    The boundary is the whole decision: on MovieLens the 4.0 rating is one of
    the most common values, so moving this comparison by one notch changes the
    size of nearly every profile and therefore every query built from one.
    """
    at_threshold = [
        Interaction("u", "m1", 4.0),
        Interaction("u", "m2", 3.5),
        Interaction("u", "m3", 4.5),
    ]
    grouped = group_interactions(at_threshold, min_weight=4.0)
    assert grouped == {"u": ("m1", "m3")}, "4.0 is in, 3.5 is out"


def test_an_interaction_weighs_one_unless_the_file_says_otherwise() -> None:
    """Uniform weighting is the normative choice, and the default is what makes
    an interaction file without a weight column mean that. A default of zero
    would be filtered out by any positive threshold, emptying every profile."""
    assert Interaction("u", "m1").weight == 1.0
    assert group_interactions([Interaction("u", "m1")], min_weight=1.0) == {"u": ("m1",)}


def test_no_threshold_at_all_keeps_every_interaction() -> None:
    """`min_weight=None` is the documented way to take a file as written."""
    every = [Interaction("u", "m1", 0.5), Interaction("u", "m2", 5.0)]
    assert group_interactions(every) == {"u": ("m1", "m2")}
    assert group_interactions(every, min_weight=None) == {"u": ("m1", "m2")}


# ---------------------------------------------------------------------------
# The grid carries features, so it must refuse an aggregation that has none
# ---------------------------------------------------------------------------
# `Profile.features` is documented as "Empty for the vector-space aggregations,
# which never build one" -- those carry the profile as a vector, and nothing in
# the grid layer embeds one. So `vector_mean` and `vector_sum` produced queries
# with no features, `evaluate` counted every one as a degenerate profile and
# skipped it, and the caller received an empty grid with `n_degenerate` equal to
# its own size. No exception, no queries, and two of the three documented
# aggregations behaved that way.
def test_a_vector_space_aggregation_is_refused_rather_than_silently_emptied() -> None:
    """Both builders, both vector aggregations, and the error names the fix.

    Refused at the point of the choice rather than discovered as an empty result
    later, in the style `build_query_grid` already uses for the item-as-query
    mode it declines to run.
    """
    grouped = group_interactions(interactions(), min_weight=4.0)
    vector_space = [ProfileAggregation.VECTOR_MEAN, ProfileAggregation.VECTOR_SUM]

    checked = 0
    for aggregation in vector_space:
        for builder in (leave_one_out_queries, user_profile_queries):
            with pytest.raises(ValueError, match="profile vector rather than a feature stream"):
                builder(
                    grouped,
                    FEATURES,
                    min_interactions=5,
                    doc_ids=DOC_IDS,
                    aggregation=aggregation,
                )
            checked += 1

    assert checked == 4, "two aggregations across two builders"


def test_the_text_concatenation_aggregation_is_still_accepted() -> None:
    """The guard's other half: it must not refuse the normative default.

    Section 7.1's wording is the concatenation, and it is what every published
    number uses, so a guard rejecting it would take the whole grid with it.
    """
    grouped = group_interactions(interactions(), min_weight=4.0)

    queries = leave_one_out_queries(
        grouped,
        FEATURES,
        min_interactions=5,
        doc_ids=DOC_IDS,
        aggregation=ProfileAggregation.TEXT_CONCAT,
    )

    assert queries.queries, "the default builds a non-empty grid"
    assert all(q.features for q in queries.queries), (
        "and every query in it carries features, which is what the grid transports"
    )
