"""Section 7.1's query-grid protocol: who is asked, what they may retrieve.

`analysis/query_grid.py` decides which queries the reported experiments run and,
for each one, which documents it is allowed to retrieve. Every A1 and A2 number
is computed over the grid this module builds, and until this file existed nothing
imported it: the scripts under `scripts/` were its only caller, so a change here
would have moved every published figure with the suite still green.

Four properties are load-bearing.

A per-query candidate set, not a corpus-wide one. `_score_one` restricts the
attribute table to the same subset as the scores, so the tie-break cannot reach a
document the query was never allowed to retrieve. Scores and table drifting apart
would rank a candidate against an outsider's attributes.

Degenerate and zero-vector are different findings. A query with no features at
all is counted and skipped; a query whose features are all out of vocabulary is
scored and comes back all-zero. Both produce a ranking decided by attributes
alone, and conflating them would report one cause for two.

The emptiness guard on `is_zero_vector`. `all()` over no scores is `True`, so a
query with no candidates once answered yes and entered the published
`n_zero_vector` as a pure-attribute ranking with nothing there to rank. The
module records this as fixed; this file is what keeps it fixed.

A deterministic prefix under `limit`. Two runs at different caps have to remain
comparable, so the cap takes a prefix rather than a sample.
"""

from __future__ import annotations

import pytest

from tfidf_stability.analysis.query_grid import (
    EvaluatedQuery,
    QueryGrid,
    build_query_grid,
    evaluate,
)
from tfidf_stability.profiles.query_modes import QueryMode
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

# ---------------------------------------------------------------------------
# Builders. A corpus, its records and its interactions have to agree, so they
# are produced together rather than assembled per test.
# ---------------------------------------------------------------------------
_FEATURES: dict[str, tuple[str, ...]] = {
    "d0": ("alpha", "beta"),
    "d1": ("alpha", "gamma"),
    "d2": ("beta", "gamma", "delta"),
    "d3": ("delta",),
    "d4": ("alpha", "beta", "gamma"),
    "d5": ("epsilon",),
    "d6": ("alpha", "delta"),
    "d7": ("beta", "epsilon"),
}
DOC_IDS: tuple[str, ...] = tuple(_FEATURES)


def records() -> list[dict[str, object]]:
    """Attribute records in G8's exact-pair form, one per document."""
    return [
        {
            "doc_id": doc_id,
            "popularity": 100 - i,
            "rating_sum2": 2 * (i + 3),
            "rating_count": 2,
            "engagement": i,
        }
        for i, doc_id in enumerate(DOC_IDS)
    ]


def model() -> object:
    return TfidfVectoriser().fit([list(_FEATURES[d]) for d in DOC_IDS], list(DOC_IDS))


def interactions_for(users: dict[str, list[str]]) -> list[tuple[str, str, float]]:
    return [(user, doc, 5.0) for user, docs in users.items() for doc in docs]


_USERS = {
    "u1": ["d0", "d1", "d2"],
    "u2": ["d3", "d4", "d5"],
    "u3": ["d0", "d6", "d7"],
}


def built(**kwargs: object) -> object:
    return build_query_grid(
        interactions_for(_USERS), _FEATURES, DOC_IDS, min_interactions=2, **kwargs
    )


# ---------------------------------------------------------------------------
# Normal: the protocol builds and evaluates
# ---------------------------------------------------------------------------
def test_a_leave_one_out_fold_is_produced_for_every_interaction_of_every_user() -> None:
    query_set = built(mode=QueryMode.LEAVE_ONE_OUT)
    assert len(query_set.queries) == 9, "three users with three interactions each"
    assert query_set.n_users == 3
    assert {q.target for q in query_set.queries} == set(DOC_IDS) - {"d0"} | {"d0"}


def test_the_user_profile_mode_asks_one_query_per_user_not_one_per_interaction() -> None:
    query_set = built(mode=QueryMode.USER_PROFILE)
    assert len(query_set.queries) == 3, "one profile query per eligible user"
    assert all(q.target is None for q in query_set.queries), "a profile query has no held-out item"


def test_the_restricted_table_holds_exactly_the_candidates_and_no_others() -> None:
    """Scores and table must describe the same set, or the tie-break ranks a
    candidate against an outsider's attributes."""
    grid = evaluate(built(), model(), records(), DOC_IDS)
    assert grid.queries, "the grid is empty, so nothing below is tested"

    for query in grid.queries:
        assert isinstance(query.table, AttributeTable)
        # Identity, not cardinality. A table built from the wrong documents has
        # the right length, so counts alone cannot see it.
        assert query.table.doc_ids == query.candidate_ids, (
            f"{query.query_id}: the table holds {query.table.doc_ids} but the "
            f"candidates are {query.candidate_ids}"
        )
        assert query.table.n_documents == len(query.scores), (
            "the attribute table and the score vector describe different sets"
        )
        assert len(query.candidate_ids) == len(query.scores)


def test_excluding_profile_items_removes_them_from_the_candidate_set() -> None:
    grid = evaluate(built(exclude_profile_items=True), model(), records(), DOC_IDS)
    checked = 0
    for query in grid.queries:
        assert query.n_excluded > 0, "G10 decision 3 excludes the profile, so something must go"
        assert len(query.candidate_ids) + query.n_excluded == len(DOC_IDS)
        checked += 1
    assert checked == 9, "the fold count changed without this test noticing"


def test_the_held_out_target_stays_retrievable_or_the_fold_measures_nothing() -> None:
    grid = evaluate(built(exclude_profile_items=True), model(), records(), DOC_IDS)
    for query in grid.queries:
        assert query.target is not None
        assert query.target in query.candidate_ids, (
            "the held-out item must remain a candidate; a fold that cannot retrieve "
            "its own target measures nothing"
        )


def test_not_excluding_profile_items_leaves_the_whole_corpus_retrievable() -> None:
    grid = evaluate(built(exclude_profile_items=False), model(), records(), DOC_IDS)
    for query in grid.queries:
        assert query.n_excluded == 0
        assert set(query.candidate_ids) == set(DOC_IDS)


# ---------------------------------------------------------------------------
# Erroneous: the mode section 7.1 excludes
# ---------------------------------------------------------------------------
def test_item_as_query_is_rejected_rather_than_silently_evaluated() -> None:
    """It is implemented in profiles/query_modes.py, which is exactly why the
    rejection has to be here: an available mode is easy to reach by accident."""
    with pytest.raises(ValueError, match=r"section 7\.1 excludes it"):
        built(mode=QueryMode.ITEM_AS_QUERY)


# ---------------------------------------------------------------------------
# Boundary: empty, capped, and the guard that was once wrong
# ---------------------------------------------------------------------------
def test_a_query_with_no_candidates_is_not_reported_as_a_zero_vector_ranking() -> None:
    """The bug the module's own docstring records as fixed.

    `all()` over an empty sequence is `True`, so without the emptiness guard a
    candidate-free query answers "every score is zero" and enters the published
    `n_zero_vector` as a pure-attribute ranking with nothing to rank.
    """
    empty = EvaluatedQuery(
        query_id="q",
        mode="leave_one_out",
        scores=(),
        table=AttributeTable.from_records(records()[:1]),
        candidate_ids=(),
        target=None,
        n_excluded=len(DOC_IDS),
    )
    assert empty.n_candidates == 0
    assert all(s == 0.0 for s in empty.scores), "the premise: all() over nothing is True"
    assert empty.is_zero_vector is False, "an empty candidate set is not a zero-vector ranking"


def test_a_query_scoring_zero_everywhere_is_reported_as_a_zero_vector() -> None:
    """The other side of the same guard: non-empty and genuinely all zero."""
    zeroed = EvaluatedQuery(
        query_id="q",
        mode="leave_one_out",
        scores=(0.0, 0.0, 0.0),
        table=AttributeTable.from_records(records()[:3]),
        candidate_ids=("d0", "d1", "d2"),
        target=None,
        n_excluded=0,
    )
    assert zeroed.is_zero_vector is True


def test_a_single_non_zero_score_is_enough_to_stop_being_a_zero_vector() -> None:
    scored = EvaluatedQuery(
        query_id="q",
        mode="leave_one_out",
        scores=(0.0, 1e-300, 0.0),
        table=AttributeTable.from_records(records()[:3]),
        candidate_ids=("d0", "d1", "d2"),
        target=None,
        n_excluded=0,
    )
    assert scored.is_zero_vector is False, "a subnormal score is still a score"


def test_an_out_of_vocabulary_query_is_counted_as_zero_vector_not_degenerate() -> None:
    """Two different findings that both produce an attribute-only ranking."""
    features = {**_FEATURES, "d0": ("alpha", "beta")}
    query_set = build_query_grid(
        interactions_for({"u1": ["d0", "d1"]}),
        {**features, "d1": ("nowhere_in_the_corpus",)},
        DOC_IDS,
        min_interactions=2,
    )
    grid = evaluate(query_set, model(), records(), DOC_IDS)
    assert grid.n_degenerate == 0, "the query has features, so it is not degenerate"

    # The count this test is named for. Asserting only `n_degenerate` left the
    # tally itself unchecked, so a `n_zero_vector` that never counted anything
    # would have passed here under a name promising it did.
    assert grid.n_zero_vector == 1, "the out-of-vocabulary fold is the zero-vector one"
    assert [q.is_zero_vector for q in grid.queries] == [True, False]


def test_a_cap_equal_to_the_query_count_leaves_the_set_untouched() -> None:
    uncapped = built()
    capped = built(limit=len(uncapped.queries))
    assert len(capped.queries) == len(uncapped.queries)
    assert [q.query_id for q in capped.queries] == [q.query_id for q in uncapped.queries]


def test_a_cap_takes_a_deterministic_prefix_so_two_caps_stay_comparable() -> None:
    """A sample would make runs at different caps incomparable."""
    full = [q.query_id for q in built().queries]
    for cap in (1, 2, 5):
        capped = [q.query_id for q in built(limit=cap).queries]
        assert capped == full[:cap], f"the cap at {cap} is not a prefix of the full grid"


def test_a_cap_larger_than_the_grid_is_not_an_error() -> None:
    assert len(built(limit=10_000).queries) == len(built().queries)


def test_provenance_over_an_empty_grid_reports_zero_rather_than_raising() -> None:
    """`min`/`max` over no queries need their defaults, or the manifest crashes
    on a configuration that produced nothing."""
    grid = QueryGrid(
        queries=(),
        mode="leave_one_out",
        aggregation="text_concat",
        min_interactions=5,
        exclude_profile_items=True,
        n_users=0,
        n_degenerate=0,
        n_zero_vector=0,
    )
    block = grid.provenance()
    assert block["n_candidates_min"] == 0
    assert block["n_candidates_max"] == 0
    assert block["n_queries"] == 0
    assert len(grid) == 0
    assert grid.score_vectors() == []


def test_provenance_reports_the_candidate_count_as_a_range_not_a_single_n() -> None:
    """G19: candidate sets vary per query, so one N would be a fiction."""
    grid = evaluate(built(), model(), records(), DOC_IDS)
    block = grid.provenance()
    assert block["n_queries"] == len(grid.queries) == len(grid)
    assert block["n_candidates_min"] <= block["n_candidates_max"]
    assert block["n_candidates_min"] == min(q.n_candidates for q in grid.queries)
    assert block["n_candidates_max"] == max(q.n_candidates for q in grid.queries)
    assert block["query_mode"], "the protocol must name its mode"
    assert block["aggregation"], "the protocol must name its aggregation"


def test_score_vectors_are_plain_lists_one_per_query() -> None:
    grid = evaluate(built(), model(), records(), DOC_IDS)
    vectors = grid.score_vectors()
    assert len(vectors) == len(grid.queries)
    assert all(isinstance(v, list) for v in vectors)
    assert [len(v) for v in vectors] == [q.n_candidates for q in grid.queries]


def test_an_eligibility_threshold_above_every_user_produces_an_empty_grid() -> None:
    query_set = build_query_grid(interactions_for(_USERS), _FEATURES, DOC_IDS, min_interactions=99)
    grid = evaluate(query_set, model(), records(), DOC_IDS)
    assert len(grid) == 0
    assert grid.n_users == 0
    assert grid.provenance()["n_candidates_min"] == 0


# ---------------------------------------------------------------------------
# Degenerate queries are counted, not scored
# ---------------------------------------------------------------------------
def test_a_query_whose_features_all_preprocess_away_is_counted_and_skipped() -> None:
    """Scoring it would give an all-zero vector indistinguishable from an
    out-of-vocabulary query, and those are different findings."""
    featureless = {**_FEATURES, "d1": (), "d2": ()}
    query_set = build_query_grid(
        interactions_for({"u1": ["d1", "d2"]}),
        featureless,
        DOC_IDS,
        min_interactions=2,
        exclude_profile_items=False,
    )
    grid = evaluate(query_set, model(), records(), DOC_IDS)

    assert grid.n_degenerate > 0, "a featureless profile must be counted"
    assert len(grid.queries) + grid.n_degenerate == len(query_set.queries), (
        "every query is either evaluated or counted degenerate, never both or neither"
    )
    for query in grid.queries:
        assert query.n_candidates > 0


# ---------------------------------------------------------------------------
# Stress: the whole grid, at nightly cost
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_every_fold_of_a_larger_grid_keeps_its_table_and_scores_aligned() -> None:
    users = {f"u{i}": [DOC_IDS[(i + j) % len(DOC_IDS)] for j in range(4)] for i in range(40)}
    query_set = build_query_grid(interactions_for(users), _FEATURES, DOC_IDS, min_interactions=2)
    grid = evaluate(query_set, model(), records(), DOC_IDS)

    assert len(grid) >= 100, f"only {len(grid)} folds; the sweep is too small to be worth anything"
    for query in grid.queries:
        assert query.table.n_documents == len(query.scores)
        assert len(query.candidate_ids) == len(query.scores)
        assert query.target in query.candidate_ids
    assert grid.provenance()["n_queries"] == len(grid)


# ---------------------------------------------------------------------------
# build_query_grid: the protocol boundary and the deterministic prefix
# ---------------------------------------------------------------------------
def _grid_inputs() -> tuple[list[tuple[str, str, float]], dict[str, list[str]], list[str]]:
    """Three documents and one user who has interacted with all of them.

    Local by house convention. Small enough that the prefix under `limit` can be
    written out rather than described.
    """
    features = {"d0": ["a"], "d1": ["b"], "d2": ["c"]}
    interactions = [("u1", "d0", 5.0), ("u1", "d1", 5.0), ("u1", "d2", 5.0)]
    return interactions, features, list(features)


def test_item_as_query_is_refused_here_rather_than_quietly_evaluated() -> None:
    """The mode is implemented in `profiles.query_modes`, and section 7.1
    excludes it from the reported experiments. Building a grid from it would
    produce numbers the paper does not license, so the refusal is at the
    protocol boundary rather than left to the reader."""
    interactions, features, doc_ids = _grid_inputs()
    with pytest.raises(ValueError, match=r"section 7\.1 excludes it"):
        build_query_grid(
            interactions, features, doc_ids, mode=QueryMode.ITEM_AS_QUERY, min_interactions=2
        )


def test_no_limit_at_all_keeps_every_fold() -> None:
    """`None` is how this codebase spells unlimited, everywhere."""
    interactions, features, doc_ids = _grid_inputs()
    assert len(build_query_grid(interactions, features, doc_ids, min_interactions=2)) == 3


def test_a_limit_of_zero_produces_an_empty_grid_rather_than_every_fold() -> None:
    """Zero is a number of queries, not an absence of a limit. A run configured
    with `limit: 0` should evaluate nothing and say so, not silently evaluate
    the lot."""
    interactions, features, doc_ids = _grid_inputs()
    assert len(build_query_grid(interactions, features, doc_ids, min_interactions=2, limit=0)) == 0


def test_a_negative_limit_silently_drops_the_last_fold() -> None:
    """`folds[:-1]` is a legal slice, so `-1` trims one query instead of raising
    or meaning unlimited.

    The fifth site in this package where a negative index is accepted where the
    positive side is checked, after `short`, `Ranking.top_k`, `compare_top_k`
    and `CsrMatrix.row`. Only `build_vocabulary`'s `max_features` guards it.
    """
    interactions, features, doc_ids = _grid_inputs()
    trimmed = build_query_grid(interactions, features, doc_ids, min_interactions=2, limit=-1)
    assert len(trimmed) == 2


def test_the_limited_prefix_is_the_same_folds_every_time() -> None:
    """A truncated grid has to be a prefix of the full one in a fixed order, or
    two runs at the same limit would evaluate different queries and their
    numbers would not be comparable."""
    interactions, features, doc_ids = _grid_inputs()
    full = [
        q.query_id for q in build_query_grid(interactions, features, doc_ids, min_interactions=2)
    ]
    prefix = [
        q.query_id
        for q in build_query_grid(interactions, features, doc_ids, min_interactions=2, limit=2)
    ]

    assert prefix == full[:2]
    assert prefix == ["u1::d0", "u1::d1"]


def test_a_threshold_no_user_reaches_produces_an_empty_grid() -> None:
    """A legitimate configuration -- a strict eligibility threshold on a small
    corpus -- rather than an error. The provenance still records what was asked
    for, which is how a run says it measured nothing on purpose."""
    interactions, features, doc_ids = _grid_inputs()
    grid = build_query_grid(interactions, features, doc_ids, min_interactions=99)
    assert len(grid) == 0


def test_the_profile_items_are_excluded_unless_the_caller_says_otherwise() -> None:
    """`exclude_profile_items: bool = True`. G10 decision 3 is the default, not
    an opt-in: a fold that could retrieve the items its own query was built from
    would score its own input and report the retrieval as a success.

    Both arms are covered above, but each passes the flag explicitly, so the
    default itself -- what every caller that does not think about it gets -- was
    never exercised.
    """
    default = evaluate(built(), model(), records(), DOC_IDS)
    asked_for = evaluate(built(exclude_profile_items=True), model(), records(), DOC_IDS)

    assert default.exclude_profile_items is True
    assert default.provenance()["exclude_profile_items"] is True
    assert [q.n_excluded for q in default.queries] == [q.n_excluded for q in asked_for.queries]
    assert all(q.n_excluded > 0 for q in default.queries)
