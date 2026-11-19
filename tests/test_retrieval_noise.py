"""BM25 text-query scoring, the near-tie census and noise between two grids.

The corpus holds an exact duplicate pair, so one query's top result is a tie that a
single ulp of score difference is enough to break.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from tfidf_stability.analysis.query_grid import QueryGrid
from tfidf_stability.evaluation.retrieval import (
    NEAR_TIE_TAUS,
    evaluate_retrieval,
    policy_noise,
    score_text_queries,
)
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline, preprocess_records
from tfidf_stability.profiles import text_queries
from tfidf_stability.utils.numerics import Reduction, reduce_sum, same_bits
from tfidf_stability.vectorisation.bm25 import Bm25Model, Bm25Vectoriser

_RECORDS = [
    {"doc_id": "d1", "text": "apple banana"},
    {"doc_id": "d2", "text": "apple banana"},
    {"doc_id": "d3", "text": "apple cherry cherry"},
    {"doc_id": "d4", "text": "durian"},
]
_IDS = [r["doc_id"] for r in _RECORDS]
_QUERIES = {"q1": "apple banana", "q2": "cherry apple", "q3": "zebra"}
_QRELS = {"q1": {"d2": 1}, "q2": {"d3": 2}, "q3": {"d4": 1}}


def _bm25(policy: Reduction = Reduction.NAIVE) -> tuple[Bm25Model, QueryGrid]:
    pipeline = PreprocessingPipeline()
    model = Bm25Vectoriser(reduction=policy).fit(preprocess_records(pipeline, _RECORDS), _IDS)
    queries = text_queries(_QUERIES, pipeline, n_documents=model.n_documents)
    return model, score_text_queries(queries, model, _RECORDS, _IDS)


def _bump(grid: QueryGrid, query: int, doc: int) -> QueryGrid:
    """``grid`` with one score raised by a single ulp."""
    scored = grid.queries[query]
    scores = list(scored.scores)
    scores[doc] = math.nextafter(scores[doc], math.inf)
    changed = dataclasses.replace(scored, scores=tuple(scores))
    queries = (*grid.queries[:query], changed, *grid.queries[query + 1 :])
    return dataclasses.replace(grid, queries=queries)


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(Reduction))
def test_a_bm25_score_is_the_plain_sum_of_query_count_times_weight(policy: Reduction) -> None:
    model, grid = _bm25(policy)
    pipeline = PreprocessingPipeline()

    for scored in grid.queries:
        query = Bm25Vectoriser.transform_query(
            pipeline.preprocess(_QUERIES[scored.query_id]), model
        )
        weights = dict(zip(query.indices, query.values, strict=True))
        for i, score in enumerate(scored.scores):
            row = model.document(i)
            products = [
                weights[t] * w for t, w in zip(row.indices, row.values, strict=True) if t in weights
            ]
            assert same_bits(score, reduce_sum(products, policy) if products else 0.0)


@pytest.mark.parametrize("policy", list(Reduction))
def test_swapping_the_policy_on_one_fit_equals_a_refit(policy: Reduction) -> None:
    naive, _ = _bm25(Reduction.NAIVE)
    refit, _ = _bm25(policy)

    assert dataclasses.replace(naive, reduction=policy) == refit


def test_the_duplicate_pair_ties_at_the_top() -> None:
    _, grid = _bm25()

    assert grid.queries[0].scores[0] == grid.queries[0].scores[1] > 0.0
    assert all(query.n_excluded == 0 for query in grid.queries)
    assert grid.n_zero_vector == 1, "'zebra' names no corpus term"


# ---------------------------------------------------------------------------
# The near-tie census
# ---------------------------------------------------------------------------
def test_the_census_at_zero_counts_the_exact_ties_at_the_cutoff() -> None:
    record = evaluate_retrieval(_bm25()[1], _QRELS, _IDS, k=1, depth=3, seeds=0).as_dict()
    census = record["near_tie_census"]["absolute"]

    assert census["0.0"]["n_queries_crossing"] == record["n_tie_at_cutoff"] == 2
    assert [census[repr(t)]["n_queries_crossing"] for t in NEAR_TIE_TAUS] == sorted(
        census[repr(t)]["n_queries_crossing"] for t in NEAR_TIE_TAUS
    ), "a wider tau never counts fewer queries"


def test_contenders_count_the_documents_past_the_cutoff_within_tau() -> None:
    report = evaluate_retrieval(_bm25()[1], _QRELS, _IDS, k=1, depth=3, seeds=0)
    q1, q2, q3 = report.queries

    assert q1.contenders["0.0"] == 1, "d2 ties d1 just past position 1"
    assert q2.contenders["0.0"] == 0
    assert q3.contenders["0.0"] == 3, "every document scores 0 for a query of no corpus term"
    assert q3.relative_contenders["1e-06"] == 3, "a zero r_k leaves the relative tau at 0"


def test_a_degenerate_query_has_no_contenders() -> None:
    _, grid = _bm25()
    qrels = {**_QRELS, "q9": {"d1": 1}}

    report = evaluate_retrieval(grid, qrels, _IDS, k=1, depth=3, seeds=0)

    assert report.queries[-1].query_id == "q9"
    assert report.queries[-1].contenders == {}
    assert report.as_dict()["near_tie_census"]["relative"]["0.0"]["n_queries_crossing"] == 2


def test_the_census_survives_a_json_round_trip() -> None:
    record = evaluate_retrieval(_bm25()[1], _QRELS, _IDS, k=2, depth=4, seeds=1).as_dict()

    assert json.loads(json.dumps(record)) == record
    assert set(record["near_tie_census"]["absolute"]) == {repr(t) for t in NEAR_TIE_TAUS}


# ---------------------------------------------------------------------------
# Policy noise
# ---------------------------------------------------------------------------
def test_a_grid_against_itself_shows_no_noise() -> None:
    _, grid = _bm25()

    noise = policy_noise(grid, grid, k=1, label="exact")

    assert noise.as_dict() == {
        "reference": "exact",
        "k": 1,
        "n_queries": 3,
        "n_scores_differ": 0,
        "n_queries_differ": 0,
        "max_abs_delta": 0.0,
        "n_top_k_set_differs": 0,
        "n_top_k_order_differs": 0,
    }


def test_one_ulp_on_a_tied_document_is_found_and_reorders_the_top() -> None:
    _, grid = _bm25()
    bumped = _bump(grid, query=0, doc=1)

    noise = policy_noise(grid, bumped, k=1, label="exact")

    assert (noise.n_scores_differ, noise.n_queries_differ) == (1, 1)
    assert noise.max_abs_delta == math.ulp(grid.queries[0].scores[1])
    assert (noise.n_top_k_set_differs, noise.n_top_k_order_differs) == (1, 1)


def test_one_ulp_off_the_top_changes_no_ranking() -> None:
    _, grid = _bm25()

    noise = policy_noise(grid, _bump(grid, query=1, doc=3), k=1, label="exact")

    assert noise.n_queries_differ == 1
    assert (noise.n_top_k_set_differs, noise.n_top_k_order_differs) == (0, 0)


def test_the_policies_disagree_by_at_most_noise_on_this_corpus() -> None:
    _, exact = _bm25(Reduction.EXACT)
    for policy in list(Reduction):
        noise = policy_noise(exact, _bm25(policy)[1], k=2, label="exact")
        assert noise.max_abs_delta <= 1e-12, policy


def test_grids_over_different_queries_or_candidates_are_refused() -> None:
    _, grid = _bm25()
    fewer = dataclasses.replace(grid, queries=grid.queries[:2])
    moved = dataclasses.replace(
        grid,
        queries=(
            dataclasses.replace(grid.queries[0], candidate_ids=tuple(reversed(_IDS))),
            *grid.queries[1:],
        ),
    )

    with pytest.raises(ValueError, match="the two grids score different queries"):
        policy_noise(grid, fewer, k=1, label="exact")
    with pytest.raises(ValueError, match="query 'q1' ranks different candidates in the two grids"):
        policy_noise(grid, moved, k=1, label="exact")
