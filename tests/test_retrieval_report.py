"""The retrieval report's derived fields: ties, contenders, census, spread.

Each grid holds hand-picked scores, so every count the report derives can be worked
out by hand: a tie below the cutoff, a gap of 7e-10 at it, and a shuffle spread.
"""

from __future__ import annotations

import math

from tfidf_stability.evaluation.external import grid_from_scores
from tfidf_stability.evaluation.metrics import MeanMetric
from tfidf_stability.evaluation.retrieval import RetrievalReport, evaluate_retrieval

_IDS = ["d1", "d2", "d3", "d4"]
_RECORDS = [{"doc_id": d} for d in _IDS]
_QRELS = {"q1": {"d1": 1}, "q2": {"d3": 1}}


def _report(scores: dict[str, tuple[float, ...]], k: int, depth: int) -> RetrievalReport:
    grid = grid_from_scores(scores, _RECORDS, _IDS)
    return evaluate_retrieval(grid, _QRELS, _IDS, k=k, depth=depth, seeds=0)


def test_the_tie_group_is_the_one_holding_position_k() -> None:
    """d2 and d3 tie at positions 2 and 3; position 1 holds d1 alone."""
    scores = {"q1": (0.9, 0.5, 0.5, 0.1), "q2": (0.9, 0.5, 0.5, 0.1)}

    at_one = _report(scores, k=1, depth=4).queries[0]
    at_two = _report(scores, k=2, depth=4).queries[0]

    assert (at_one.tie_group, at_one.margin) == (1, 0.4)
    assert (at_two.tie_group, at_two.margin) == (2, 0.0), "the group spans positions 2 and 3"


def test_absolute_and_relative_contenders_differ_at_a_small_score() -> None:
    """r_2 sits 7e-10 below r_1 = 0.5: inside 1e-9, outside 1e-9 * 0.5."""
    gap = 7e-10
    scores = {"q1": (0.5, 0.5 - gap, 0.1, 0.0), "q2": (0.5, 0.5 - gap, 0.1, 0.0)}

    query = _report(scores, k=1, depth=4).queries[0]

    assert query.contenders["0.0"] == 0
    assert query.contenders["1e-09"] == 1
    assert query.relative_contenders["1e-09"] == 0
    assert query.relative_contenders["1e-06"] == 1


def test_the_census_averages_contenders_over_queries() -> None:
    scores = {"q1": (0.5, 0.5, 0.5, 0.1), "q2": (0.9, 0.5, 0.1, 0.0)}

    census = _report(scores, k=1, depth=4).as_dict()["near_tie_census"]["absolute"]["0.0"]

    assert census == {"n_queries_crossing": 1, "max_contenders": 2, "mean_contenders": 1.0}


def test_a_cutoff_equal_to_the_depth_is_accepted() -> None:
    report = _report({"q1": (0.4, 0.3, 0.2, 0.1)}, k=4, depth=4)

    assert report.depth == report.k == 4
    assert report.queries[0].margin is None, "k = N leaves no r_{k+1}"


def test_an_empty_report_has_an_empty_census_and_no_spread() -> None:
    names = ("ndcg@1", "recall@1", "rr@1")
    report = RetrievalReport(k=1, depth=1, means={}, queries=(), partial=False)

    record = report.as_dict()

    assert record["shuffle_spread"] == {}
    assert record["near_tie_census"]["absolute"]["1e-12"] == {
        "n_queries_crossing": 0,
        "max_contenders": 0,
        "mean_contenders": 0.0,
    }
    assert report.metric_names == names


def test_the_shuffle_spread_is_the_range_of_the_shuffled_means() -> None:
    def metrics(value: float) -> dict[str, MeanMetric]:
        return {name: MeanMetric(value, 1, 0) for name in ("ndcg@1", "recall@1", "rr@1")}

    means = {
        "id_asc": metrics(0.5),
        "id_desc": metrics(0.5),
        "shuffle_0": metrics(0.25),
        "shuffle_1": metrics(0.75),
    }
    report = RetrievalReport(k=1, depth=1, means=means, queries=(), partial=False)

    spread = report.as_dict()["shuffle_spread"]["ndcg@1"]

    assert spread == {"min": 0.25, "max": 0.75, "range": 0.5}
    assert math.isclose(spread["range"], spread["max"] - spread["min"])
