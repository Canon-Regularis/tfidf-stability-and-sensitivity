"""Partial candidate lists, as approximate search returns, and agreement.

Each grid is built by hand: a query ranks only the documents its list names, with an
attribute table over that list, so margins and tie orders are measured inside it.
"""

from __future__ import annotations

import dataclasses

import pytest

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.retrieval import evaluate_retrieval, top_k_agreement
from tfidf_stability.profiles import QueryMode
from tfidf_stability.profiles.user_profile import ProfileAggregation
from tfidf_stability.ranking.attributes import AttributeTable

_IDS = ["d1", "d2", "d3", "d4", "d5"]
_QRELS = {"q1": {"d2": 1}, "q2": {"d4": 1}}


def _grid(lists: dict[str, list[tuple[str, float]]]) -> QueryGrid:
    queries = tuple(
        EvaluatedQuery(
            query_id=qid,
            mode=QueryMode.TEXT,
            scores=tuple(score for _, score in pairs),
            table=AttributeTable.from_records([{"doc_id": d} for d, _ in pairs]),
            candidate_ids=tuple(d for d, _ in pairs),
            target=None,
            n_excluded=len(_IDS) - len(pairs),
        )
        for qid, pairs in lists.items()
    )
    return QueryGrid(
        queries=queries,
        mode=QueryMode.TEXT,
        aggregation=ProfileAggregation.TEXT_CONCAT,
        min_interactions=0,
        exclude_profile_items=False,
        n_users=0,
        n_degenerate=0,
        n_zero_vector=0,
    )


#: q1's list ties d1 with d2 at the top; q2's names d4, d5 and d3, in no particular order.
_LISTS = {
    "q1": [("d1", 0.75), ("d2", 0.75), ("d5", 0.5)],
    "q2": [("d5", 0.25), ("d4", 0.75), ("d3", 0.5)],
}


def test_a_partial_list_is_refused_unless_allowed() -> None:
    with pytest.raises(ValueError, match="query 'q1' does not rank the whole corpus"):
        evaluate_retrieval(_grid(_LISTS), _QRELS, _IDS, k=1, depth=3, seeds=0)


def test_an_allowed_partial_list_is_ranked_inside_itself() -> None:
    report = evaluate_retrieval(
        _grid(_LISTS), _QRELS, _IDS, k=1, depth=3, seeds=2, allow_partial=True
    )
    q1, q2 = report.queries

    assert q1.metrics["id_asc"]["ndcg@1"] == 0.0, "d1 leads its own tie"
    assert q1.metrics["id_desc"]["ndcg@1"] == 1.0
    assert (q1.margin, q1.tie_group) == (0.0, 2)
    assert q2.metrics["id_asc"]["rr@1"] == 1.0, "d4 scores highest in q2's list"
    assert report.as_dict()["partial"] is True
    assert set(report.means) == {"id_asc", "id_desc", "shuffle_0", "shuffle_1"}


def test_a_whole_corpus_report_says_so() -> None:
    whole = {"q1": [(d, 0.5) for d in _IDS]}

    assert (
        evaluate_retrieval(_grid(whole), {"q1": {"d1": 1}}, _IDS, k=1, depth=2, seeds=0).partial
        is False
    )


def test_lists_of_any_length_share_one_tie_order_per_identifier() -> None:
    """A shuffle orders d1 and d2 alike in a list of three and of five."""
    short = {"q1": [("d1", 0.75), ("d2", 0.75), ("d5", 0.5)]}
    long = {"q1": [(d, 0.75 if d in ("d1", "d2") else 0.5) for d in _IDS]}

    for seed in range(4):
        label = f"shuffle_{seed}"
        a = evaluate_retrieval(
            _grid(short), _QRELS, _IDS, k=1, depth=2, seeds=4, allow_partial=True
        )
        b = evaluate_retrieval(_grid(long), _QRELS, _IDS, k=1, depth=2, seeds=4)
        assert a.means[label]["ndcg@1"] == b.means[label]["ndcg@1"], label


def test_whole_and_partial_lists_in_one_grid_each_get_their_own_tables() -> None:
    """q1 and q3 rank the corpus, q2 a list; none borrows another's tables."""
    whole = [(d, 0.5 if d != "d4" else 0.75) for d in _IDS]
    grid = _grid({"q1": whole, "q2": _LISTS["q2"], "q3": whole})
    qrels = {"q1": {"d4": 1}, "q2": {"d4": 1}, "q3": {"d1": 1}}

    report = evaluate_retrieval(grid, qrels, _IDS, k=1, depth=3, seeds=1, allow_partial=True)

    assert [q.metrics["id_asc"]["rr@1"] for q in report.queries] == [1.0, 1.0, 0.0]
    assert report.queries[2].metrics["id_asc"]["recall@3"] == 1.0, "d1 leads the 0.5 tie"


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([("d1", 0.5), ("d1", 0.25)], "query 'q1' lists a candidate twice"),
        ([("d1", 0.5), ("d9", 0.25)], "query 'q1' lists a document outside the corpus"),
    ],
)
def test_a_bad_candidate_list_is_refused(pairs: list[tuple[str, float]], message: str) -> None:
    grid = _grid({"q1": [("d2", 0.5)]})
    broken = dataclasses.replace(
        grid.queries[0],
        scores=tuple(s for _, s in pairs),
        candidate_ids=tuple(d for d, _ in pairs),
        table=AttributeTable.from_records([{"doc_id": f"x{i}"} for i in range(len(pairs))]),
    )

    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(
            dataclasses.replace(grid, queries=(broken,)),
            _QRELS,
            _IDS,
            k=1,
            depth=2,
            seeds=0,
            allow_partial=True,
        )


# ---------------------------------------------------------------------------
# Agreement at the top
# ---------------------------------------------------------------------------
def test_a_grid_agrees_with_itself_completely() -> None:
    agreement = top_k_agreement(_grid(_LISTS), _grid(_LISTS), k=2)

    assert agreement.as_dict() == {
        "k": 2,
        "n_queries": 2,
        "mean_recall": 1.0,
        "mean_fks": 0.0,
        "n_sets_differ": 0,
        "recall": {"q1": 1.0, "q2": 1.0},
        "fks": {"q1": 0.0, "q2": 0.0},
    }


def test_a_missed_document_lowers_recall_against_the_reference() -> None:
    """The approximate q2 list lost d4, the reference's best, and kept d3 and d5."""
    approximate = {**_LISTS, "q2": [("d5", 0.25), ("d3", 0.5)]}

    agreement = top_k_agreement(_grid(_LISTS), _grid(approximate), k=2)

    assert agreement.recall == {"q1": 1.0, "q2": 0.5}
    assert agreement.mean_recall == 0.75
    assert agreement.n_sets_differ == 1
    assert agreement.fks["q2"] > 0.0
    assert agreement.mean_fks == agreement.fks["q2"] / 2, "q1 agrees, so its fks is 0"


def test_agreement_with_no_queries_is_complete() -> None:
    empty = _grid({})

    assert (
        top_k_agreement(empty, empty, k=1).mean_recall,
        top_k_agreement(empty, empty, k=1).mean_fks,
    ) == (1.0, 0.0)


@pytest.mark.parametrize(
    ("other", "k", "message"),
    [
        ({"q1": _LISTS["q1"]}, 1, "the two grids score different queries"),
        (_LISTS, 0, "k must be at least 1, got 0"),
    ],
)
def test_agreement_refuses_mismatched_grids_and_bad_cutoffs(
    other: dict[str, list[tuple[str, float]]], k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        top_k_agreement(_grid(_LISTS), _grid(other), k=k)
