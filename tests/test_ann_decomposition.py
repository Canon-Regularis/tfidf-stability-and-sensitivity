"""E9: each ANN top-k difference classed against the exact top-k.

Fixture lists are written by hand over a six-document corpus, so every class, refusal
and order-only difference has a list that produces it and nothing else.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.ann_decomposition import (
    MISSING_RUNS,
    Difference,
    QueryDecomposition,
    classify_query,
    decompose,
    run_ann,
)
from tfidf_stability.evaluation.external import ExternalDataError, grid_from_scores
from tfidf_stability.evaluation.score_store import ScoreStore, StoreKind

_IDS = ["d1", "d2", "d3", "d4", "d5", "d6"]
_RECORDS = [{"doc_id": d} for d in _IDS]
#: ``qa`` has no tie; ``qt`` ties d2, d3 and d4 across the cutoffs 2 and 3.
_EXACT = {
    "qa": [0.9, 0.8, 0.7, 0.6, 0.5, 0.0],
    "qt": [0.9, 0.5, 0.5, 0.5, 0.1, 0.0],
}
_LABEL = "tfidf_n2.exact"
_SPREAD_KEYS = {"experiment", "metric", "spread", "source"}


def _grid() -> QueryGrid:
    return grid_from_scores(_EXACT, _RECORDS, _IDS)


def _query(query_id: str) -> EvaluatedQuery:
    return next(q for q in _grid().queries if q.query_id == query_id)


def _store(tmp_path: Path) -> ScoreStore:
    store = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS, create=True)
    return store.add(_LABEL, _EXACT, kind=StoreKind.LEXICAL, params={})


def test_a_document_the_list_never_names_is_approximation() -> None:
    row = classify_query(_query("qa"), {"d1": 0.9, "d2": 0.8, "d4": 0.6, "d5": 0.5}, k=3)

    assert row.missed == {"d3": Difference.APPROXIMATION}
    assert (row.order, row.n_listed, row.n_scores_differ) == (None, 4, 0)
    assert row.overlap == 2 / 3
    assert row.differs


def test_a_document_the_ann_scores_push_out_is_numerics() -> None:
    row = classify_query(_query("qa"), {"d1": 0.9, "d2": 0.8, "d4": 0.75, "d3": 0.7}, k=3)

    assert row.missed == {"d3": Difference.NUMERICS}
    assert row.n_scores_differ == 1, "only d4's ANN score differs from the exact one"
    assert not row.boundary_tie


def test_a_document_lost_inside_an_exact_tie_is_tie_breaking() -> None:
    row = classify_query(_query("qt"), {"d1": 0.9, "d3": 0.5, "d2": 0.5}, k=2)

    assert row.missed == {"d2": Difference.TIE_BREAKING}
    assert row.boundary_tie
    assert row.n_scores_differ == 0


def test_a_length_k_list_breaking_a_boundary_tie_otherwise_is_tie_breaking() -> None:
    row = classify_query(_query("qt"), {"d1": 0.9, "d4": 0.5}, k=2)

    assert row.missed == {"d2": Difference.TIE_BREAKING}
    assert (row.order, row.n_scores_differ, row.boundary_tie) == (None, 0, True)
    assert row.overlap == 0.5


_TIE = Difference.TIE_BREAKING
_APPROX = Difference.APPROXIMATION


@pytest.mark.parametrize(
    ("scores", "listed", "k", "missed"),
    [
        (_EXACT["qt"], {"d1": 0.9, "d4": 0.5, "d3": 0.5}, 2, {"d2": _TIE}),
        (_EXACT["qt"], {"d1": 0.9, "d4": 0.5, "d5": 0.1}, 3, {"d2": _TIE, "d3": _APPROX}),
        (_EXACT["qt"], {"d1": 0.9, "d2": 0.5, "d5": 0.1}, 3, {"d3": _APPROX}),
        (_EXACT["qt"], {"d1": 0.9, "d5": 0.6, "d4": 0.5}, 2, {"d2": _TIE}),
        (
            [0.9, 0.5, 0.5, 0.5, 0.5, 0.0],
            {"d1": 0.9, "d5": 0.5, "d4": 0.5, "d3": 0.5},
            3,
            {"d2": _TIE, "d3": _TIE},
        ),
    ],
    ids=["k_plus_one", "one_substitute_for_two", "exact_top_never_stands_in", "rescored", "group"],
)
def test_each_tied_substitute_covers_one_unlisted_miss(
    scores: list[float], listed: dict[str, float], k: int, missed: dict[str, Difference]
) -> None:
    """Substitutes come from the exact re-ranking, outside the exact top k."""
    query = grid_from_scores({"q": scores}, _RECORDS, _IDS).queries[0]

    assert classify_query(query, listed, k=k).missed == missed


def test_a_list_shorter_than_k_misses_the_rest_as_approximation() -> None:
    row = classify_query(_query("qa"), {"d1": 0.9}, k=3)

    assert row.missed == {"d2": Difference.APPROXIMATION, "d3": Difference.APPROXIMATION}
    assert row.count(Difference.APPROXIMATION) == 2
    assert row.overlap == 1 / 3


def test_the_exact_set_in_tie_order_is_order_ties() -> None:
    row = classify_query(_query("qt"), {"d1": 0.9, "d3": 0.5, "d2": 0.5}, k=3)

    assert (row.missed, row.order) == ({}, Difference.ORDER_TIES)
    assert row.overlap == 1.0
    assert row.differs


def test_the_exact_set_reordered_by_ann_scores_is_order_numerics() -> None:
    row = classify_query(_query("qa"), {"d2": 0.95, "d1": 0.9, "d3": 0.7}, k=3)

    assert (row.missed, row.order) == ({}, Difference.ORDER_NUMERICS)
    assert row.n_scores_differ == 1


def test_the_exact_list_differs_in_nothing_but_a_signed_zero() -> None:
    listed = dict(zip(_IDS, [0.9, 0.8, 0.7, 0.6, 0.5, -0.0], strict=True))
    row = classify_query(_query("qa"), listed, k=10)

    assert (row.missed, row.order, row.differs) == ({}, None, False)
    assert row.n_scores_differ == 1, "-0.0 and 0.0 differ in their bits"
    assert not row.boundary_tie, "k past the corpus leaves no boundary"


@pytest.mark.parametrize(
    ("listed", "message"),
    [
        ({"d1": 0.9, "zz": 0.1}, r"query 'qa' lists 'zz', outside the corpus"),
        ({"d1": math.nan}, r"query 'qa' gives 'd1' the score nan"),
        ({"d1": 0.5, "d2": 0.6}, r"query 'qa' is not sorted descending at 'd2'"),
    ],
)
def test_a_list_outside_the_corpus_non_finite_or_unsorted_is_refused(
    listed: dict[str, float], message: str
) -> None:
    with pytest.raises(ExternalDataError, match=message):
        classify_query(_query("qa"), listed, k=2)


def test_a_cutoff_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match=r"k must be at least 1, got 0"):
        classify_query(_query("qa"), {"d1": 0.9}, k=0)


def test_a_row_coerces_its_classes_and_serialises() -> None:
    row = QueryDecomposition(
        query_id="q",
        k=2,
        missed={"d1": "approximation"},  # type: ignore[dict-item]
        order="order_ties",  # type: ignore[arg-type]
        n_listed=1,
        n_scores_differ=0,
        boundary_tie=False,
        overlap=0.5,
    )

    assert row.missed["d1"] is Difference.APPROXIMATION
    assert row.order is Difference.ORDER_TIES
    assert row.as_dict()["counts"] == {"approximation": 1, "numerics": 0, "tie_breaking": 0}
    assert row.as_dict()["order"] == "order_ties"
    assert row.as_dict()["overlap"] == 0.5


@pytest.mark.property
@given(
    scores=st.lists(st.sampled_from([0.0, 0.25, 0.5]), min_size=6, max_size=6),
    subset=st.lists(st.sampled_from(_IDS), unique=True),
    k=st.integers(min_value=1, max_value=7),
)
@settings(max_examples=150, deadline=None)
def test_exact_scores_in_any_tie_order_leave_no_numerics(
    scores: list[float], subset: list[str], k: int
) -> None:
    """A list scored and sorted exactly differs by approximation or ties."""
    query = grid_from_scores({"q": scores}, _RECORDS, _IDS).queries[0]
    exact = dict(zip(_IDS, scores, strict=True))
    listed = {doc: exact[doc] for doc in sorted(subset, key=lambda d: -exact[d])}
    row = classify_query(query, listed, k=k)

    assert row.count(Difference.NUMERICS) == 0
    assert row.order in (None, Difference.ORDER_TIES)
    assert len(row.missed) == round((1 - row.overlap) * min(k, len(_IDS)))


def test_decompose_counts_every_class_over_the_reference() -> None:
    run = {"qt": {"d1": 0.9, "d3": 0.5, "d2": 0.5}}
    result = decompose(_grid(), run, k=2, label="hnsw-a")

    assert (result.n_queries, result.n_sets_differ, result.n_boundary_ties) == (2, 2, 1)
    assert result.mean_overlap == 0.25, "qa's omitted list keeps nothing, qt's keeps one of two"
    assert result.totals == {
        "approximation": 2,
        "numerics": 0,
        "tie_breaking": 1,
        "order_ties": 0,
        "order_numerics": 0,
    }
    assert [q.query_id for q in result.queries] == ["qa", "qt"]
    assert result.as_dict()["queries"][1]["missed"] == {"d2": "tie_breaking"}


def test_decompose_keeps_only_differing_rows_and_counts_order_classes() -> None:
    run = {"qa": {"d1": 0.9, "d2": 0.8, "d3": 0.7}, "qt": {"d1": 0.9, "d3": 0.5, "d2": 0.5}}
    result = decompose(_grid(), run, k=3, label="hnsw-a")

    assert [q.query_id for q in result.queries] == ["qt"]
    assert result.totals["order_ties"] == 1
    assert (result.n_sets_differ, result.mean_overlap) == (0, 1.0)
    assert result.as_dict()["label"] == "hnsw-a"


def test_decompose_refuses_an_unknown_query_and_an_empty_reference() -> None:
    with pytest.raises(ExternalDataError, match=r"run 'r' lists 'qz', a query the reference"):
        decompose(_grid(), {"qz": {}}, k=1, label="r")
    with pytest.raises(ValueError, match=r"the reference scores no queries"):
        decompose(grid_from_scores({}, _RECORDS, _IDS), {}, k=1, label="r")


def test_run_ann_without_runs_names_what_is_missing(tmp_path: Path) -> None:
    payload = run_ann(_store(tmp_path), _LABEL, {}, cutoffs=[2, 3])

    assert payload == {
        "reference": _LABEL,
        "cutoffs": [2, 3],
        "runs": [],
        "missing": MISSING_RUNS,
        "spreads": [],
    }


def test_run_ann_gives_one_spread_per_cutoff_for_each_shared_group(tmp_path: Path) -> None:
    runs = {
        "ivf": {"qa": {"d1": 0.9}},
        "hnsw-b": {"qa": {"d1": 0.9, "d2": 0.8, "d3": 0.7}, "qt": {"d1": 0.9, "d2": 0.5}},
        "hnsw-a": {"qa": {"d1": 0.9, "d4": 0.6}, "qt": {"d1": 0.9, "d3": 0.5}},
    }
    groups = {"hnsw-a": "hnsw", "hnsw-b": "hnsw", "ivf": "ivf"}
    payload = run_ann(_store(tmp_path), _LABEL, runs, cutoffs=[1, 2], groups=groups)

    assert payload["missing"] is None
    assert [r["label"] for r in payload["runs"]] == ["hnsw-a", "hnsw-b", "ivf"]
    assert [r["group"] for r in payload["runs"]] == ["hnsw", "hnsw", "ivf"]
    assert payload["spreads"] == [
        {
            "experiment": "e9",
            "metric": "overlap@1",
            "spread": 0.0,
            "source": {"group": "hnsw", "min_run": "hnsw-a", "max_run": "hnsw-a"},
        },
        {
            "experiment": "e9",
            "metric": "overlap@2",
            "spread": 0.5,
            "source": {"group": "hnsw", "min_run": "hnsw-a", "max_run": "hnsw-b"},
        },
    ]


def test_run_ann_puts_every_run_in_one_group_by_default(tmp_path: Path) -> None:
    runs = {"a": {"qa": {"d1": 0.9}}, "b": {"qa": {"d2": 0.8}}}
    payload = run_ann(_store(tmp_path), _LABEL, runs, cutoffs=[1])

    assert [set(s) for s in payload["spreads"]] == [_SPREAD_KEYS]
    assert payload["spreads"][0]["source"] == {"group": "all", "min_run": "b", "max_run": "a"}
    assert payload["spreads"][0]["spread"] == 0.5


def test_run_ann_gives_no_spread_for_a_lone_run(tmp_path: Path) -> None:
    payload = run_ann(_store(tmp_path), _LABEL, {"a": {"qa": {"d1": 0.9}}}, cutoffs=[1])

    assert payload["spreads"] == []
    assert payload["runs"][0]["decompositions"][0]["mean_overlap"] == 0.5


@pytest.mark.parametrize("cutoffs", [[], [2, 2], [0, 1]])
def test_run_ann_refuses_bad_cutoffs(tmp_path: Path, cutoffs: list[int]) -> None:
    with pytest.raises(ValueError, match=r"cutoffs must be distinct positive integers"):
        run_ann(_store(tmp_path), _LABEL, {}, cutoffs=cutoffs)


def test_run_ann_refuses_groups_that_do_not_name_the_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"groups must name every run and nothing else"):
        run_ann(_store(tmp_path), _LABEL, {"a": {}}, cutoffs=[1], groups={"b": "x"})
