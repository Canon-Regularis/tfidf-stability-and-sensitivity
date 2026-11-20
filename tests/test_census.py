"""E6: the tie census over a small score store and over hand-built reports.

The store's four documents give one query an exact tie at the cutoff, one a gap of
2**-24 below a score of 64, one a zero vector, and leave one degenerate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tfidf_stability.evaluation.census import E6_ROWS, CensusRow, census_row, run_census
from tfidf_stability.evaluation.metrics import MeanMetric
from tfidf_stability.evaluation.retrieval import NEAR_TIE_TAUS, RetrievalReport
from tfidf_stability.evaluation.score_store import ScoreStore, StoreKind

_IDS = ["d1", "d2", "d3", "d4"]
_RECORDS = [{"doc_id": d, "text": f"text {d}"} for d in _IDS]
_SCORES = {
    "q1": [0.5, 0.5, 0.25, 0.0],
    "q2": [64.0, 64.0 - 2.0**-24, 1.0, 0.0],
    "q3": [0.0, 0.0, 0.0, 0.0],
}
#: d2 and d4 each win only under ``id_desc``; q4 is degenerate.
_QRELS = {"q1": {"d2": 1}, "q2": {"d1": 1}, "q3": {"d4": 1}, "q4": {"d1": 1}}
_TAUS = [repr(t) for t in NEAR_TIE_TAUS]


def _store(tmp_path: Path) -> ScoreStore:
    store = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS, create=True)
    for label, kind in [
        ("tfidf_n2.naive", StoreKind.LEXICAL),
        ("tfidf_n2.exact", StoreKind.LEXICAL),
        ("other.naive", StoreKind.LEXICAL),
        ("ance.run", StoreKind.RUN),
        ("glove.exact", StoreKind.MATRIX),
    ]:
        store = store.add(label, _SCORES, kind=kind, params={}, degenerate=["q4"])
    return store


def _census(tmp_path: Path) -> dict[str, Any]:
    return run_census(_store(tmp_path), _QRELS, cutoffs=[1, 3], depth=2, seeds=2)


def test_the_rows_follow_the_issue_order_at_the_naive_policy() -> None:
    assert [(row.name, row.label) for row in E6_ROWS] == [
        ("TF-IDF", "tfidf_n2.naive"),
        ("BM25 exact", "bm25-exact_n1.naive"),
        ("BM25 lucene_byte", "bm25-lucene_byte_n1.naive"),
        ("SPLADE float", "splade-float.naive"),
        ("SPLADE x100", "splade-x100.naive"),
        ("dense fp32", "dense-fp32.run"),
    ]


def test_a_stored_row_is_measured_and_an_absent_one_gives_its_reason(tmp_path: Path) -> None:
    payload = _census(tmp_path)
    rows = payload["rows"]

    assert [row["status"] for row in rows] == ["measured"] + ["missing"] * 5
    assert rows[1] == {
        "name": "BM25 exact",
        "label": "bm25-exact_n1.naive",
        "status": "missing",
        "reason": "build the score store first",
    }
    assert rows[3]["reason"] == "SPLADE scores arrive with #18"
    assert rows[5]["reason"] == "dense scores arrive with #16"
    assert (rows[0]["variant"], rows[0]["kind"]) == ("naive", "lexical")
    assert (payload["cutoffs"], payload["depth"], payload["seeds"]) == ([1, 3], 2, 2)
    assert json.loads(json.dumps(payload)) == payload


def test_exact_and_near_ties_at_the_cutoff_are_shares_of_scored_queries(tmp_path: Path) -> None:
    """q1 and q3 tie at 1; q2's gap of 2**-24 sits inside 1e-9 only relative to 64."""
    rows = _census(tmp_path)["rows"]
    census = rows[0]["by_cutoff"]["1"]

    assert (census["n_queries"], census["n_degenerate"]) == (4, 1)
    assert (census["n_zero_vector"], census["n_scored"]) == (1, 3)
    assert census["exact_tie"] == {
        "n": 2,
        "share": 2 / 3,
        "share_excluding_zero_vector": 1 / 2,
    }
    assert census["near_ties"] == dict(
        zip(
            _TAUS,
            [
                {"absolute": 2 / 3, "relative": 2 / 3},
                {"absolute": 2 / 3, "relative": 2 / 3},
                {"absolute": 2 / 3, "relative": 1.0},
                {"absolute": 1.0, "relative": 1.0},
                {"absolute": 1.0, "relative": 1.0},
            ],
            strict=True,
        )
    )


def test_the_metric_range_spans_id_asc_and_id_desc(tmp_path: Path) -> None:
    rows = _census(tmp_path)["rows"]
    census = rows[0]["by_cutoff"]["1"]

    assert census["metric_range"]["ndcg@1"] == {
        "min": 0.25,
        "max": 0.75,
        "range": 0.5,
        "min_variant": "id_asc",
        "max_variant": "id_desc",
    }
    assert census["metric_range"]["recall@2"]["range"] == 0.25
    assert census["tie_aware"]["ndcg@1"] == {"mean": 0.4375, "n_queries": 4}


def test_a_cutoff_past_the_depth_is_evaluated_to_itself(tmp_path: Path) -> None:
    """At 3 only the zero vector ties: q1 and q2 fall by 0.25 and 1 past the third."""
    rows = _census(tmp_path)["rows"]
    census = rows[0]["by_cutoff"]["3"]

    assert set(census["metric_range"]) == {"ndcg@3", "recall@3", "rr@3"}
    assert set(census["tie_aware"]) == {"ndcg@3", "recall@3", "rr@3"}
    assert census["exact_tie"] == {"n": 1, "share": 1 / 3, "share_excluding_zero_vector": 0.0}


def test_extra_rows_hold_unnamed_naive_run_and_matrix_labels(tmp_path: Path) -> None:
    payload = _census(tmp_path)
    extra = payload["extra_rows"]

    assert [(row["name"], row["label"]) for row in extra] == [
        ("ance.run", "ance.run"),
        ("glove.exact", "glove.exact"),
        ("other.naive", "other.naive"),
    ]
    assert [row["kind"] for row in extra] == ["run", "matrix", "lexical"]


def test_rows_given_by_the_caller_replace_the_issue_rows(tmp_path: Path) -> None:
    rows = (CensusRow("Mine", "other.naive", "absent"), CensusRow("Gone", "gone.naive", "later"))
    payload = run_census(_store(tmp_path), _QRELS, cutoffs=[1], depth=1, seeds=0, rows=rows)
    extra = payload["extra_rows"]

    assert [row["label"] for row in extra] == ["ance.run", "glove.exact", "tfidf_n2.naive"]
    assert payload["rows"][1] == {
        "name": "Gone",
        "label": "gone.naive",
        "status": "missing",
        "reason": "later",
    }


def test_each_measured_label_gives_one_spread_per_cutoff_and_metric(tmp_path: Path) -> None:
    spreads = _census(tmp_path)["spreads"]

    assert len(spreads) == 4 * 2 * 3
    assert spreads[0] == {
        "experiment": "e6",
        "metric": "ndcg@1",
        "spread": 0.5,
        "source": {
            "row": "TF-IDF",
            "label": "tfidf_n2.naive",
            "min_variant": "id_asc",
            "max_variant": "id_desc",
        },
    }
    assert {frozenset(s) for s in spreads} == {
        frozenset({"experiment", "metric", "spread", "source"})
    }
    assert all(math.isfinite(s["spread"]) and s["spread"] >= 0.0 for s in spreads)
    assert {s["source"]["row"] for s in spreads} == {
        "TF-IDF",
        "ance.run",
        "glove.exact",
        "other.naive",
    }


def test_degenerate_queries_are_counted_but_left_out_of_every_share(tmp_path: Path) -> None:
    """z holds one zero-vector query; y scores none, so every share is over none."""
    store = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS, create=True)
    store = store.add(
        "z.naive", {"q1": [0.0] * 4}, kind=StoreKind.LEXICAL, params={}, degenerate=["q2"]
    )
    store = store.add("y.naive", {}, kind=StoreKind.LEXICAL, params={}, degenerate=["q1", "q2"])
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    payload = run_census(store, qrels, cutoffs=[1], depth=1, seeds=0, rows=())
    y, z = (row["by_cutoff"]["1"] for row in payload["extra_rows"])

    assert z["exact_tie"] == {"n": 1, "share": 1.0, "share_excluding_zero_vector": 0.0}
    assert (z["n_degenerate"], z["n_zero_vector"], z["n_scored"]) == (1, 1, 1)
    assert y["n_scored"] == 0
    assert y["exact_tie"] == {"n": 0, "share": 0.0, "share_excluding_zero_vector": 0.0}
    assert y["near_ties"] == {tau: {"absolute": 0.0, "relative": 0.0} for tau in _TAUS}


def _means(value: float) -> dict[str, MeanMetric]:
    return {name: MeanMetric(value, 2, 0) for name in ("ndcg@1", "recall@2", "rr@1")}


def test_a_shuffle_can_bound_the_range_beside_the_identifier_orders() -> None:
    report = RetrievalReport(
        k=1,
        depth=2,
        means={"id_asc": _means(0.5), "id_desc": _means(0.25), "shuffle_0": _means(0.75)},
        queries=(),
        partial=False,
    )
    census = census_row(report, {"ndcg@1": MeanMetric(0.375, 2, 1)})

    assert census["metric_range"]["rr@1"] == {
        "min": 0.25,
        "max": 0.75,
        "range": 0.5,
        "min_variant": "id_desc",
        "max_variant": "shuffle_0",
    }
    assert census["tie_aware"] == {"ndcg@1": {"mean": 0.375, "n_queries": 2}}
    assert census["n_scored"] == 0
