"""Queries a dataset ships as text: construction, ordering and evaluation."""

from __future__ import annotations

import pytest

from tfidf_stability.analysis.query_grid import build_query_grid, evaluate
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline, preprocess_records
from tfidf_stability.profiles import QueryMode, text_queries
from tfidf_stability.profiles.user_profile import ProfileAggregation
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

_RECORDS = [
    {"doc_id": "d1", "text": "apple banana"},
    {"doc_id": "d2", "text": "cherry durian"},
    {"doc_id": "d3", "text": "apple cherry"},
]


def test_each_text_becomes_one_query_in_utf8_order() -> None:
    pipeline = PreprocessingPipeline()
    queries = text_queries(
        {"q9": "cherry", "q10": "Apple bananas", "Q": "durian"}, pipeline, n_documents=3
    )

    assert [q.query_id for q in queries] == ["Q", "q10", "q9"], "UTF-8 order: 'q10' < 'q9'"
    first = queries.queries[1]
    assert first.mode is QueryMode.TEXT
    assert first.features == tuple(pipeline.preprocess("Apple bananas"))
    assert first.profile is None
    assert first.target is None
    assert first.excluded == frozenset()
    assert first.n_candidates == 3


def test_a_text_query_set_builds_no_profile() -> None:
    queries = text_queries({"q": "apple"}, PreprocessingPipeline(), n_documents=5)

    assert queries.mode is QueryMode.TEXT
    assert queries.aggregation is ProfileAggregation.TEXT_CONCAT
    assert queries.provenance() == {
        "mode": "text",
        "aggregation": "text_concat",
        "min_interactions": 0,
        "exclude_profile_items": False,
        "n_users": 0,
        "n_queries": 1,
        "candidates_min": 5,
        "candidates_median": 5,
        "candidates_max": 5,
    }


def test_a_query_of_stopwords_alone_is_degenerate_and_counted() -> None:
    pipeline = PreprocessingPipeline()
    features = preprocess_records(pipeline, _RECORDS)
    model = TfidfVectoriser().fit(features, [r["doc_id"] for r in _RECORDS])
    queries = text_queries(
        {"q1": "apple", "q2": "the of and"}, pipeline, n_documents=model.n_documents
    )

    grid = evaluate(queries, model, _RECORDS, [r["doc_id"] for r in _RECORDS])

    assert queries.queries[1].is_degenerate
    assert grid.n_degenerate == 1
    assert [q.query_id for q in grid.queries] == ["q1"]
    assert grid.queries[0].scores[1] == 0.0, "d2 shares no term with 'apple'"
    assert grid.queries[0].scores[0] > 0.0


def test_a_negative_corpus_size_is_refused() -> None:
    with pytest.raises(ValueError, match="n_documents must be non-negative, got -1"):
        text_queries({}, PreprocessingPipeline(), n_documents=-1)


def test_the_interaction_grid_refuses_the_text_mode() -> None:
    with pytest.raises(ValueError, match=r"text is implemented but section 7\.1 excludes it"):
        build_query_grid([("u", "d1", 5.0)], {"d1": ["apple"]}, ["d1"], mode="text")


def test_the_text_mode_spells_its_value() -> None:
    assert QueryMode("text") is QueryMode.TEXT
    assert str(QueryMode.TEXT) == "text"
