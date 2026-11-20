"""Tie-aware expected metrics against every order of each tie group.

The brute force ranks each permutation inside the groups and averages exact fractions,
so the closed forms are checked for equality, not closeness.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.metrics import (
    discount,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from tfidf_stability.evaluation.tie_aware import (
    exact_expectation,
    expected_means,
    expected_metrics,
    score_groups,
)
from tfidf_stability.profiles import QueryMode
from tfidf_stability.profiles.user_profile import ProfileAggregation
from tfidf_stability.ranking.attributes import AttributeTable


def _orders(scores: list[float]) -> list[list[int]]:
    """Every ranking that sorts by score and orders each tie group freely."""
    groups = score_groups(scores)
    return [
        [i for part in parts for i in part]
        for parts in itertools.product(*(itertools.permutations(g) for g in groups))
    ]


def _brute(
    scores: list[float], ids: list[str], judgements: dict[str, int], k: int, depth: int
) -> tuple[Fraction, Fraction, Fraction]:
    """Mean exact DCG at k, hits at depth and RR at k over every tie order."""
    dcg = hits = rr = Fraction(0)
    orders = _orders(scores)
    for order in orders:
        ranked = [ids[i] for i in order]
        grades = [judgements.get(doc, 0) for doc in ranked]
        dcg += sum(
            (Fraction(g) * Fraction(discount(r)) for r, g in enumerate(grades[:k], 1) if g > 0),
            Fraction(0),
        )
        hits += sum(1 for g in grades[:depth] if g > 0)
        first = next((r for r, g in enumerate(grades[:k], 1) if g > 0), None)
        rr += Fraction(0) if first is None else Fraction(1, first)
    n = len(orders)
    return dcg / n, hits / n, rr / n


_CASES = [
    # A group straddling the cutoff, with two relevant documents inside it.
    ([0.9, 0.5, 0.5, 0.5, 0.5, 0.1], {"d1": 2, "d3": 1, "d5": 1}, 2, 3),
    # Every document tied, one relevant: RR averages 1/rank over all positions.
    ([0.25] * 5, {"d2": 1}, 5, 5),
    # The first relevant group starts past the cutoff.
    ([0.9, 0.8, 0.5, 0.5], {"d3": 1}, 2, 4),
    # Negative grades gain nothing and are not relevant.
    ([0.5, 0.5, 0.5, 0.2], {"d0": -1, "d1": 2, "d3": 1}, 2, 2),
    # -0.0 and 0.0 tie; k beyond the list.
    ([0.0, -0.0, 0.0, 0.5], {"d1": 1, "d2": 3}, 7, 7),
    # Three relevant among four tied documents, cut at the second position.
    ([0.4, 0.4, 0.4, 0.4, 0.3], {"d0": 1, "d1": 1, "d3": 1}, 2, 2),
    # Depth past k: recall counts a group that nDCG and RR never reach.
    ([0.9, 0.7, 0.5, 0.5, 0.5], {"d2": 1, "d4": 1}, 1, 4),
]


@pytest.mark.parametrize(("scores", "judgements", "k", "depth"), _CASES)
def test_the_closed_forms_equal_the_mean_over_every_tie_order(
    scores: list[float], judgements: dict[str, int], k: int, depth: int
) -> None:
    ids = [f"d{i}" for i in range(len(scores))]
    exact = exact_expectation(scores, ids, judgements, k=k, depth=depth)

    assert (sum(exact.dcg_terms, Fraction(0)), exact.hits, exact.rr) == _brute(
        scores, ids, judgements, k, depth
    )


@pytest.mark.property
@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.sampled_from([0.0, 0.25, 0.5, 1.0]), min_size=1, max_size=7),
    st.lists(st.integers(-1, 3), min_size=7, max_size=7),
    st.integers(1, 8),
    st.integers(0, 4),
)
def test_any_forced_ties_match_the_brute_force(
    scores: list[float], grades: list[int], k: int, extra: int
) -> None:
    ids = [f"d{i}" for i in range(len(scores))]
    judgements = {f"d{i}": g for i, g in enumerate(grades)}
    exact = exact_expectation(scores, ids, judgements, k=k, depth=k + extra)

    assert (sum(exact.dcg_terms, Fraction(0)), exact.hits, exact.rr) == _brute(
        scores, ids, judgements, k, k + extra
    )


@pytest.mark.property
@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.floats(-4, 4, allow_nan=False), min_size=1, max_size=40, unique=True),
    st.dictionaries(st.integers(0, 45), st.integers(-1, 3), max_size=12),
    st.integers(1, 12),
    st.integers(0, 30),
)
def test_a_tie_free_query_scores_bit_for_bit_as_its_one_ranking(
    scores: list[float], graded: dict[int, int], k: int, extra: int
) -> None:
    """Distinct scores leave one order, ranked here by hand."""
    ids = [f"d{i}" for i in range(len(scores))]
    judgements = {f"d{i}": g for i, g in graded.items()}
    depth = k + extra
    ranked = [ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]

    assert expected_metrics(scores, ids, judgements, k=k, depth=depth) == {
        f"ndcg@{k}": ndcg_at_k(ranked[:depth], judgements, k),
        f"recall@{depth}": recall_at_k(ranked[:depth], judgements, depth),
        f"rr@{k}": reciprocal_rank_at_k(ranked[:depth], judgements, k),
    }


def test_a_tie_at_the_cutoff_gives_the_mean_of_the_two_orders() -> None:
    """d1 and d2 tie for position 2; d2 is relevant, so it is there half the time."""
    metrics = expected_metrics([0.9, 0.5, 0.5], ["d0", "d1", "d2"], {"d2": 1}, k=2, depth=2)

    assert metrics["recall@2"] == 0.5
    assert metrics["rr@2"] == 0.25
    assert metrics["ndcg@2"] == float(Fraction(discount(2)) / 2)


def test_dcg_terms_stop_at_the_group_that_starts_on_the_cutoff() -> None:
    """d1 and d2 tie at rank 2 with k=1, so only d0's group enters the DCG."""
    exact = exact_expectation([0.9, 0.5, 0.5], ["d0", "d1", "d2"], {"d0": 2, "d1": 1}, k=1, depth=3)

    assert exact.dcg_terms == (Fraction(2) * Fraction(discount(1)),)
    assert (exact.hits, exact.rr) == (Fraction(2), Fraction(1))


def test_a_relevant_document_outside_the_list_still_counts() -> None:
    metrics = expected_metrics([0.5, 0.5], ["d0", "d1"], {"d0": 1, "gone": 1}, k=1, depth=2)

    assert metrics["recall@2"] == 0.5
    assert metrics["rr@1"] == 0.5
    assert metrics["ndcg@1"] == 0.5


def test_nothing_relevant_leaves_every_metric_undefined() -> None:
    assert expected_metrics([0.5, 0.1], ["d0", "d1"], {"d0": 0, "d1": -1}, k=1, depth=2) == {
        "ndcg@1": None,
        "recall@2": None,
        "rr@1": None,
    }


def test_an_empty_list_scores_zero_where_something_is_relevant() -> None:
    assert expected_metrics([], [], {"d0": 1}, k=3, depth=4) == {
        "ndcg@3": 0.0,
        "recall@4": 0.0,
        "rr@3": 0.0,
    }


def test_groups_join_signed_zeros_and_list_values_highest_first() -> None:
    assert score_groups([0.0, 0.5, -0.0, 0.5, -1.0]) == [(1, 3), (0, 2), (4,)]
    assert score_groups([]) == []


@pytest.mark.parametrize(
    ("scores", "ids", "k", "depth", "message"),
    [
        ([0.5], ["d0"], 0, 1, "need 1 <= k <= depth, got k=0, depth=1"),
        ([0.5], ["d0"], 2, 1, "need 1 <= k <= depth, got k=2, depth=1"),
        ([0.5, 0.1], ["d0"], 1, 1, "2 scores for 1 documents"),
        ([math.nan], ["d0"], 1, 1, "a NaN score has no place in a ranking"),
        ([0.5, 0.1], ["d0", "d0"], 1, 1, "d0"),
    ],
)
def test_bad_input_is_refused(
    scores: list[float], ids: list[str], k: int, depth: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        expected_metrics(scores, ids, {"d0": 1}, k=k, depth=depth)


def _grid(rows: dict[str, list[float]]) -> QueryGrid:
    ids = ("d0", "d1", "d2")
    table = AttributeTable.from_records([{"doc_id": d} for d in ids])
    return QueryGrid(
        queries=tuple(
            EvaluatedQuery(
                query_id=qid,
                mode=QueryMode.TEXT,
                scores=tuple(scores),
                table=table,
                candidate_ids=ids,
                target=None,
                n_excluded=0,
            )
            for qid, scores in rows.items()
        ),
        mode=QueryMode.TEXT,
        aggregation=ProfileAggregation.TEXT_CONCAT,
        min_interactions=0,
        exclude_profile_items=False,
        n_users=0,
        n_degenerate=0,
        n_zero_vector=0,
    )


def test_means_cover_every_judged_query_and_score_a_missing_one_zero() -> None:
    """q1 ties d1 and d2; q2 is unscored; q3 judges nothing relevant."""
    qrels = {"q1": {"d2": 1}, "q2": {"d0": 1}, "q3": {"d0": 0}}
    means = expected_means(
        _grid({"q1": [0.9, 0.5, 0.5], "q3": [0.1, 0.2, 0.3]}), qrels, k=2, depth=2
    )

    assert means["recall@2"].mean == 0.25, "q1 expects 0.5, the missing q2 scores 0"
    assert (means["rr@2"].n_queries, means["rr@2"].n_undefined) == (2, 1)
    assert means["ndcg@2"].mean == float(Fraction(discount(2)) / 2) / 2
