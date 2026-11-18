"""nDCG, recall and reciprocal rank against qrels, and their mean over queries.

Expected values are worked by hand from the trec_eval definitions: linear gain,
discount ``1 / log2(i + 1)``, and a grade of zero or below is not relevant.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from tfidf_stability.evaluation.metrics import (
    MeanMetric,
    discount,
    mean_over_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from tfidf_stability.utils.numerics import ulps_between
from tfidf_stability.utils.validation import DuplicateIdentifierError

_GRADED = {"a": 2, "b": 1, "c": 0}


# ---------------------------------------------------------------------------
# The discount
# ---------------------------------------------------------------------------
def test_the_discount_is_exact_where_the_ratio_is() -> None:
    """``ln 4`` rounds to exactly twice the rounded ``ln 2``, so rank 3 gives 0.5."""
    assert discount(1) == 1.0
    assert discount(3) == 0.5


def test_the_discount_is_within_one_ulp_of_the_platform_value() -> None:
    for rank in range(1, 201):
        assert ulps_between(discount(rank), 1.0 / math.log2(rank + 1)) <= 1, rank


def test_the_discount_falls_with_rank() -> None:
    values = [discount(rank) for rank in range(1, 50)]
    assert all(a > b for a, b in pairwise(values))


def test_a_rank_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="rank must be at least 1, got 0"):
        discount(0)


# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------
def test_ndcg_by_hand_with_graded_judgements() -> None:
    """c (grade 0) at rank 1, b (grade 1) at 2, a (grade 2) at 3.

    DCG = 1/log2 3 + 2/2 and IDCG = 2 + 1/log2 3.
    """
    expected = (1 / math.log2(3) + 1.0) / (2.0 + 1 / math.log2(3))

    assert ndcg_at_k(["c", "b", "a", "x"], _GRADED, 3) == pytest.approx(expected, rel=1e-15)


def test_the_ideal_order_scores_exactly_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], _GRADED, 3) == 1.0
    assert ndcg_at_k(["a", "b"], _GRADED, 10) == 1.0, "a list shorter than k"


def test_the_order_given_is_scored_never_re_sorted() -> None:
    swapped = ndcg_at_k(["b", "a"], _GRADED, 2)

    assert swapped is not None
    assert swapped < 1.0, "the same documents in another order score less"


def test_the_ideal_list_is_cut_at_k() -> None:
    """Three relevant documents, k = 1: the ideal is the single best grade."""
    judgements = {"a": 1, "b": 3, "c": 2}

    assert ndcg_at_k(["b"], judgements, 1) == 1.0
    assert ndcg_at_k(["c"], judgements, 1) == pytest.approx(2 / 3, rel=1e-15)


def test_an_unjudged_document_gains_nothing() -> None:
    assert ndcg_at_k(["x", "y"], _GRADED, 2) == 0.0


@pytest.mark.parametrize("judgements", [{}, {"a": 0, "b": -1}])
def test_without_a_relevant_document_every_metric_is_undefined(
    judgements: dict[str, int],
) -> None:
    ranked = ["a", "b"]

    assert ndcg_at_k(ranked, judgements, 2) is None
    assert recall_at_k(ranked, judgements, 2) is None
    assert reciprocal_rank_at_k(ranked, judgements, 2) is None


# ---------------------------------------------------------------------------
# Recall and reciprocal rank
# ---------------------------------------------------------------------------
def test_recall_counts_relevant_documents_inside_the_cutoff() -> None:
    assert recall_at_k(["a", "x", "b"], _GRADED, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], _GRADED, 3) == 1.0
    assert recall_at_k([], _GRADED, 5) == 0.0


def test_reciprocal_rank_is_that_of_the_first_relevant_document() -> None:
    assert reciprocal_rank_at_k(["c", "x", "b", "a"], _GRADED, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(["a"], _GRADED, 1) == 1.0
    assert reciprocal_rank_at_k(["c", "x", "b"], _GRADED, 2) == 0.0, "b falls past k"


def test_binary_judgements_are_relevant_at_grade_one() -> None:
    """SciFact's qrels are all grade 1, so no metric may need a grade above it."""
    binary = {"a": 1, "b": 1}

    assert reciprocal_rank_at_k(["x", "a"], binary, 2) == 0.5
    assert recall_at_k(["a", "x"], binary, 2) == 0.5
    assert ndcg_at_k(["a", "b"], binary, 2) == 1.0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k, reciprocal_rank_at_k])
def test_every_metric_refuses_a_cutoff_below_one(metric: object) -> None:
    with pytest.raises(ValueError, match="k must be at least 1, got 0"):
        metric(["a"], _GRADED, 0)  # type: ignore[operator]


@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k, reciprocal_rank_at_k])
def test_every_metric_refuses_a_document_ranked_twice(metric: object) -> None:
    with pytest.raises(DuplicateIdentifierError, match="'a' appears at positions 0 and 2"):
        metric(["a", "b", "a"], _GRADED, 3)  # type: ignore[operator]


def test_a_repeat_past_the_cutoff_is_not_inspected() -> None:
    assert recall_at_k(["a", "b", "a"], _GRADED, 2) == 1.0


# ---------------------------------------------------------------------------
# The mean over queries
# ---------------------------------------------------------------------------
def test_the_mean_leaves_out_undefined_values_and_counts_them() -> None:
    mean = mean_over_queries({"q1": 1.0, "q2": None, "q3": 0.5})

    assert mean == MeanMetric(mean=0.75, n_queries=2, n_undefined=1)


def test_the_mean_is_exactly_rounded_once() -> None:
    """fsum sees 0.1 + 0.2 + 0.3 exactly; a running sum would round twice."""
    values = {"a": 0.1, "b": 0.2, "c": 0.3}

    assert mean_over_queries(values).mean == math.fsum(values.values()) / 3


def test_a_mean_with_nothing_defined_is_refused() -> None:
    with pytest.raises(ValueError, match="no defined value among 2 queries"):
        mean_over_queries({"q1": None, "q2": None})
