"""The tie rules trec_eval, Anserini and FAISS apply, and E8's payload.

Each stage is checked on its own against hand-worked values, then the rules are
checked against the orders they claim, and the driver against a small store.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.hidden_rules import (
    HiddenRule,
    _float32_up,
    anserini_adjust,
    anserini_round,
    java_round,
    rank_anserini,
    rank_faiss,
    rank_trec_eval,
    rule_means,
    run_hidden_rules,
    trec_text,
)
from tfidf_stability.evaluation.metrics import (
    MeanMetric,
    mean_over_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from tfidf_stability.evaluation.retrieval import evaluate_retrieval
from tfidf_stability.evaluation.score_store import ScoreStore, StoreKind
from tfidf_stability.profiles import QueryMode
from tfidf_stability.profiles.user_profile import ProfileAggregation
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.utils.numerics import float32_round

#: UTF-8 order, so "d10" sorts between "d1" and "d2".
_IDS = ("d1", "d10", "d2", "d3", "d4")
_LONG_MAX = 2**63 - 1


def _grid(rows: dict[str, list[float]], ids: tuple[str, ...] = _IDS) -> QueryGrid:
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


# ---------------------------------------------------------------------------
# Java's rounding and printing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("y", "expected"),
    [
        (2.5, 3),
        (-2.5, -2),
        (0.5, 1),
        (-0.5, 0),
        (-1.5, -1),
        (0.49999999999999994, 0),
        (-0.49999999999999994, 0),
        (-(2.0**-60), 0),
        (2.0**60 + 0.0, 2**60),
        (1e300, _LONG_MAX),
        (-1e300, -(2**63)),
    ],
)
def test_java_round_takes_a_half_toward_positive_infinity(y: float, expected: int) -> None:
    """``floor(y + 0.5)`` would give 1 for 0.49999999999999994; Java gives 0."""
    assert java_round(y) == expected


@pytest.mark.parametrize("y", [math.nan, math.inf, -math.inf])
def test_java_round_refuses_a_value_it_does_not_model(y: float) -> None:
    with pytest.raises(ValueError, match="is not modelled"):
        java_round(y)


def test_the_adjuster_rounds_a_half_up_in_binary64() -> None:
    """0.03125 times 1e4 is exactly 312.5: Java rounds it up, Python's round down."""
    assert anserini_round(0.03125) == float32_round(0.0313)
    assert anserini_round(-0.03125) == float32_round(-0.0312)
    assert anserini_round(float32_round(0.12345)) == float32_round(0.1235)
    assert round(312.5) == 312


@pytest.mark.property
@settings(max_examples=300, deadline=None)
@given(
    st.floats(width=32, allow_nan=False, allow_infinity=False),
    st.floats(width=32, allow_nan=False, allow_infinity=False),
)
def test_rounding_to_1e4_never_reverses_two_float32_scores(a: float, b: float) -> None:
    lo, hi = sorted((a, b))

    assert anserini_round(lo) <= anserini_round(hi)


@pytest.mark.parametrize(
    ("score", "printed"),
    [
        (0.25, "0.250000"),
        (5e-07, "0.000001"),
        (-5e-07, "-0.000001"),
        (1.5e-06, "0.000002"),
        (-0.0, "-0.000000"),
        (float32_round(0.1), "0.100000"),
        (1e22, "10000000000000000000000.000000"),
    ],
)
def test_the_run_writer_rounds_the_shortest_decimal_half_up(score: float, printed: str) -> None:
    """5e-07 lies just below its decimal in binary, so C's printf gives 0.000000."""
    assert trec_text(score) == printed


@pytest.mark.parametrize("score", [math.nan, math.inf])
def test_the_run_writer_refuses_a_value_it_does_not_model(score: float) -> None:
    with pytest.raises(ValueError, match="is not modelled"):
        trec_text(score)


# ---------------------------------------------------------------------------
# The rules, one stage at a time
# ---------------------------------------------------------------------------
def test_a_tied_run_drops_by_one_step_per_place() -> None:
    adjusted = anserini_adjust([("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.25)])

    assert [trec_text(s) for _, s in adjusted] == ["0.500000", "0.499999", "0.499998", "0.250000"]
    assert [d for d, _ in adjusted] == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    ("first", "printed"), [(1e-4, ["0.000100", "-0.000001"]), (2e-4, ["0.000200", "0.000000"])]
)
def test_a_score_exactly_the_tie_gap_below_joins_its_run(first: float, printed: list[str]) -> None:
    """``1e-4f`` minus zero is the gap itself, so zero joins the run and steps down."""
    adjusted = anserini_adjust([("a", float32_round(first)), ("b", 0.0)])

    assert [trec_text(s) for _, s in adjusted] == printed


def test_trec_eval_breaks_a_tie_by_docno_descending_in_bytes() -> None:
    assert rank_trec_eval([0.5, 0.5, 0.5, 0.9], ["d1", "d10", "d2", "d0"], 4) == [
        "d0",
        "d2",
        "d10",
        "d1",
    ]


def test_anserini_keeps_lucene_order_for_small_scores() -> None:
    """Score descending, then docid ascending: trec_eval's tie order reversed."""
    scores = [0.5, 0.25, 0.5, 0.25]
    ids = ["d0", "d1", "d2", "d3"]

    assert rank_anserini(scores, ids, 4, matched_only=True) == ["d0", "d2", "d1", "d3"]
    assert rank_trec_eval(scores, ids, 4) == ["d2", "d0", "d3", "d1"]


@pytest.mark.parametrize(
    ("score", "ranked"), [(31.0, ["d0", "d1"]), (32.0, ["d0", "d1"]), (33.0, ["d1", "d0"])]
)
def test_past_32_the_step_is_lost_and_docno_descending_decides(
    score: float, ranked: list[str]
) -> None:
    """A float32 ulp is 2**-19 below 32, 2**-18 above, so 1e-6 is lost past 32."""
    adjusted = anserini_adjust([("d0", score), ("d1", score)])

    assert (adjusted[1][1] == score) is (score > 32.0)
    assert rank_anserini([score, score], ["d0", "d1"], 2, matched_only=True) == ranked


def test_a_float32_cast_creates_a_tie_that_lucene_order_breaks() -> None:
    """Two binary64 scores one ulp apart share a float32 value."""
    above = math.nextafter(0.1, 1.0)
    scores = [0.1, above]

    assert float32_round(0.1) == float32_round(above)
    assert rank_trec_eval(scores, ["d0", "d1"], 2) == ["d1", "d0"]
    assert rank_anserini(scores, ["d0", "d1"], 2, matched_only=True) == ["d0", "d1"]


def test_matched_only_drops_zero_scores() -> None:
    """Kept, the zeros tie: the second is stepped below zero and printed negative."""
    scores = [0.0, 0.5, 0.0]
    ids = ["d0", "d1", "d2"]

    assert rank_anserini(scores, ids, 3, matched_only=True) == ["d1"]
    assert rank_anserini(scores, ids, 3, matched_only=False) == ["d1", "d0", "d2"]


def test_faiss_scans_by_insertion_and_reads_ties_by_id_descending() -> None:
    """Insertion ids c=0, a=1, b=2, d=3: b never enters, then d evicts c, the lowest."""
    scores = [0.5, 0.5, 0.5, 0.9]
    ids = ["a", "b", "c", "d"]
    insertion = {"c": 0, "a": 1, "b": 2, "d": 3}

    assert rank_faiss(scores, ids, insertion, 2) == ["d", "a"]
    assert rank_faiss(scores, ids, insertion, 4) == ["d", "b", "a", "c"]


@pytest.mark.parametrize(
    ("scores", "ids", "depth", "message"),
    [
        ([0.5], ["d0"], 0, "depth must be at least 1, got 0"),
        ([0.5, 0.1], ["d0"], 1, "2 scores for 1 documents"),
    ],
)
def test_bad_lists_are_refused(
    scores: list[float], ids: list[str], depth: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rank_trec_eval(scores, ids, depth)


def test_faiss_refuses_a_document_with_no_insertion_id() -> None:
    with pytest.raises(ValueError, match="no insertion id for 'd1'"):
        rank_faiss([0.5, 0.5], ["d0", "d1"], {"d0": 0}, 1)


# ---------------------------------------------------------------------------
# FAISS's handlers, against faiss-cpu 1.16.0 IndexFlatIP(1).search
# ---------------------------------------------------------------------------
#: ``numeric_limits<float>::max()``.
_F32_MAX = 3.4028234663852886e38


def _faiss_labels(by_label: list[float], depth: int) -> list[int]:
    """rank_faiss over documents listed in reverse, mapped back to FAISS labels.

    ``by_label[j]`` scores the document FAISS stores as label ``j``.
    """
    insertion = {f"doc{j}": j for j in range(len(by_label))}
    docs = list(reversed(insertion))
    ranked = rank_faiss([by_label[insertion[d]] for d in docs], docs, insertion, depth)
    return [insertion[d] for d in ranked]


def _lexical(n: int) -> list[float]:
    """Labels 150 to 159 score ``1 + j / 1024``; every other label scores zero."""
    return [1.0 + j / 1024 if 150 <= j < 160 else 0.0 for j in range(n)]


def _digest(labels: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, labels)).encode()).hexdigest()


_MATCHED = list(range(159, 149, -1))


@pytest.mark.parametrize(
    ("n", "depth", "zeros"),
    [
        (5000, 99, range(98, 9, -1)),
        (5000, 100, range(89, -1, -1)),
        (208, 100, range(99, 9, -1)),
        (209, 100, range(89, -1, -1)),
    ],
)
def test_faiss_keeps_the_ties_its_handler_keeps(n: int, depth: int, zeros: range) -> None:
    """The heap evicts the lowest ids; a partitioned reservoir keeps the first.

    The reservoir holds 208 at depth 100, so 208 documents never partition it.
    """
    assert _faiss_labels(_lexical(n), depth) == [*_MATCHED, *zeros]


def test_a_reservoir_never_filled_reads_score_then_id_descending() -> None:
    expected = [*range(47, -1, -3), *range(49, 0, -3), *range(48, -1, -3)]

    assert _faiss_labels([float(j % 3) for j in range(50)], 100) == expected


@pytest.mark.parametrize("depth", [5, 100])
def test_faiss_admits_only_scores_above_float32_lowest(depth: int) -> None:
    scores = [-math.inf, 0.5, math.nan, -_F32_MAX, -1e300, 0.25]

    assert _faiss_labels(scores, depth) == [1, 5]


def test_a_partition_with_nothing_to_sample_keeps_the_first_q_min() -> None:
    """205 of the first 208 sit at float32's max, above every value sampled."""
    scores = [_F32_MAX] * 208 + [2.0] * 32
    for j in (0, 104, 207):
        scores[j] = 1.0

    assert _faiss_labels(scores, 100) == list(range(100, 0, -1))


def test_a_partition_inside_its_window_keeps_every_value_above_the_median() -> None:
    """120 of the first 208 lie above the zero median, between 100 and 154: all stay."""
    scores = [0.0] * 260
    for j in range(1, 18):
        scores[j] = 3.0
    for j in range(105, 208):
        scores[j] = 1.0 + j / 1024
    for j in range(208, 260):
        scores[j] = 0.5

    assert _faiss_labels(scores, 100) == [*range(17, 0, -1), *range(207, 124, -1)]


def test_a_sample_reads_no_value_at_float32_max() -> None:
    """160 of the first 208 sit at float32's max, which no sample reads.

    A sample above zero finds only 1; the bisection takes it as its threshold, then
    finds nothing above it to sample and keeps the first q_min.
    """
    scores = [0.0] * 208 + [0.5] * 10
    for j in range(1, 161):
        scores[j] = _F32_MAX
    scores[200] = 1.0

    assert _faiss_labels(scores, 100) == list(range(100, 0, -1))


def _ascending_case() -> list[float]:
    """Pairs the sampling stride meets in ascending order, then two top values.

    At depth 1000 the bisection climbs one pair a step for 199 steps, reaches 5000,
    with only 6000 above it, and spends its last step below 5000.
    """
    capacity = 2000
    head = [float(j) for j in range(2, 200) for _ in range(2)] + [5000.0, 5000.0, 6000.0]
    tail = [300.0 + r / 8 for r in range(capacity)]
    top: list[float | None] = [None] * capacity
    for pos in (0, capacity // 2, capacity - 1):
        top[pos] = 1.0
    fill = iter([*head, *tail])
    for i in range(1, capacity):
        pos = i * 6700417 % capacity
        if top[pos] is None:
            top[pos] = next(fill)
    return [float(v) for v in top if v is not None] + [0.5] * 5


def test_a_bisection_out_of_steps_near_the_top_still_records_q_min() -> None:
    """One value lies above the last threshold tested, so FAISS records 1000 stored.

    The threshold moves one float32 up from 300.125; the first 1000 labels above it stay.
    """
    scores = _ascending_case()
    kept = [j for j, s in enumerate(scores) if s > 300.125][:1000]

    labels = _faiss_labels(scores, 1000)

    assert len(labels) == 1000
    assert labels == sorted(kept, key=lambda j: (scores[j], j), reverse=True)
    assert scores[labels[0]] == 6000.0


def _bisection_case() -> list[float]:
    """Distinct values the sampling stride meets in descending order, then a tail.

    The bisection at depth 500 descends two values a step and stops after 200 steps.
    """
    capacity = 1008
    top = [0.0] * capacity
    top[0], top[capacity // 2], top[-1] = 1008.0, 1007.0, 1006.0
    rank = 3
    for i in range(1, capacity):
        pos = i * 6700417 % capacity
        if pos not in (capacity // 2, capacity - 1):
            top[pos] = float(capacity - rank)
            rank += 1
    return top + [j % 7 + 0.5 for j in range(60)]


def test_a_bisection_out_of_steps_records_q_min_over_stale_slots() -> None:
    """FAISS returns 41 labels twice: slots past the kept values hold copies."""
    labels = _faiss_labels(_bisection_case(), 500)

    assert len(labels) == 500
    assert len(labels) - len(set(labels)) == 41
    assert labels[:5] == [0, 504, 1007, 241, 482]
    assert labels[-5:] == [419, 484, 440, 461, 417]
    assert _digest(labels) == "21773ace02d9e210aa5129ad13c740d2c4991f96acfd562d03e3113e90bcc515"


def test_a_bisection_out_of_steps_moves_its_threshold_one_float32_up() -> None:
    """The steps end at 606, so 606 + 2**-14 is the threshold and stays out.

    Label 1008 forces the partition and is stored regardless; label 1009 is refused.
    """
    scores = [*_bisection_case()[:1008], 1000.5, 606 + 2**-14, *[0.5] * 58]
    labels = _faiss_labels(scores, 500)

    assert 1008 in labels
    assert 1009 not in labels
    assert _digest(labels) == "ad6ba11cb4dc65285cb596d06a8f3c966457208d11738a25d9ed0296e628d1e3"


def test_a_bisection_out_of_steps_keeps_ties_at_the_moved_threshold() -> None:
    """Label 1 scores 606 + 2**-14, the moved threshold, and is kept as a tie."""
    scores = _bisection_case()[:1008]
    scores[1] = 606 + 2**-14
    labels = _faiss_labels([*scores, 1000.5, *[0.5] * 59], 500)

    assert labels.count(1) == 1
    assert _digest(labels) == "052985833020376a2b20ae889befbd2751f3e1724e13ca09674660942b91abd3"


@pytest.mark.parametrize(
    ("x", "up"),
    [
        (1.0, 1.0 + 2**-23),
        (-1.0, -1.0 + 2**-24),
        (0.0, 2.0**-149),
        (-0.0, 2.0**-149),
        (-(2.0**-149), -0.0),
        (0.1, float32_round(0.1) + 2**-27),
        (_F32_MAX, math.inf),
        (1e39, math.inf),
    ],
)
def test_the_fallback_step_is_nextafterf_toward_infinity(x: float, up: float) -> None:
    stepped = _float32_up(x)

    assert stepped == up
    assert math.copysign(1.0, stepped) == math.copysign(1.0, up)


@pytest.mark.parametrize(
    ("score", "n", "depth", "digest"),
    [
        (
            lambda j: float((j * j) % 11 < 3) + float(j % 97 == 0),
            2000,
            100,
            "b4d7c9e90d367ae50fd1707ac4783087e275c263acc36c91a3c1da4856beba89",
        ),
        (
            lambda j: float((j * 7919) % 53 == 0) * (1 + j % 3),
            3000,
            101,
            "e619b051e7831fe01aab1de9db7584c3dd68e906577065a05254c2a5371656e0",
        ),
    ],
)
def test_the_reservoir_reproduces_faiss_where_a_heap_would_not(
    score: Callable[[int], float], n: int, depth: int, digest: str
) -> None:
    """At these depths a heap of the same size keeps other ties than FAISS returns."""
    assert _digest(_faiss_labels([score(j) for j in range(n)], depth)) == digest


def _hashed(n: int, levels: int, salt: int, density: int) -> list[float]:
    """Integer scores from a 32-bit multiplicative hash, zero outside ``density`` %."""
    scores = []
    for j in range(n):
        h = (j * 2654435761 + salt) % 2**32
        hit = (h >> 24) * 100 < density * 256
        scores.append(float((h >> 8) % levels + 1) if hit else 0.0)
    return scores


@pytest.mark.parametrize(
    ("case", "depth", "digest"),
    [
        ((300, 2, 0, 10), 104, "fdf520fd770ffd14aa4871bfad685c6f6b729bde9fc05ae1f60ade227a74137a"),
        ((300, 3, 0, 100), 100, "4b39fcf76b97a92741659027d164e58f57a3819289654f2996f8290b94aa0069"),
        ((700, 3, 0, 40), 100, "2336936457a6bdddcff46dd8795221a9d94bce44f681424111a3a8026219f985"),
        ((700, 3, 0, 40), 104, "e3a8d2551b6f6273e50d78f61b108344f221f51676f1bb1e24aefbbfb31f77f7"),
        ((700, 5, 0, 100), 150, "2e3df19d671cca3b681594978dfddf5c8abc510371e9224357840ae08ea599f8"),
        ((450, 4, 1, 100), 120, "1236215d05fe0073c22fcce4774e76fc7b1601e3e1c43bc98988d223daf74b91"),
        (
            (2500, 12, 5, 40),
            101,
            "7dbad5779f5f1931173c46aa4a5f10e0400c10bbbb4d84e0a7c5bb07e36340ad",
        ),
    ],
)
def test_the_reservoir_reproduces_faiss_on_hashed_lists(
    case: tuple[int, int, int, int], depth: int, digest: str
) -> None:
    """Each list partitions the reservoir at least once at its depth."""
    assert _digest(_faiss_labels(_hashed(*case), depth)) == digest


# ---------------------------------------------------------------------------
# Means over a grid
# ---------------------------------------------------------------------------
_ROWS = {
    "q1": [0.5, 0.5, 0.25, 0.5, 0.0],
    "q2": [0.0, 0.0, 0.0, 0.0, 0.0],
    "q4": [0.75, 0.25, 0.25, 0.25, 0.75],
}
_QRELS = {
    "q1": {"d10": 1, "d3": 2},
    "q2": {"d2": 1},
    "q3": {"d1": 1},
    "q4": {"d1": 1, "d2": 1},
}


@pytest.mark.parametrize(("k", "depth"), [(1, 1), (2, 3), (3, 5)])
def test_the_trec_eval_rule_reproduces_id_desc_bit_for_bit(k: int, depth: int) -> None:
    """q1 and q4 tie across the cutoff, q2 scores only zeros, q3 is degenerate."""
    grid = _grid(_ROWS)
    report = evaluate_retrieval(grid, _QRELS, _IDS, k=k, depth=depth, seeds=0)
    insertion = {d: i for i, d in enumerate(_IDS)}
    means = rule_means(grid, _QRELS, _IDS, k=k, depth=depth, insertion=insertion)

    assert means[str(HiddenRule.TREC_EVAL)] == report.means["id_desc"]
    assert set(means) == {str(rule) for rule in list(HiddenRule)}


def _means_of(ranked: dict[str, list[str]]) -> dict[str, MeanMetric]:
    """Means at k=2, depth=3 of hand-ranked lists; q3 ranks nothing."""
    lists = {q: ranked.get(q, []) for q in _QRELS}
    return {
        "ndcg@2": mean_over_queries({q: ndcg_at_k(r, _QRELS[q], 2) for q, r in lists.items()}),
        "recall@3": mean_over_queries({q: recall_at_k(r, _QRELS[q], 3) for q, r in lists.items()}),
        "rr@2": mean_over_queries(
            {q: reciprocal_rank_at_k(r, _QRELS[q], 2) for q, r in lists.items()}
        ),
    }


_UTF8 = {d: i for i, d in enumerate(_IDS)}
_REVERSED = {d: i for i, d in enumerate(reversed(_IDS))}
#: What each rule ranks at depth 3. Lucene lists q4's 0.75 tie d1 first, then d4 a step
#: below; trec_eval reads the printed scores, so the step survives.
_RANKED = {
    "trec_eval": {"q1": ["d3", "d10", "d1"], "q2": ["d4", "d3", "d2"], "q4": ["d4", "d1", "d3"]},
    "faiss_utf8": {"q1": ["d3", "d10", "d1"], "q2": ["d2", "d10", "d1"], "q4": ["d4", "d1", "d2"]},
    "faiss_reversed": {
        "q1": ["d1", "d10", "d3"],
        "q2": ["d2", "d3", "d4"],
        "q4": ["d1", "d4", "d2"],
    },
    "anserini": {"q1": ["d1", "d10", "d3"], "q2": ["d1", "d10", "d2"], "q4": ["d1", "d4", "d10"]},
    "anserini_matched_only": {"q1": ["d1", "d10", "d3"], "q4": ["d1", "d4", "d10"]},
}


@pytest.mark.parametrize(
    ("insertion", "faiss"), [(_UTF8, "faiss_utf8"), (_REVERSED, "faiss_reversed")]
)
def test_each_rule_ranks_the_grid_as_its_tool_does(insertion: dict[str, int], faiss: str) -> None:
    """q1 and q4 tie across the cutoff; matched_only drops q2's zeros."""
    means = rule_means(_grid(_ROWS), _QRELS, _IDS, k=2, depth=3, insertion=insertion)

    assert means["trec_eval"] == _means_of(_RANKED["trec_eval"])
    assert means["faiss_insertion"] == _means_of(_RANKED[faiss])
    assert means["anserini"] == _means_of(_RANKED["anserini"])
    assert means["anserini_matched_only"] == _means_of(_RANKED["anserini_matched_only"])
    assert means["anserini"]["recall@3"].mean == 0.625
    assert means["anserini_matched_only"]["recall@3"].mean == 0.375


def test_the_faiss_rule_follows_the_insertion_order() -> None:
    utf8, reverse = (
        rule_means(_grid(_ROWS), _QRELS, _IDS, k=2, depth=3, insertion=order)["faiss_insertion"]
        for order in (_UTF8, _REVERSED)
    )

    assert utf8["ndcg@2"].mean != reverse["ndcg@2"].mean
    assert utf8["ndcg@2"] != _means_of(_RANKED["trec_eval"])["ndcg@2"]


def test_a_missing_query_ranks_nothing_under_every_rule() -> None:
    insertion = {d: i for i, d in enumerate(_IDS)}
    means = rule_means(_grid({}), {"q3": {"d1": 1}}, _IDS, k=1, depth=2, insertion=insertion)

    assert {m["rr@1"].mean for m in means.values()} == {0.0}


@pytest.mark.parametrize(
    ("grid", "k", "depth", "message"),
    [
        (_grid({}), 0, 1, "need 1 <= k <= depth, got k=0, depth=1"),
        (_grid({}), 3, 2, "need 1 <= k <= depth, got k=3, depth=2"),
        (_grid({"q1": [0.5]}, ("d1",)), 1, 1, "query 'q1' does not rank the whole corpus"),
    ],
)
def test_rule_means_refuses_a_bad_cutoff_or_a_partial_list(
    grid: QueryGrid, k: int, depth: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rule_means(grid, _QRELS, _IDS, k=k, depth=depth, insertion={})


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
_RECORDS = [{"doc_id": d, "text": f"text {d}"} for d in _IDS]
_ABOVE = math.nextafter(0.1, 1.0)


def _store(tmp_path: Path) -> ScoreStore:
    store = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS, create=True)
    lexical = {"q1": _ROWS["q1"], "q2": _ROWS["q2"], "q4": _ROWS["q4"]}
    byte = {**lexical, "q4": [0.75, 0.25, 0.8, 0.25, 0.75]}
    cast_tie = {**lexical, "q1": [0.1, _ABOVE, 0.0, 0.0, 0.0]}
    dense = {**lexical, "q1": [0.25, 0.5, 0.5, 0.5, 0.5]}
    adds: list[tuple[str, dict[str, list[float]], StoreKind, str | None]] = [
        ("bm25-exact_n1.exact", lexical, StoreKind.LEXICAL, None),
        ("bm25-exact_n1.naive", lexical, StoreKind.LEXICAL, "bm25-exact_n1.exact"),
        ("bm25-lucene_byte_n1.naive", byte, StoreKind.LEXICAL, None),
        ("bm25-exact_n2.naive", lexical, StoreKind.LEXICAL, None),
        ("tfidf_n2.naive", cast_tie, StoreKind.LEXICAL, None),
        ("splade.exact", lexical, StoreKind.SPARSE, None),
        ("dense-fp64.exact", dense, StoreKind.MATRIX, None),
        ("dense-fp32.run", dense, StoreKind.RUN, None),
    ]
    for label, rows, kind, reference in adds:
        store = store.add(
            label, rows, kind=kind, params={}, degenerate=["q3"], reference_label=reference
        )
    return store


_E8_LABELS = [
    "bm25-exact_n1.naive",
    "bm25-exact_n2.naive",
    "bm25-lucene_byte_n1.naive",
    "dense-fp32.run",
    "dense-fp64.exact",
    "tfidf_n2.naive",
]


def test_the_payload_covers_each_label_e6_measures(tmp_path: Path) -> None:
    """Naive-policy labels and every run or matrix; other policies are left out."""
    store = _store(tmp_path)
    payload = run_hidden_rules(
        store, _QRELS, cutoffs=(1, 3), depth=2, insertion_order=_IDS, insertion_source="test"
    )

    assert payload["insertion"] == {"source": "test", "n_documents": 5}
    assert list(payload["labels"]) == _E8_LABELS
    row = payload["labels"]["bm25-exact_n1.naive"]["3"]
    assert row["depth"] == 3, "a cutoff past the depth ranks to the cutoff"
    neutral = row["neutral"]["ndcg@3"]["mean"]
    anserini = row["rules"]["anserini"]["ndcg@3"]
    assert anserini["delta_vs_neutral"] == anserini["mean"] - neutral
    assert set(row["rules"]) == {str(rule) for rule in list(HiddenRule)}


def test_the_float32_neutral_separates_the_cast(tmp_path: Path) -> None:
    """q1 of tfidf_n2 ties only once cast, so only the float32 neutral moves."""
    payload = run_hidden_rules(
        _store(tmp_path), _QRELS, cutoffs=(1,), depth=1, insertion_order=_IDS, insertion_source="x"
    )
    row = payload["labels"]["tfidf_n2.naive"]["1"]

    assert row["neutral"]["rr@1"]["mean"] != row["anserini_float32_neutral"]["rr@1"]["mean"]
    exact = payload["labels"]["bm25-exact_n1.naive"]["1"]
    assert exact["neutral"] == exact["anserini_float32_neutral"]


def test_the_length_rule_pairs_exact_with_byte_lengths(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = run_hidden_rules(
        store, _QRELS, cutoffs=(1,), depth=2, insertion_order=_IDS, insertion_source="x"
    )
    pair = ["bm25-exact_n1.naive", "bm25-lucene_byte_n1.naive"]
    neutral = [payload["labels"][label]["1"]["neutral"]["recall@2"]["mean"] for label in pair]
    asc = [
        evaluate_retrieval(store.grid(label), _QRELS, _IDS, k=1, depth=2, seeds=0)
        .means["id_asc"]["recall@2"]
        .mean
        for label in pair
    ]

    assert [row["labels"] for row in payload["length_rule"]] == [pair]
    deltas = payload["length_rule"][0]["cutoffs"]["1"]["recall@2"]
    assert deltas == {"neutral_delta": neutral[1] - neutral[0], "id_asc_delta": asc[1] - asc[0]}
    assert deltas["neutral_delta"] != 0.0


def test_every_spread_follows_the_shared_contract(tmp_path: Path) -> None:
    payload = run_hidden_rules(
        _store(tmp_path),
        _QRELS,
        cutoffs=(1, 3),
        depth=2,
        insertion_order=_IDS,
        insertion_source="x",
    )
    spreads = payload["spreads"]
    rule_rows = [s for s in spreads if "label" in s["source"]]
    length_rows = [s for s in spreads if s["source"].get("rule") == "lucene_byte_length"]

    assert len(rule_rows) == 6 * 2 * 4 * 3, "labels x cutoffs x rules x metrics"
    assert len(length_rows) == 2 * 3
    assert len(spreads) == len(rule_rows) + len(length_rows)
    assert all(
        set(s) == {"experiment", "metric", "spread", "source"}
        and s["experiment"] == "e8"
        and math.isfinite(s["spread"])
        and s["spread"] >= 0.0
        for s in spreads
    )
    assert {s["metric"] for s in spreads} == {
        "ndcg@1",
        "recall@2",
        "rr@1",
        "ndcg@3",
        "recall@3",
        "rr@3",
    }
    assert length_rows[0]["source"]["labels"] == [
        "bm25-exact_n1.naive",
        "bm25-lucene_byte_n1.naive",
    ]


def _delta(row: dict[str, Any], rule: str, metric: str) -> float:
    delta: float = row["rules"][rule][metric]["delta_vs_neutral"]
    return delta


def test_a_rule_spread_is_the_size_of_its_delta(tmp_path: Path) -> None:
    payload = run_hidden_rules(
        _store(tmp_path),
        _QRELS,
        cutoffs=(1, 3),
        depth=2,
        insertion_order=_IDS,
        insertion_source="x",
    )
    deltas = {
        (label, rule, metric): _delta(row, rule, metric)
        for label, by_cutoff in payload["labels"].items()
        for row in by_cutoff.values()
        for rule in row["rules"]
        for metric in row["rules"][rule]
    }
    means = {
        (label, rule, metric): row["rules"][rule][metric]["mean"]
        for label, by_cutoff in payload["labels"].items()
        for row in by_cutoff.values()
        for rule in row["rules"]
        for metric in row["rules"][rule]
    }
    spreads = {
        (s["source"]["label"], s["source"]["rule"], s["metric"]): s["spread"]
        for s in payload["spreads"]
        if "label" in s["source"]
    }

    assert spreads == {key: abs(delta) for key, delta in deltas.items()}
    assert any(spread != 0.0 and spread != abs(means[key]) for key, spread in spreads.items())
    assert spreads[("bm25-exact_n1.naive", "trec_eval", "rr@1")] == pytest.approx(11 / 120)


def test_a_length_spread_is_the_size_of_its_neutral_delta(tmp_path: Path) -> None:
    payload = run_hidden_rules(
        _store(tmp_path), _QRELS, cutoffs=(1,), depth=2, insertion_order=_IDS, insertion_source="x"
    )
    deltas = payload["length_rule"][0]["cutoffs"]["1"]
    spreads = {
        s["metric"]: s["spread"]
        for s in payload["spreads"]
        if s["source"].get("rule") == "lucene_byte_length"
    }

    assert spreads == {name: abs(d["neutral_delta"]) for name, d in deltas.items()}
    assert spreads["recall@2"] != 0.0
    assert spreads["recall@2"] != abs(deltas["recall@2"]["id_asc_delta"])


def test_two_insertion_orders_give_two_faiss_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    utf8, reverse = (
        run_hidden_rules(
            store, _QRELS, cutoffs=(1,), depth=1, insertion_order=order, insertion_source="x"
        )["labels"]["dense-fp32.run"]["1"]["rules"]
        for order in (_IDS, tuple(reversed(_IDS)))
    )

    assert utf8["faiss_insertion"] != reverse["faiss_insertion"]
    assert {r: utf8[r] for r in utf8 if r != "faiss_insertion"} == {
        r: reverse[r] for r in reverse if r != "faiss_insertion"
    }


@pytest.mark.parametrize("order", [_IDS[:4], (*_IDS[:4], "d1"), (*_IDS[:4], "d9"), (*_IDS, "d9")])
def test_an_insertion_order_that_is_not_a_permutation_is_refused(
    tmp_path: Path, order: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="not a permutation of the corpus"):
        run_hidden_rules(
            _store(tmp_path),
            _QRELS,
            cutoffs=(1,),
            depth=1,
            insertion_order=order,
            insertion_source="x",
        )


def test_a_rule_prints_as_its_value() -> None:
    assert str(HiddenRule.ANSERINI_MATCHED_ONLY) == "anserini_matched_only"
    assert HiddenRule("faiss_insertion") is HiddenRule.FAISS_INSERTION
