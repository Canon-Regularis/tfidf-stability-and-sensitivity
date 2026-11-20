"""E7: a top-k set changes only where its margin is at most 2 * eta.

Each grid is built by hand over three documents, so every margin and eta is exact
and each confusion cell is reached on purpose.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation import margin_noise as module
from tfidf_stability.evaluation.margin_noise import (
    Confusion,
    Falsification,
    Outcome,
    Scope,
    classify,
    falsifies,
    margin_noise,
    noise_pairs,
    run_margin_noise,
)
from tfidf_stability.evaluation.score_store import ScoreStore, StoreKind
from tfidf_stability.profiles import QueryMode
from tfidf_stability.profiles.user_profile import ProfileAggregation
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.ranker import Ranking, rank_top_k

_IDS = ("d0", "d1", "d2")
#: One ulp below 0.5, where the spacing is 2**-54.
_BELOW = math.nextafter(0.5, 0.0)
_ULP = Fraction(1, 2**54)


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


def _compare(ref: dict[str, list[float]], cur: dict[str, list[float]], k: int) -> Any:
    return margin_noise(_grid(ref), _grid(cur), k=k, reference_label="t.exact", label="t.naive")


def _cells(result: Any) -> dict[str, dict[str, int]]:
    return {scope: cell.as_dict() for scope, cell in result.confusion.items()}


@pytest.mark.parametrize(
    ("margin", "eta", "observed", "outcome", "falsified"),
    [
        (None, Fraction(0), False, Outcome.UNDEFINED, False),
        (None, Fraction(0), True, Outcome.UNDEFINED, True),
        (Fraction(1), Fraction(1, 2), True, Outcome.TP, False),
        (Fraction(1), Fraction(1, 2), False, Outcome.FP, False),
        (Fraction(0), Fraction(0), False, Outcome.FP, False),
        (Fraction(2), Fraction(1, 2), False, Outcome.TN, False),
        (Fraction(2), Fraction(1, 2), True, Outcome.FN, True),
    ],
)
def test_classify_reaches_every_cell(
    margin: Fraction | None, eta: Fraction, observed: bool, outcome: Outcome, falsified: bool
) -> None:
    assert classify(margin, eta, observed) is outcome
    assert falsifies(outcome, observed) is falsified
    assert str(outcome) == outcome.value


def test_identical_grids_have_no_eta_and_a_tie_at_the_cutoff_is_predicted() -> None:
    """q1 ties d1 and d2 at the cutoff; q2's signed zero moves nothing."""
    result = _compare(
        {"q1": [0.9, 0.5, 0.5], "q2": [0.9, 0.5, 0.0]},
        {"q1": [0.9, 0.5, 0.5], "q2": [0.9, 0.5, -0.0]},
        k=2,
    )

    assert result.eta == 0
    expected = {"tp": 0, "fp": 1, "tn": 1, "fn": 0, "undefined": 0}
    assert _cells(result) == {"global": expected, "query": expected}
    assert (result.falsifications, result.n_bound_disagreements) == ((), 0)


def test_a_one_ulp_move_that_flips_a_near_tie_is_predicted() -> None:
    result = _compare({"q1": [0.9, 0.5, _BELOW]}, {"q1": [0.9, _BELOW, 0.5]}, k=2)
    report = result.as_dict()

    assert result.eta == _ULP
    assert _cells(result)["global"] == {"tp": 1, "fp": 0, "tn": 0, "fn": 0, "undefined": 0}
    assert _cells(result)["query"] == _cells(result)["global"]
    assert (report["eta"], report["eta_hex"]) == (2.0**-54, (2.0**-54).hex())
    assert report["eta_exact"] == f"1/{2**54}"
    assert (report["reference"], report["variant"], report["k"], report["n_queries"]) == (
        "t.exact",
        "t.naive",
        2,
        1,
    )


def test_a_margin_of_exactly_two_eta_is_predicted_and_the_tie_break_decides() -> None:
    """The boundary scores meet at 0.25; d1 wins the tie, so q2's set changes."""
    meet = [0.9, 0.25, 0.25]
    result = _compare({"q1": [0.9, 0.5, 0.0], "q2": [0.9, 0.0, 0.5]}, {"q1": meet, "q2": meet}, k=2)

    assert result.eta == Fraction(1, 4)
    expected = {"tp": 1, "fp": 1, "tn": 0, "fn": 0, "undefined": 0}
    assert _cells(result) == {"global": expected, "query": expected}
    assert result.n_bound_disagreements == 0, "is_top_k_stable needs eta < m/2, as here"


def test_the_query_scope_uses_each_query_s_own_eta() -> None:
    """q1 is unchanged; only the pair's eta of 0.25 predicts a change for it."""
    result = _compare(
        {"q1": [0.9, 0.5, 0.25], "q2": [0.9, 0.5, 0.0]},
        {"q1": [0.9, 0.5, 0.25], "q2": [0.9, 0.5, 0.25]},
        k=2,
    )

    assert _cells(result)["global"] == {"tp": 0, "fp": 2, "tn": 0, "fn": 0, "undefined": 0}
    assert _cells(result)["query"] == {"tp": 0, "fp": 1, "tn": 1, "fn": 0, "undefined": 0}


def test_a_cutoff_at_the_list_length_leaves_the_margin_undefined() -> None:
    result = _compare({"q1": [0.9, 0.5, 0.1]}, {"q1": [0.8, 0.5, 0.1]}, k=3)

    assert _cells(result)["global"] == {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "undefined": 1}
    assert result.n_bound_disagreements == 0, "an undefined margin is not cross-checked"


def test_a_ranking_that_disagrees_with_the_scores_is_a_falsification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variant's ranking is reversed, so its top 1 crosses a 0.4 margin."""
    reverse: Iterator[bool] = iter([False, True])

    def ranked(*args: Any, **kwargs: Any) -> Ranking:
        ranking = rank_top_k(*args, **kwargs)
        flipped = tuple(reversed(ranking.order))
        return dataclasses.replace(ranking, order=flipped) if next(reverse) else ranking

    monkeypatch.setattr(module, "rank_top_k", ranked)
    result = _compare({"q1": [0.9, 0.5, 0.0]}, {"q1": [0.9, 0.5, 0.0]}, k=1)
    margin = Fraction(0.9) - Fraction(0.5)

    assert _cells(result)["global"] == {"tp": 0, "fp": 0, "tn": 0, "fn": 1, "undefined": 0}
    assert [f.scope for f in result.falsifications] == [Scope.GLOBAL, Scope.QUERY]
    assert result.as_dict()["falsifications"][1] == {
        "query_id": "q1",
        "k": 1,
        "scope": "query",
        "margin": float(margin),
        "margin_hex": float(margin).hex(),
        "margin_exact": f"{margin.numerator}/{margin.denominator}",
        "eta": 0.0,
        "eta_hex": "0x0.0p+0",
        "eta_exact": "0/1",
    }


def test_a_falsification_at_an_undefined_margin_records_no_margin() -> None:
    record = Falsification("q1", 3, "query", None, Fraction(0))  # type: ignore[arg-type]

    assert record.scope is Scope.QUERY
    assert record.as_dict()["margin"] is None
    assert (record.as_dict()["margin_hex"], record.as_dict()["margin_exact"]) == (None, None)


def test_confusion_counts_each_outcome() -> None:
    outcomes = [Outcome.TP, Outcome.TP, Outcome.UNDEFINED, Outcome.FN]

    assert Confusion.of(outcomes) == Confusion(tp=2, fn=1, undefined=1)
    assert str(Scope.GLOBAL) == "global"


def test_a_confusion_built_without_counts_is_all_zero() -> None:
    zero = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "undefined": 0}

    assert Confusion() == Confusion.of([])
    assert Confusion().as_dict() == zero


def test_grids_without_queries_have_no_eta() -> None:
    result = _compare({}, {}, k=1)
    zero = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "undefined": 0}

    assert (result.eta, result.n_queries) == (Fraction(0), 0)
    assert result.as_dict()["eta_exact"] == "0/1"
    assert _cells(result) == {"global": zero, "query": zero}


def test_a_variant_with_fewer_scores_than_candidates_is_refused() -> None:
    """The variant's scores and table cover two of its three candidates."""
    short = _grid({"q1": [0.9, 0.5]}, ("d0", "d1")).queries[0]
    other = dataclasses.replace(
        _grid({"q1": [0.9, 0.5, 0.1]}),
        queries=(dataclasses.replace(short, candidate_ids=_IDS),),
    )

    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is shorter than argument 1"):
        margin_noise(_grid({"q1": [0.9, 0.5, 0.1]}), other, k=1, reference_label="a", label="b")


@pytest.mark.parametrize(
    ("cur", "k", "message"),
    [
        ({"q2": [0.9, 0.5, 0.1]}, 1, "the two grids score different queries"),
        ({"q1": [0.9, 0.5, 0.1]}, 0, r"need at least one cutoff, each at least 1, got \[0\]"),
    ],
)
def test_misaligned_grids_or_a_bad_cutoff_are_refused(
    cur: dict[str, list[float]], k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _compare({"q1": [0.9, 0.5, 0.1]}, cur, k=k)


def test_grids_over_different_candidates_are_refused() -> None:
    other = _grid({"q1": [0.9, 0.5, 0.1]}, ("d0", "d1", "d3"))

    with pytest.raises(ValueError, match="query 'q1' ranks different candidates"):
        margin_noise(_grid({"q1": [0.9, 0.5, 0.1]}), other, k=1, reference_label="a", label="b")


@pytest.mark.property
@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.sampled_from([0.0, 0.25, 0.5, _BELOW, 1.0]), min_size=3, max_size=3),
    st.lists(st.sampled_from([0.0, 2.0**-54, -(2.0**-54), 0.25, -0.25]), min_size=3, max_size=3),
    st.integers(1, 4),
)
def test_no_perturbation_falsifies_the_bound(
    scores: list[float], moves: list[float], k: int
) -> None:
    moved = [s + d for s, d in zip(scores, moves, strict=True)]
    result = _compare({"q1": scores}, {"q1": moved}, k=k)

    assert result.falsifications == ()
    assert sum(result.confusion["query"].as_dict().values()) == 1


_RECORDS = [{"doc_id": d, "text": f"text {d}"} for d in _IDS]
_EXACT = {"q1": [0.9, 0.5, _BELOW], "q2": [0.1, 0.2, 0.3]}
_QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}}


def _store(tmp_path: Path) -> ScoreStore:
    store = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS, create=True)
    return store.add("toy.exact", _EXACT, kind=StoreKind.LEXICAL, params={})


def test_pairs_follow_label_order_and_skip_entries_without_a_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for label in ("toy.pairwise", "toy.naive"):
        store = store.add(
            label, _EXACT, kind=StoreKind.LEXICAL, params={}, reference_label="toy.exact"
        )
    store = store.add("other.run", _EXACT, kind=StoreKind.RUN, params={})

    assert noise_pairs(store) == [("toy.exact", "toy.naive"), ("toy.exact", "toy.pairwise")]


def test_the_driver_reports_each_pair_at_each_cutoff(tmp_path: Path) -> None:
    """toy.naive swaps q1's near tie at the cutoff; toy.pairwise is unchanged."""
    flipped = {**_EXACT, "q1": [0.9, _BELOW, 0.5]}
    store = _store(tmp_path)
    store = store.add(
        "toy.naive", flipped, kind=StoreKind.LEXICAL, params={}, reference_label="toy.exact"
    )
    store = store.add(
        "toy.pairwise", _EXACT, kind=StoreKind.LEXICAL, params={}, reference_label="toy.exact"
    )
    payload = run_margin_noise(store, _QRELS, cutoffs=(2, 3), depth=2)
    naive, pairwise = payload["pairs"]

    assert (payload["cutoffs"], payload["depth"], payload["missing"]) == ([2, 3], 2, None)
    assert (naive["reference"], naive["variant"]) == ("toy.exact", "toy.naive")
    assert naive["by_cutoff"]["2"]["confusion"]["global"]["tp"] == 1
    assert naive["by_cutoff"]["3"]["confusion"]["query"]["undefined"] == 2
    assert set(naive["by_cutoff"]["3"]["means"]) == {"ndcg@3", "recall@3", "rr@3"}
    assert payload["n_falsifications"] == 0
    assert pairwise["by_cutoff"]["2"]["eta"] == 0.0
    spreads = payload["spreads"]
    assert len(spreads) == 12, "two pairs, two cutoffs, three metrics"
    assert all(set(s) == {"experiment", "metric", "spread", "source"} for s in spreads)
    assert {s["experiment"] for s in spreads} == {"e7"}
    by_variant = {(s["source"]["variant"], s["metric"]): s["spread"] for s in spreads}
    assert by_variant[("toy.naive", "rr@2")] == 0.25, "q1's d1 falls from rank 2 to rank 3"
    assert by_variant[("toy.pairwise", "rr@2")] == 0.0
    means = naive["by_cutoff"]["2"]["means"]["rr@2"]
    assert means == {"reference": 0.75, "variant": 0.5, "abs_delta": 0.25}


def test_the_driver_breaks_an_exact_tie_by_id_ascending(tmp_path: Path) -> None:
    """toy.naive ties q1's d1 and the relevant d2; id_asc keeps d2 third."""
    tied = {**_EXACT, "q1": [0.9, 0.5, 0.5]}
    store = _store(tmp_path).add(
        "toy.naive", tied, kind=StoreKind.LEXICAL, params={}, reference_label="toy.exact"
    )
    qrels = {"q1": {"d2": 1}, "q2": {"d2": 1}}
    payload = run_margin_noise(store, qrels, cutoffs=(2,), depth=2)
    means = payload["pairs"][0]["by_cutoff"]["2"]["means"]

    assert means["rr@2"] == {"reference": 0.5, "variant": 0.5, "abs_delta": 0.0}
    assert means["recall@2"] == {"reference": 0.5, "variant": 0.5, "abs_delta": 0.0}


def test_the_driver_loads_each_reference_once_per_run_of_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two pairs share a.exact; b.naive is measured against b.exact alone."""
    other = {"q1": [0.1, 0.2, 0.3], "q2": [0.9, 0.5, 0.0]}
    store = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS, create=True)
    store = store.add("a.exact", _EXACT, kind=StoreKind.LEXICAL, params={})
    store = store.add("b.exact", other, kind=StoreKind.LEXICAL, params={})
    for label, scores, reference in (
        ("a.naive", _EXACT, "a.exact"),
        ("a.pairwise", _EXACT, "a.exact"),
        ("b.naive", other, "b.exact"),
    ):
        store = store.add(
            label, scores, kind=StoreKind.LEXICAL, params={}, reference_label=reference
        )
    loads: list[str] = []
    original = ScoreStore.grid

    def counted(self: ScoreStore, label: str) -> QueryGrid:
        loads.append(label)
        return original(self, label)

    monkeypatch.setattr(ScoreStore, "grid", counted)
    payload = run_margin_noise(store, _QRELS, cutoffs=(2,), depth=2)

    assert loads == ["a.exact", "a.naive", "a.pairwise", "b.exact", "b.naive"]
    variants = [(pair["reference"], pair["variant"]) for pair in payload["pairs"]]
    assert variants == [("a.exact", "a.naive"), ("a.exact", "a.pairwise"), ("b.exact", "b.naive")]
    assert [pair["by_cutoff"]["2"]["eta"] for pair in payload["pairs"]] == [0.0, 0.0, 0.0]


def test_a_store_without_pairs_reports_what_is_missing(tmp_path: Path) -> None:
    payload = run_margin_noise(_store(tmp_path), _QRELS, cutoffs=(1,), depth=2)

    assert (payload["pairs"], payload["spreads"], payload["n_falsifications"]) == ([], [], 0)
    assert payload["missing"] == "the store holds no entry with a reference label"
