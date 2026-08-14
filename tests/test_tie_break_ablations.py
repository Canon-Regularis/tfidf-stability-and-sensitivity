"""Tie-break ablations and margin stratification (README sections 4.5, 7.3).

Research question A2 asks whether ranking outcomes change purely because of the
secondary ordering rule, with no numerical perturbation at all. Two halves:

* with all scores distinct the three operators coincide, so the ablation has a
  well-defined null;
* with ties present they diverge at ``delta s = 0`` bit-for-bit.

Also that the stratification is not circular: grouping an operator comparison by
``m_k`` is legitimate only because the margin is tie-break independent, which is
asserted rather than assumed.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from tfidf_stability.analysis.stratify import (
    EXACT_TIE_BAND,
    UNDEFINED_BAND,
    margin_bands,
    stratify_by_margin,
)
from tfidf_stability.analysis.tie_break_ablations import (
    ablate_queries,
    ablate_query,
    disagreement_rate,
)
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.utils.numerics import same_bits

KS = (2, 3)


def table_of(pops: list[int], engs: list[int] | None = None) -> AttributeTable:
    engs = engs if engs is not None else list(reversed(pops))
    return AttributeTable.from_records(
        [
            {
                "doc_id": f"d{i}",
                "popularity": pops[i],
                "rating_sum2": 5,
                "rating_count": 2,
                "engagement": engs[i],
            }
            for i in range(len(pops))
        ]
    )


# ---------------------------------------------------------------------------
# The null: no ties, no disagreement
# ---------------------------------------------------------------------------
def test_all_distinct_scores_gives_no_disagreement_anywhere(
    mini_attributes: AttributeTable,
) -> None:
    """A2's premise. Without ties the attribute tuple is never consulted."""
    result = ablate_query([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], mini_attributes, ks=KS)
    assert result.scores_all_distinct is True
    assert not any(p.sets_differ for p in result.pairs)
    assert all(p.comparison.fks == 0.0 for p in result.pairs)


def test_ties_make_the_operators_diverge() -> None:
    """The converse, with the scores held bit-identical between operators."""
    table = table_of([3, 2, 1, 0])
    result = ablate_query([0.5, 0.5, 0.5, 0.5], table, ks=KS)
    assert result.scores_all_distinct is False
    assert any(p.sets_differ for p in result.pairs)


def test_every_operator_shares_one_sorted_score_array() -> None:
    """The structural guarantee that makes A1 and A2 independent.

    Sharing the array object means a margin cannot drift between operators, so
    stratifying an operator comparison by ``m_k`` is not circular.
    """
    table = table_of([3, 2, 1, 0])
    result = ablate_query([0.5, 0.5, 0.2, 0.0], table, ks=KS)
    arrays = [r.sorted_scores for r in result.rankings.values()]
    assert all(a is arrays[0] for a in arrays)
    for pair in result.pairs:
        if pair.margin.defined:
            assert same_bits(pair.margin.value, arrays[0][pair.k - 1] - arrays[0][pair.k])


def test_the_margin_recorded_is_identical_across_operator_pairs() -> None:
    table = table_of([3, 2, 1, 0])
    result = ablate_query([0.5, 0.5, 0.2, 0.0], table, ks=KS)
    for k in KS:
        margins = {p.margin.value for p in result.at_k(k) if p.margin.defined}
        assert len(margins) <= 1


# ---------------------------------------------------------------------------
# Structure of the result
# ---------------------------------------------------------------------------
def test_the_baseline_is_not_compared_against_itself(
    mini_attributes: AttributeTable,
) -> None:
    result = ablate_query([0.5] * 6, mini_attributes, ks=KS)
    assert {p.variant for p in result.pairs} == {"pi_score", "pi_alt"}
    assert all(p.baseline == "pi" for p in result.pairs)
    assert len(result.pairs) == 2 * len(KS)


def test_degenerate_query_is_flagged_and_still_ablated(
    mini_attributes: AttributeTable,
) -> None:
    """G3: a zero query is excluded from margin distributions and included in
    tie-break ablations, where it is the extreme case."""
    result = ablate_query([0.0] * 6, mini_attributes, ks=KS)
    assert result.query_degenerate is True
    assert result.pairs, "the degenerate query must still be compared"
    assert any(p.sets_differ for p in result.pairs)


def test_k_larger_than_the_corpus_is_handled_leniently(
    mini_attributes: AttributeTable,
) -> None:
    """A k of 50 on a 6-document corpus is a legitimate sweep point."""
    result = ablate_query([0.5] * 6, mini_attributes, ks=(5, 10, 20, 50))
    assert {p.k for p in result.pairs} == {5, 6}
    undefined = [p for p in result.pairs if not p.margin.defined]
    assert undefined, "m_k is undefined once k reaches N"


def test_query_id_is_carried_through() -> None:
    table = table_of([1, 0])
    results = ablate_queries([("q1", [0.5, 0.5]), ("q2", [1.0, 0.0])], table, ks=(1,))
    assert [r.query_id for r in results] == ["q1", "q2"]


# ---------------------------------------------------------------------------
# The headline statistic
# ---------------------------------------------------------------------------
def test_disagreement_rate_returns_its_denominator() -> None:
    """Section 7.1 requires the query count; a rate alone is not a claim."""
    table = table_of([3, 2, 1, 0])
    results = ablate_queries(
        [("a", [0.5, 0.5, 0.5, 0.5]), ("b", [0.9, 0.8, 0.7, 0.6])], table, ks=(2,)
    )
    rate, n = disagreement_rate(results, "pi", "pi_alt", 2)
    assert n == 2
    assert 0.0 <= rate <= 1.0


def test_disagreement_rate_is_zero_with_a_denominator_when_nothing_disagrees() -> None:
    table = table_of([3, 2, 1, 0])
    results = ablate_queries([("a", [0.9, 0.8, 0.7, 0.6])], table, ks=(2,))
    assert disagreement_rate(results, "pi", "pi_alt", 2) == (0.0, 1)


def test_disagreement_rate_on_an_empty_selection() -> None:
    assert disagreement_rate([], "pi", "pi_alt", 5) == (0.0, 0)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def test_margin_bands_bracket_tau() -> None:
    bands = margin_bands(1e-9)
    labels = [b[0] for b in bands]
    assert "(tau/10, tau]" in labels
    assert "(tau, 10*tau]" in labels
    # Contiguous and covering (0, inf).
    for (_, _, hi), (_, lo_next, _) in pairwise(bands):
        assert hi == lo_next
    assert bands[0][1] == 0.0
    assert bands[-1][2] == math.inf


def test_exact_ties_get_their_own_band() -> None:
    """``m_k == 0`` differs in kind from "small": the tie-break decides
    membership outright. On short text most of the mass sits there, so folding
    it into the smallest numeric band would hide the dominant effect."""
    table = table_of([3, 2, 1, 0])
    results = ablate_queries([("a", [0.5, 0.5, 0.5, 0.5])], table, ks=(2,))
    strata = stratify_by_margin(results, tau=1e-9, variant="pi_alt", ks=(2,))
    exact = next(s for s in strata if s.label == EXACT_TIE_BAND)
    assert exact.n == 1
    assert exact.disagreement_rate in (0.0, 1.0)


def test_undefined_margins_are_counted_not_dropped(
    mini_attributes: AttributeTable,
) -> None:
    results = ablate_queries([("a", [0.5] * 6)], mini_attributes, ks=(6,))
    strata = stratify_by_margin(results, tau=1e-9, variant="pi_alt", ks=(6,))
    undefined = next(s for s in strata if s.label == UNDEFINED_BAND)
    assert undefined.n == 1


def test_empty_bands_report_nan_not_zero() -> None:
    """An empty band means "no evidence", which is not "no disagreement".

    Returning 0.0 would invent a data point on the transition plot section 7.3
    asks for.
    """
    table = table_of([3, 2, 1, 0])
    results = ablate_queries([("a", [0.9, 0.8, 0.7, 0.6])], table, ks=(2,))
    strata = stratify_by_margin(results, tau=1e-9, variant="pi_alt", ks=(2,))
    empty = [s for s in strata if s.n == 0]
    assert empty, "this tiny corpus cannot populate every band"
    assert all(math.isnan(s.disagreement_rate) for s in empty)


def test_every_band_appears_for_every_k() -> None:
    """A complete x-axis, so a plot is not silently ragged."""
    table = table_of([3, 2, 1, 0])
    results = ablate_queries([("a", [0.5, 0.5, 0.2, 0.0])], table, ks=KS)
    strata = stratify_by_margin(results, tau=1e-9, ks=KS)
    n_bands = len(margin_bands(1e-9)) + 2  # + exact-tie + undefined
    assert len(strata) == n_bands * len(KS)
    for k in KS:
        assert len({s.label for s in strata if s.k == k}) == n_bands


def test_every_pair_lands_in_exactly_one_band() -> None:
    """A partition rather than mere coverage: the totals must reconcile."""
    table = table_of([3, 2, 1, 0])
    results = ablate_queries(
        [("a", [0.5, 0.5, 0.2, 0.0]), ("b", [0.9, 0.9, 0.9, 0.9]), ("c", [1.0, 0.7, 0.3, 0.0])],
        table,
        ks=KS,
    )
    strata = stratify_by_margin(results, tau=0.1, variant="pi_alt", ks=KS)
    for k in KS:
        assert sum(s.n for s in strata if s.k == k) == len(results)


def test_stratify_rejects_a_negative_tau() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stratify_by_margin([], tau=-1.0)


def test_tau_is_never_defaulted() -> None:
    """Section 7.1 makes every tie-break result conditional on tau, so no call
    site may inherit one silently."""
    import inspect

    signature = inspect.signature(stratify_by_margin)
    assert signature.parameters["tau"].default is inspect.Parameter.empty


def test_one_query_is_counted_once_even_when_several_k_clamp_together(
    mini_attributes: AttributeTable,
) -> None:
    """The denominator must be a query count, as the docstring promises.

    `ablate_query` clamps k to the candidate count (G3's lenient mode), so on a
    6-document corpus k in (10, 20, 50) all become 6 and emit three identical
    pairs for one query. Counting all three reported n = 3 for a single query,
    inflating the denominator of section 7.3's headline statistic.
    """
    result = ablate_query([0.5] * 6, mini_attributes, ks=(5, 10, 20, 50), query_id="q0")

    clamped = [
        p for p in result.pairs if p.baseline == "pi" and p.variant == "pi_score" and p.k == 6
    ]
    assert len(clamped) > 1, "the fixture must actually exercise the clamping"

    _, n = disagreement_rate([result], "pi", "pi_score", 6)
    assert n == 1, f"one query must contribute once, got a denominator of {n}"
