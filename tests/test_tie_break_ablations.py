"""Tie-break ablations and margin stratification (README sections 4.5, 7.3).

A2 asks whether the secondary ordering rule alone changes a ranking. With
distinct scores the three operators coincide; with ties they diverge at
``delta s = 0``. ``m_k`` is asserted tie-break independent, so it can stratify.
"""

from __future__ import annotations

import dataclasses
import math
from itertools import pairwise

import pytest

from tfidf_stability.analysis.stratify import (
    EXACT_TIE_BAND,
    UNDEFINED_BAND,
    Stratum,
    _band_of,
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
    """A zero query (G3) is flagged and still ablated as the extreme case.

    G3 excludes it from margin distributions and includes it in tie-break
    ablations.
    """
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


def test_the_disagreement_rate_is_the_share_of_queries_that_differed() -> None:
    """The numerator, pinned to an exact value rather than to a range.

    Section 7.3 publishes it, and a constant 0.0 passes the denominator, empty
    and all-agree tests. Query "a" is a four-way exact tie where pi and pi_alt
    pick different top-2 sets; query "b" has distinct scores. One of two: 0.5.
    """
    table = table_of([3, 2, 1, 0])
    results = ablate_queries(
        [("a", [0.5, 0.5, 0.5, 0.5]), ("b", [0.9, 0.8, 0.7, 0.6])], table, ks=(2,)
    )

    assert disagreement_rate(results, "pi", "pi_alt", 2) == (0.5, 2)
    assert disagreement_rate(results, "pi", "pi_score", 2) == (0.0, 2), (
        "the same denominator with a different numerator, so the pair above is "
        "not passing on the denominator alone"
    )


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
    """An exact tie, ``m_k == 0``, has a band of its own.

    At an exact tie the tie-break decides membership outright. On short text
    most of the mass sits there, so a numeric band would hide the main effect.
    """
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


# ---------------------------------------------------------------------------
# Boundary: a k larger than the corpus, which ablate_query calls a legitimate
# grid point rather than a configuration error
# ---------------------------------------------------------------------------
def test_a_k_larger_than_the_corpus_is_stratified_under_the_k_that_was_asked_for() -> None:
    """Two ``k`` values above ``N`` are two strata, not one stratum counted twice.

    ``comparison`` is taken at ``k_eff = min(k, N)`` and ``margin`` at the
    requested ``k``; strata are keyed by the margin's ``k``. With ten documents
    20 and 50 clamp to 10, and each of ``k`` = 10, 20, 50 holds one pair.
    """
    table = table_of([9, 3, 7, 1, 8, 2, 6, 4, 5, 0])
    results = ablate_queries([("a", [0.9, 0.9, 0.8, 0.75, 0.6, 0.6, 0.5, 0.4, 0.3, 0.1])], table)
    above = (20, 50)

    strata = stratify_by_margin(results, tau=1e-9, ks=(5, 10, *above))

    for k in (10, *above):
        placed = [s for s in strata if s.k == k and s.n]
        assert len(placed) == 1, f"k={k}: exactly one band holds this query's single pair"
        assert placed[0].n == 1, (
            f"k={k}: n={placed[0].n} means pairs from another k were counted here"
        )
        assert placed[0].label == UNDEFINED_BAND, (
            f"k={k}: k >= N, so m_k has no r_(k+1) and the margin is undefined"
        )


def test_no_pair_is_dropped_when_the_clamped_k_is_absent_from_the_requested_set() -> None:
    """No pair is dropped when the clamped ``k`` is not a requested grid point.

    Eight documents under ``(5, 10, 20)`` clamp 10 and 20 to 8, which is not in
    that set; each requested ``k`` still holds its one pair. The test above
    covers the case where the clamped value is in ``ks``.
    """
    table = table_of([9, 3, 7, 1, 8, 2, 6, 4])
    results = ablate_queries([("a", [0.9, 0.9, 0.8, 0.75, 0.6, 0.6, 0.5, 0.1])], table)
    ks = (5, 10, 20)

    strata = stratify_by_margin(results, tau=1e-9, ks=ks)

    assert sum(s.n for s in strata) == len(ks), (
        "one query, one pair per requested k, every one of them placed"
    )
    for k in ks:
        assert sum(s.n for s in strata if s.k == k) == 1, f"k={k} lost its pair"


def test_stratify_rejects_a_negative_tau() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stratify_by_margin([], tau=-1.0)


def test_tau_is_never_defaulted() -> None:
    """``stratify_by_margin`` has no default for ``tau``.

    Section 7.1 makes every tie-break result conditional on tau, so no call
    site may inherit one silently.
    """
    import inspect

    signature = inspect.signature(stratify_by_margin)
    assert signature.parameters["tau"].default is inspect.Parameter.empty


def test_one_query_is_counted_once_even_when_several_k_clamp_together(
    mini_attributes: AttributeTable,
) -> None:
    """The denominator is a query count, as the docstring promises.

    `ablate_query` clamps k to the candidate count (G3's lenient mode), so on a
    6-document corpus k in (10, 20, 50) all become 6 and give three pairs at
    k = 6 for one query, which section 7.3's denominator still counts once.
    """
    result = ablate_query([0.5] * 6, mini_attributes, ks=(5, 10, 20, 50), query_id="q0")

    clamped = [
        p for p in result.pairs if p.baseline == "pi" and p.variant == "pi_score" and p.k == 6
    ]
    assert len(clamped) > 1, "the fixture must actually exercise the clamping"

    _, n = disagreement_rate([result], "pi", "pi_score", 6)
    assert n == 1, f"one query must contribute once, got a denominator of {n}"


def test_a_query_with_no_pair_at_that_k_is_left_out_of_the_denominator() -> None:
    """A query with no pair at the requested ``k`` is left out of the count.

    `ablate_query` clamps k to the candidate count, so a short candidate list
    has no pair at the requested k. Counting it as an agreement would dilute
    the rate; the count returned beside the rate makes that visible.
    """
    results = [
        ablate_query([0.5, 0.5, 0.2, 0.0], table_of([3, 2, 1, 0]), query_id="long", ks=(3,)),
        ablate_query([0.5, 0.5], table_of([2, 1]), query_id="short", ks=(3,)),
    ]
    rate, n = disagreement_rate(results, "pi", "pi_alt", 3)
    assert n == 1, "only the query with three candidates could be compared at k = 3"
    assert 0.0 <= rate <= 1.0

    # The short query is not lost; it contributes at the k it is clamped to.
    _, n_at_two = disagreement_rate(results, "pi", "pi_alt", 2)
    assert n_at_two == 1


def test_a_result_can_be_sliced_by_operator_pair_and_by_k() -> None:
    """``for_pair`` and ``at_k`` slice a result by operator pair and by ``k``.

    One gives a comparison across every k, the other every comparison at one k.
    Between them they partition the pairs, so either slice is safe to summarise.
    """
    table = table_of([3, 2, 1, 0])
    (result,) = ablate_queries([("a", [0.5, 0.5, 0.2, 0.0])], table, ks=KS)

    for_pair = result.for_pair("pi", "pi_alt")
    assert for_pair, "the pi/pi_alt comparison is one of the ones run"
    assert {p.k for p in for_pair} == set(KS)
    assert all(p.baseline == "pi" and p.variant == "pi_alt" for p in for_pair)

    at_k = result.at_k(KS[0])
    assert at_k, "k values from the requested set are present"
    assert all(p.k == KS[0] for p in at_k)
    assert len(at_k) == len(result.pairs) // len(KS)

    assert result.for_pair("pi", "no_such_operator") == ()
    assert result.at_k(9999) == ()


def test_a_margin_outside_every_band_is_undefined_rather_than_well_separated() -> None:
    """A margin outside every band is undefined rather than well separated.

    The bands cover (0, inf), so only a negative value or NaN misses them all.
    It stays in the partition, so per-k totals reconcile, and lands in
    `undefined`, since the top band would report a broken sort as best separated.
    """
    bands = margin_bands(1e-9)
    assert _band_of(-1.0, True, bands) == UNDEFINED_BAND
    assert _band_of(-5e-324, True, bands) == UNDEFINED_BAND, "including the smallest negative"
    assert _band_of(math.inf, True, bands) == bands[-1][0], "the top band is unbounded above"
    assert _band_of(math.nan, True, bands) == UNDEFINED_BAND
    assert _band_of(0.5, False, bands) == UNDEFINED_BAND, "undefined wins over the value"
    assert _band_of(0.0, True, bands) == EXACT_TIE_BAND


# ---------------------------------------------------------------------------
# The banding rule and the statistics it produces, by value
# ---------------------------------------------------------------------------
def test_the_bands_are_half_open_at_the_bottom_and_closed_at_the_top() -> None:
    """Each band is `lo < value <= hi`: open below and closed above.

    The ends decide where a margin of exactly tau lands. Section 7.3 reads the
    transition off the band straddling tau, so a boundary one band too low moves
    the reported transition without changing any total.
    """
    tau = 1e-9
    bands = margin_bands(tau)
    labels = {b[0] for b in bands}
    assert "(tau/10, tau]" in labels
    assert "(tau, 10*tau]" in labels

    # Exactly tau is the top of its band, not the bottom of the next.
    assert _band_of(tau, True, bands) == "(tau/10, tau]"
    assert _band_of(tau + math.ulp(tau), True, bands) == "(tau, 10*tau]"

    # The same rule one decade down, so the result does not depend on where tau
    # sits.
    assert _band_of(tau / 10.0, True, bands) == "(tau/100, tau/10]"
    assert _band_of(tau / 10.0 + math.ulp(tau / 10.0), True, bands) == "(tau/10, tau]"


def test_a_tau_of_exactly_zero_is_accepted_as_the_exact_tie_baseline() -> None:
    """A tau of exactly zero is accepted as the exact-tie baseline.

    The guard is `not (tau >= 0.0)`, so zero passes. A `> 0` guard would reject
    the one point every tie-break result is normalised against.
    """
    strata = stratify_by_margin([], tau=0.0, ks=(2,))
    assert strata, "a tau of zero still produces the full band set"
    assert any(s.label == EXACT_TIE_BAND for s in strata)


def test_the_exact_tie_band_is_the_single_point_zero() -> None:
    """The exact-tie band has both bounds at zero.

    Its bounds are reported like every other band's and a plot's axis is built
    from them. The band holds one value, which is why it is separate from the
    smallest numeric band.
    """
    stratum = next(
        s for s in stratify_by_margin([], tau=1e-9, ks=(2,)) if s.label == EXACT_TIE_BAND
    )
    assert (stratum.lo, stratum.hi) == (0.0, 0.0)


def test_the_disagreement_rate_and_means_are_the_arithmetic_they_claim() -> None:
    """Both tied queries disagree, so the exact-tie band's rate is exactly 1.0.

    The rate and both means are pinned to exact values, so dividing by the wrong
    count, or multiplying instead, fails. The rate is section 7.3's headline.
    """
    table = table_of([3, 2, 1, 0])
    results = ablate_queries(
        [
            ("tied_a", [0.5, 0.5, 0.5, 0.5]),
            ("tied_b", [0.5, 0.5, 0.5, 0.5]),
            ("distinct_a", [0.9, 0.8, 0.7, 0.6]),
            ("distinct_b", [0.9, 0.8, 0.7, 0.6]),
        ],
        table,
        ks=(2,),
    )
    strata = stratify_by_margin(results, tau=1e-9, variant="pi_alt", ks=(2,))

    tied = next(s for s in strata if s.label == EXACT_TIE_BAND)
    assert tied.n == 2, "both all-tied queries land in the exact-tie band"
    assert tied.disagreement_rate == tied.n_disagree / 2.0
    assert tied.disagreement_rate in (0.0, 0.5, 1.0)

    assert tied.n_disagree == 2
    assert tied.disagreement_rate == 1.0
    # Both means are over the band's two members, so a sum multiplied by the
    # count instead of divided by it reads 4.0 here.
    assert tied.mean_fks == 1.0
    assert tied.mean_jaccard == 1.0

    distinct = next(s for s in strata if s.label == "(100*tau, inf)")
    assert distinct.n == 2, "the two strictly-ordered queries land far above tau"
    assert distinct.disagreement_rate == 0.0, "distinct scores leave nothing to break"


def test_a_comparison_nobody_ran_is_empty_rather_than_everything() -> None:
    """A comparison that was not run yields empty bands, not every pair.

    The filter keeps the pairs matching the requested operators. An inverted
    test keeps everything else, which on a result set holding several variants
    fills the bands with a plausible number of the wrong pairs.
    """
    table = table_of([3, 2, 1, 0])
    results = ablate_queries([("a", [0.5, 0.5, 0.2, 0.0])], table, ks=(2,))
    assert any(p.variant == "pi_alt" for r in results for p in r.pairs), "the premise"

    for absent in ({"variant": "no_such_variant"}, {"baseline": "no_such_baseline"}):
        strata = stratify_by_margin(results, tau=1e-9, ks=(2,), **absent)
        assert strata, "the bands are still reported"
        assert all(s.n == 0 for s in strata), f"nothing should match {absent}"


def test_a_rate_over_an_empty_band_is_undefined_rather_than_zero() -> None:
    """Both means over an empty band are NaN, as the disagreement rate is.

    No evidence is not a measurement of zero.
    """
    empty = next(s for s in stratify_by_margin([], tau=1e-9, ks=(2,)) if s.n == 0)
    assert math.isnan(empty.disagreement_rate)
    assert math.isnan(empty.mean_fks)
    assert math.isnan(empty.mean_jaccard)


def test_the_zero_norm_document_count_is_carried_rather_than_assumed() -> None:
    """``n_zero_norm_docs`` defaults to zero and is recorded on the result.

    Padding a candidate set with all-stopword documents produces an exact-tie
    block unrelated to the tie-break rule under test. A non-zero default would
    report that block on corpora that have none.
    """
    table = table_of([3, 2, 1, 0])
    plain = ablate_query([0.9, 0.8, 0.7, 0.6], table, ks=(2,))
    assert plain.n_zero_norm_docs == 0

    padded = ablate_query([0.9, 0.8, 0.0, 0.0], table, ks=(2,), n_zero_norm_docs=2)
    assert padded.n_zero_norm_docs == 2


def test_the_rate_matches_on_all_three_of_operator_pair_and_k() -> None:
    """The rate matches on all three of baseline, variant and ``k``.

    Relaxing any one of the three to an inequality picks up a pair from another
    comparison, which still yields a rate and a denominator, but not the ones
    requested.
    """
    table = table_of([3, 2, 1, 0])
    results = ablate_queries([("a", [0.5, 0.5, 0.5, 0.5])], table, ks=(2, 3))

    matched, n = disagreement_rate(results, "pi", "pi_alt", 2)
    assert n == 1, "exactly the one pair at k = 2 for this operator pair"

    # None of these three exists, and each differs from the request in one field.
    for baseline, variant, k in [
        ("no_such_baseline", "pi_alt", 2),
        ("pi", "no_such_variant", 2),
        ("pi", "pi_alt", 99),
    ]:
        rate, count = disagreement_rate(results, baseline, variant, k)
        assert (rate, count) == (0.0, 0), f"({baseline}, {variant}, k={k}) matched something"
    assert matched in (0.0, 1.0)


def test_a_stratum_cannot_be_edited_after_it_is_measured() -> None:
    """A published cell of table A2, so an editable one is a rewritable result.

    `n` and `n_disagree` give the section 7.3 rate, travel into a run manifest
    and are not re-derived downstream. Both `frozen=True` and `slots=True` of
    the dataclass are asserted.
    """
    stratum = Stratum(
        label="(0, tau/100]",
        k=5,
        lo=0.0,
        hi=1e-11,
        n=4,
        n_disagree=1,
        mean_fks=0.25,
        mean_jaccard=0.75,
    )

    assert stratum.disagreement_rate == 0.25, "the premise: it reports a rate from these two"
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field 'n'"):
        stratum.n = 400  # type: ignore[misc]
    denominator = "cannot assign to field 'n_disagree'"
    with pytest.raises(dataclasses.FrozenInstanceError, match=denominator):
        stratum.n_disagree = 0  # type: ignore[misc]
    assert not hasattr(stratum, "__dict__"), "slots=True: no ad-hoc attributes either"
