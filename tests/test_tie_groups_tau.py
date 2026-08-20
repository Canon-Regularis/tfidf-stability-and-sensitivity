"""Tie groups (README section 2.3.3, ``spec_addenda.md#g1``).

G1's central claim: section 2.3.3's tie group is a ball rather than an
equivalence class, and conflating the two has consequences. The adversarial
ladder ``s_i = i * 2^-20`` makes the claim executable: every value and every
difference is representable in binary64, so the tests below carry no
floating-point content and demonstrate structure rather than rounding.

On that ladder the ball relation is non-transitive, a single chain swallows all
six documents, and complete linkage sees only adjacent pairs. The three objects
disagree maximally, which is what the ``rho`` diagnostic exists to flag.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tfidf_stability.ranking.ranker import sorted_scores_desc
from tfidf_stability.ranking.tie_groups import (
    TieGroupIndex,
    chain_inflation_ratio,
    tie_ball_interval,
    tie_chains,
    tie_cliques,
)
from tfidf_stability.utils.validation import (
    ChainInflationWarning,
    TauExceedsScoreRangeWarning,
)

TAU = 2.0**-20  # a power of two: exact, and i*TAU is exact for small i


def ladder(n: int = 6, step: float = TAU) -> tuple[float, ...]:
    """``n`` scores in descending order, each exactly ``step`` below the last."""
    return tuple((n - 1 - i) * step for i in range(n))


def ball_members(scores, j, tau):  # type: ignore[no-untyped-def]
    lo, hi = tie_ball_interval(scores, j, tau)
    return set(range(lo, hi))


def brute_force_ball(scores, j, tau):  # type: ignore[no-untyped-def]
    """The predicate as section 2.3.3 and G9 write it."""
    return {i for i in range(len(scores)) if abs(scores[i] - scores[j]) <= tau}


# ---------------------------------------------------------------------------
# The adversarial ladder: G1's witness
# ---------------------------------------------------------------------------
def test_the_ladder_shows_the_ball_relation_is_not_transitive() -> None:
    """Document 2 is within tau of 1 and 1 is within tau of 0, while 2 is not
    within tau of 0. The relation is reflexive and symmetric and non-transitive,
    so tie groups do not partition the corpus."""
    s = ladder(6)
    assert ball_members(s, 1, TAU) == {0, 1, 2}
    assert ball_members(s, 0, TAU) == {0, 1}
    assert 2 in ball_members(s, 1, TAU)
    assert 0 in ball_members(s, 1, TAU)
    assert 2 not in ball_members(s, 0, TAU), "non-transitivity, demonstrated"


def test_the_extremes_of_one_ball_can_differ_by_two_tau() -> None:
    """Which is why section 2.3.3's "indistinguishable" is too strong as written."""
    s = ladder(6)
    members = sorted(ball_members(s, 1, TAU))
    spread = s[members[0]] - s[members[-1]]
    assert spread == 2 * TAU
    assert spread > TAU


def test_a_chain_swallows_the_ladder() -> None:
    """Every adjacent gap is exactly tau, so single linkage connects everything."""
    s = ladder(6)
    assert tie_chains(s, TAU) == ((0, 6),)
    assert len(tie_cliques(s, TAU)) == 5  # only adjacent pairs
    assert max(hi - lo for lo, hi in tie_cliques(s, TAU)) == 2
    assert chain_inflation_ratio(s, TAU) == 3.0  # 6 / 2


def test_the_ladder_triggers_the_chain_inflation_warning() -> None:
    with pytest.warns(ChainInflationWarning, match="chain inflation"):
        index = TieGroupIndex.build(ladder(6), TAU)
    assert index.rho == 3.0
    assert index.largest_chain == 6
    assert index.largest_clique == 2


def test_the_ladder_shatters_one_ulp_below_tau() -> None:
    """A one-ulp change in tau takes the largest chain from 6 to 1.

    The kind of decision discontinuity the paper studies, arising here in the
    diagnostic rather than in the ranking.
    """
    s = ladder(6)
    just_under = TAU - math.ulp(TAU)
    assert tie_chains(s, TAU) == ((0, 6),)
    assert len(tie_chains(s, just_under)) == 6
    assert chain_inflation_ratio(s, just_under) == 1.0


# ---------------------------------------------------------------------------
# tau = 0: the exact-tie baseline
# ---------------------------------------------------------------------------
def test_tau_zero_is_the_exact_tie_baseline() -> None:
    """At tau = 0 all three objects collapse onto exact-equality classes."""
    s = (1.0, 0.5, 0.5, 0.5, 0.25)
    chains = tie_chains(s, 0.0)
    cliques = tie_cliques(s, 0.0)
    assert chains == ((0, 1), (1, 4), (4, 5))
    assert cliques == chains
    assert ball_members(s, 1, 0.0) == {1, 2, 3}
    assert chain_inflation_ratio(s, 0.0) == 1.0


def test_tau_zero_distinguishes_scores_one_ulp_apart() -> None:
    """G9 pins the raw double: near is not the same as equal."""
    a = 0.5
    b = a + math.ulp(a)
    s = (b, a)
    assert ball_members(s, 0, 0.0) == {0}
    assert len(tie_chains(s, 0.0)) == 2


def test_negative_tau_is_rejected() -> None:
    for fn in (tie_chains, tie_cliques):
        with pytest.raises(ValueError, match="non-negative"):
            fn((1.0, 0.5), -1e-9)
    with pytest.raises(ValueError, match="non-negative"):
        tie_ball_interval((1.0, 0.5), 0, -1e-9)


# ---------------------------------------------------------------------------
# G9: the predicate must be the one the paper writes
# ---------------------------------------------------------------------------
def test_the_ball_is_inclusive_at_exactly_tau() -> None:
    """``<=`` rather than ``<``, on dyadic values so the boundary is exact."""
    s = (0.5, 0.25)
    assert ball_members(s, 0, 0.25) == {0, 1}
    assert ball_members(s, 0, 0.25 - math.ulp(0.25)) == {0}


@pytest.mark.property
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=40),
    st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    st.data(),
)
def test_binary_search_agrees_with_a_linear_scan(
    scores: list[float], tau: float, data: st.DataObject
) -> None:
    """Certifies the monotone-difference search against the literal predicate.

    Catches the tempting wrong implementation: binary-searching for
    ``S[j] +/- tau`` evaluates ``S[i] <= fl(S[j] + tau)``, a different test from
    G9's ``|s_i - s_j| <= tau``, and the two disagree at the boundary, the only
    place tie groups are interesting.

    The rank is drawn after the list, from its actual length, rather than drawn
    independently and filtered. Filtering rejects most examples (a rank up to 39
    against a list as short as 1) and trips Hypothesis's ``filter_too_much``
    health check intermittently, since the example database caches successful
    draws between runs.
    """
    s = sorted_scores_desc(scores)
    j = data.draw(st.integers(min_value=0, max_value=len(s) - 1), label="rank")
    assert ball_members(s, j, tau) == brute_force_ball(s, j, tau)


def test_the_naive_bound_shortcut_would_actually_differ() -> None:
    """Evidence that the caution above is more than theoretical.

    A value exists for which ``s - c <= tau`` and ``s <= c + tau`` disagree
    because ``c + tau`` rounds; the rounded bound would place that document in
    the wrong group.
    """
    c = 0.1
    tau = 0.2
    # fl(c + tau) != c + tau exactly; find a score in the gap.
    rounded_bound = c + tau
    disagreements = [
        x
        for x in (rounded_bound, rounded_bound - math.ulp(rounded_bound))
        if (x - c <= tau) != (x <= rounded_bound)
    ]
    assert disagreements, "expected the rounded bound to differ from the exact predicate"


# ---------------------------------------------------------------------------
# Structural properties of the three objects
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=40),
    st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
)
def test_chains_partition_the_corpus(scores: list[float], tau: float) -> None:
    s = sorted_scores_desc(scores)
    chains = tie_chains(s, tau)
    covered: list[int] = []
    for lo, hi in chains:
        covered.extend(range(lo, hi))
    assert covered == list(range(len(s))), "disjoint, contiguous and covering"


@pytest.mark.property
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=25),
    st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
)
def test_every_clique_has_diameter_at_most_tau(scores: list[float], tau: float) -> None:
    s = sorted_scores_desc(scores)
    for lo, hi in tie_cliques(s, tau):
        assert s[lo] - s[hi - 1] <= tau


@pytest.mark.property
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=12),
    st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
)
def test_maximal_cliques_match_brute_force(scores: list[float], tau: float) -> None:
    """Validates the O(N) sweep against an O(N^2) enumerator.

    The sweep is complete only because the near-tie graph is an indifference
    graph, so every maximal clique is a contiguous interval. Checks that lemma
    empirically rather than trusting it.
    """
    s = sorted_scores_desc(scores)
    n = len(s)
    intervals = {(a, b + 1) for a in range(n) for b in range(a, n) if s[a] - s[b] <= tau}
    maximal = {
        (a, b)
        for (a, b) in intervals
        if not any((c, d) != (a, b) and c <= a and b <= d for (c, d) in intervals)
    }
    assert set(tie_cliques(s, tau)) == maximal


@pytest.mark.property
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=25),
    st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
)
def test_rho_is_at_least_one_and_cliques_sit_inside_chains(scores: list[float], tau: float) -> None:
    """A clique's adjacent gaps are all ``<= tau``, so it lies within a chain."""
    s = sorted_scores_desc(scores)
    assert chain_inflation_ratio(s, tau) >= 1.0
    chains = tie_chains(s, tau)
    for a, b in tie_cliques(s, tau):
        assert any(lo <= a and b <= hi for lo, hi in chains)


@pytest.mark.property
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=25),
    st.floats(min_value=0.0, max_value=0.2, allow_nan=False),
    st.floats(min_value=0.0, max_value=0.2, allow_nan=False),
)
def test_tie_groups_are_monotone_in_tau(scores: list[float], tau_a: float, tau_b: float) -> None:
    """``tau_1 <= tau_2`` implies every group only grows."""
    lo_tau, hi_tau = sorted((tau_a, tau_b))
    s = sorted_scores_desc(scores)

    for j in range(len(s)):
        assert ball_members(s, j, lo_tau) <= ball_members(s, j, hi_tau)
    assert len(tie_chains(s, lo_tau)) >= len(tie_chains(s, hi_tau))
    assert max(hi - lo for lo, hi in tie_chains(s, lo_tau)) <= max(
        hi - lo for lo, hi in tie_chains(s, hi_tau)
    )


def test_ball_is_always_contiguous_in_rank_space() -> None:
    s = sorted_scores_desc([0.9, 0.5, 0.45, 0.44, 0.1, 0.0])
    for j in range(len(s)):
        lo, hi = tie_ball_interval(s, j, 0.06)
        assert lo <= j < hi


def test_out_of_range_rank_is_rejected() -> None:
    with pytest.raises(IndexError):
        tie_ball_interval((1.0, 0.5), 7, 0.1)


# ---------------------------------------------------------------------------
# Degenerate configurations (G3)
# ---------------------------------------------------------------------------
def test_tau_covering_the_score_range_warns_and_does_not_raise() -> None:
    """A legitimate point at the top of a sweep, so a warning rather than an
    error.

    ">=" rather than ">": at tau equal to the range every ball is already the
    whole corpus, so the degeneracy has begun. An erratum to G3, which says
    "tau > score range".
    """
    s = (1.0, 0.6, 0.2)
    span = s[0] - s[-1]
    with pytest.warns(TauExceedsScoreRangeWarning, match="entire score range"):
        index = TieGroupIndex.build(s, span)
    assert index.ball_members(0) == (0, 1, 2)
    assert index.n_chains == 1


def test_all_equal_scores_gives_rho_one_not_an_inflation_warning() -> None:
    """An erratum to G3, which says rho "fires" here.

    With every score equal, chain and clique coincide, so ``rho = 1``, its
    minimum. The score-range warning fires instead, the range being zero.
    """
    s = (0.5, 0.5, 0.5, 0.5)
    with pytest.warns(TauExceedsScoreRangeWarning):
        index = TieGroupIndex.build(s, 1e-9)
    assert index.rho == 1.0
    assert index.n_chains == 1
    assert index.largest_clique == 4


def test_zero_range_with_zero_tau_is_the_legitimate_baseline() -> None:
    """The one exclusion from the ">= range" rule: exact ties at tau = 0 are the
    intended baseline rather than a degenerate configuration."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        index = TieGroupIndex.build((0.5, 0.5, 0.5), 0.0)
    assert index.rho == 1.0


def test_single_document_corpus() -> None:
    index = TieGroupIndex.build((0.5,), 0.0)
    assert index.chains == ((0, 1),)
    assert index.cliques == ((0, 1),)
    assert index.rho == 1.0
    assert index.ball_members(0) == (0,)


def test_empty_corpus_yields_no_groups() -> None:
    assert tie_chains((), 0.1) == ()
    assert tie_cliques((), 0.1) == ()
    assert math.isnan(chain_inflation_ratio((), 0.1))


def test_the_zero_score_block_is_one_exact_tie_group(mini_model) -> None:  # type: ignore[no-untyped-def]
    """Documents with no in-vocabulary tokens all score 0.

    They form one exact-tie group at the bottom of every ranking even at
    tau = 0, so on short text section 7.3's "near-tie regime" is substantially
    an exact-tie regime.
    """
    from tfidf_stability.similarity.cosine import cosine_against_corpus
    from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

    q = TfidfVectoriser.transform_query(["numerical", "stability"], mini_model)
    docs = [mini_model.document(i) for i in range(mini_model.n_documents)]
    s = sorted_scores_desc(cosine_against_corpus(q, docs, mini_model.norms))

    zeros = [i for i, x in enumerate(s) if x == 0.0]
    assert zeros, "the mini corpus has documents orthogonal to this query"
    chain = next(c for c in tie_chains(s, 0.0) if c[0] == zeros[0])
    assert set(range(*chain)) == set(zeros)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_report_carries_both_the_ball_and_the_partition_statistics() -> None:
    """G1 requires them side by side: the ball is what the paper defines, the
    chain is the object with a well-defined "group containing document i", and
    rho says how far apart the two have drifted."""
    with pytest.warns(ChainInflationWarning):
        report = TieGroupIndex.build(ladder(6), TAU).report()
    assert set(report) == {
        "tau",
        "n_documents",
        "n_chains",
        "largest_chain",
        "n_cliques",
        "largest_clique",
        "rho",
    }
    assert report["rho"] == 3.0


def test_chain_of_returns_the_unique_containing_chain() -> None:
    s = (1.0, 0.99, 0.5, 0.49)
    index = TieGroupIndex.build(s, 0.02)
    assert index.chain_of(0) == (0, 2)
    assert index.chain_of(3) == (2, 4)
    with pytest.raises(IndexError):
        index.chain_of(99)


@pytest.mark.parametrize("bad", [-1e-9, float("nan"), float("-inf")])
def test_an_inadmissible_tau_is_rejected_by_every_tie_group_object(bad: float) -> None:
    """NaN slipped through a guard whose own message said non-negative.

    The check was ``if tau < 0.0`` and every comparison with NaN is false. With
    ``tau = NaN`` both ``gap > tau`` and ``gap <= tau`` are false, so
    ``tie_chains`` returned a single group covering the corpus, ``tie_cliques``
    returned all singletons and ``rho`` reported N: three contradictory answers
    to one question, with no error and ``rho`` at its maximum.
    """
    scores = (1.0, 0.9, 0.5, 0.5, 0.1)
    for call in (tie_chains, tie_cliques, chain_inflation_ratio):
        with pytest.raises(ValueError, match="non-negative"):
            call(scores, bad)
    with pytest.raises(ValueError, match="non-negative"):
        tie_ball_interval(scores, 2, bad)


def test_an_empty_corpus_builds_an_index_without_warning_about_its_score_range() -> None:
    """A corpus with no documents has no span to compare tau against, so the
    degeneracy warning has nothing to say. Warning anyway would put a caveat on
    every empty sweep point and train a reader to ignore the ones that matter.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        index = TieGroupIndex.build((), tau=1.0)

    assert index.chains == ()
    assert index.cliques == ()
    assert index.n_chains == 0


def test_rho_is_undefined_rather_than_one_on_an_empty_corpus() -> None:
    """Both the largest chain and the largest clique are zero, and 0/0 is not
    1.0: an empty corpus is no evidence about chain inflation, and reporting the
    value that means "no inflation" would put a data point on section 7.3's plot
    that no corpus produced.
    """
    assert math.isnan(TieGroupIndex.build((), tau=1.0).rho)
    assert math.isnan(chain_inflation_ratio((), 1.0))


# ---------------------------------------------------------------------------
# Erroneous: the rank index, with the range it was checked against
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("j", [-1, -7, 2, 99])
def test_a_rank_outside_the_corpus_names_the_range_it_was_checked_against(j: int) -> None:
    """The message carries the computed upper bound, not just the index. On a
    truncated ranking the reader's question is "how many were there", and the
    bound answers it without a second call.

    Negative indices matter especially: they are legal Python and would silently
    address the *worst*-ranked documents rather than raising.
    """
    with pytest.raises(IndexError, match=f"rank index {j} out of range 0..1"):
        tie_ball_interval((1.0, 0.5), j, 0.0)


def test_an_empty_corpus_reports_an_empty_range_rather_than_a_negative_one() -> None:
    """`0..-1` is how "there are no valid ranks" renders. Ugly and correct: a
    message saying `0..0` would claim rank 0 exists."""
    with pytest.raises(IndexError, match=r"rank index 0 out of range 0\.\.-1"):
        tie_ball_interval((), 0, 0.0)


def test_the_rank_is_checked_before_the_tolerance() -> None:
    """Both guards fire for an empty corpus with a NaN tau. Which error the
    caller sees is part of the interface, and the rank is the one they can act
    on: a bad tau is a sweep parameter, a bad rank is a bug in the caller."""
    with pytest.raises(IndexError, match="rank index"):
        tie_ball_interval((), 0, math.nan)

    with pytest.raises(ValueError, match="tau must be non-negative, got nan"):
        tie_ball_interval((1.0,), 0, math.nan)


# ---------------------------------------------------------------------------
# Erroneous: the tolerance, across its whole invalid domain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tau", [-1.0, -5e-324, -math.inf, math.nan, -sys.float_info.max, float("-nan")]
)
def test_every_inadmissible_tolerance_is_refused_with_its_value(tau: float) -> None:
    """The guard is spelled `not (tau >= 0.0)` rather than `tau < 0.0` precisely
    so NaN is caught: every comparison with NaN is false, so the natural spelling
    lets it through a guard whose own message says non-negative.

    A NaN tau is the worst case because it does not fail loudly downstream: it
    makes `gap > tau` and `gap <= tau` both false, so chains return one group
    covering the corpus, cliques return all singletons, and rho reports N --
    three contradictory answers to the same question, with nothing raised.
    """
    with pytest.raises(ValueError, match="tau must be non-negative"):
        tie_ball_interval((1.0, 0.5), 0, tau)


@pytest.mark.parametrize("tau", [0.0, -0.0, 5e-324, 1.0, math.inf])
def test_every_admissible_tolerance_is_accepted_including_both_zeros(tau: float) -> None:
    """`-0.0 >= 0.0` is true, so a negative zero is a legal tolerance and
    behaves as zero. Worth stating: it is the one negative-looking value the
    guard lets through, and the sweep that produced it may have divided."""
    lo, hi = tie_ball_interval((1.0, 1.0, 0.5), 0, tau)
    assert lo == 0
    assert hi >= 1


def test_a_negative_zero_tolerance_recovers_exact_ties_like_a_positive_zero() -> None:
    """Not merely accepted -- identical. If the two zeros gave different groups,
    a tau derived by halving could silently change a tie-group statistic."""
    scores = (1.0, 1.0, 0.5)
    assert tie_ball_interval(scores, 0, -0.0) == tie_ball_interval(scores, 0, 0.0)
    assert tie_chains(scores, -0.0) == tie_chains(scores, 0.0)
    assert tie_cliques(scores, -0.0) == tie_cliques(scores, 0.0)


def test_the_group_builders_return_nothing_for_an_empty_corpus_before_checking_tau() -> None:
    """`tie_chains` and `tie_cliques` short-circuit on an empty array, so they
    accept a tau that `tie_ball_interval` refuses. Pinned as the guard-order
    difference it is: a sweep reaching tau = NaN on an empty corpus gets three
    different behaviours from the three entry points.
    """
    assert tie_chains((), math.nan) == ()
    assert tie_cliques((), math.nan) == ()

    with pytest.raises(IndexError):
        tie_ball_interval((), 0, math.nan)


# ---------------------------------------------------------------------------
# Boundary: G9's predicate, at the one place it is interesting
# ---------------------------------------------------------------------------
def test_the_search_agrees_with_the_predicate_where_the_rounded_bound_would_not() -> None:
    """G9 pins the test as ``|s_i - s_j| <= tau``. Binary-searching for
    ``S[j] + tau`` instead evaluates ``S[i] <= fl(S[j] + tau)``, which is a
    different predicate whenever that addition rounds.

    The sweep chooses triples where it does round, so the two spellings actually
    disagree, and checks the interval against the predicate written out
    literally. A count outside the loop keeps the sweep from passing vacuously.
    """
    checked = 0
    rounded_cases = 0
    for exponent in range(-8, 8):
        centre = 1.0 + 2.0**exponent
        for tau in (2.0**-40, 2.0**-30, 2.0**-20 + 2.0**-53):
            scores = tuple(
                sorted((centre, centre - tau, centre + tau, centre - 2 * tau), reverse=True)
            )
            for j in range(len(scores)):
                assert ball_members(scores, j, tau) == brute_force_ball(scores, j, tau)
                checked += 1
                if (scores[j] + tau) - scores[j] != tau:
                    rounded_cases += 1

    assert checked == 192, "the sweep did not run the shape it claims"
    assert rounded_cases > 0, "no case actually rounded; the sweep proves nothing about G9"


def test_the_ball_at_exactly_tau_includes_the_boundary_document() -> None:
    """The comparison is inclusive (G9). One ulp further out it is not, which is
    the whole content of the choice."""
    scores = (1.0, 1.0 - 2.0**-30, 1.0 - 2.0**-29)
    assert ball_members(scores, 0, 2.0**-30) == {0, 1}
    assert ball_members(scores, 0, math.nextafter(2.0**-30, 0.0)) == {0}


# ---------------------------------------------------------------------------
# chain_inflation_ratio
# ---------------------------------------------------------------------------
def test_an_empty_corpus_has_no_inflation_ratio() -> None:
    """0/0. NaN rather than 1.0, because "no evidence" is not "no inflation"."""
    assert math.isnan(chain_inflation_ratio((), 0.1))


def test_all_equal_scores_give_the_ratio_its_minimum_rather_than_a_warning() -> None:
    """The errata correct G3 here: when every score is equal, the chain and the
    clique are both the whole corpus, so rho is 1 -- its *minimum*. It is the
    range warning that fires in that situation, not the inflation one."""
    assert chain_inflation_ratio((0.5,) * 4, 0.0) == 1.0


def test_the_ladder_inflates_the_ratio_above_one() -> None:
    """G1's witness: single-linkage chaining walks the whole ladder while no two
    ends are within tau of each other."""
    assert chain_inflation_ratio(ladder(6), TAU) == 3.0


# ---------------------------------------------------------------------------
# The diagnostics: a warn / do-not-warn matrix
# ---------------------------------------------------------------------------
# `filterwarnings = ["error"]` means an unexpected warning fails the suite, so
# the "does not warn" rows are asserted by simply calling `build` -- but that
# reads as an accident. `catch_warnings(record=True)` states it, and lets the
# rows that fire *both* diagnostics be checked as a set rather than one at a
# time, which `pytest.warns` cannot do.
def _diagnostics(scores: tuple[float, ...], tau: float, **kwargs: float) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TieGroupIndex.build(scores, tau, **kwargs)  # type: ignore[arg-type]
    return sorted(str(w.category.__name__) for w in caught)


_SPANNING = (1.0, 0.5, 0.0)


def test_a_tolerance_below_the_score_range_is_not_degenerate() -> None:
    """One ulp below the span, nothing fires -- and the groups are still
    meaningful, which is what makes the warning above it a boundary rather than
    a blanket."""
    assert _diagnostics(_SPANNING, math.nextafter(1.0, 0.0)) == []


@pytest.mark.parametrize("tau", [1.0, math.nextafter(1.0, math.inf), 2.0, math.inf])
def test_a_tolerance_at_or_above_the_score_range_is_flagged(tau: float) -> None:
    """ ">=" rather than ">": at tau equal to the range every ball is already the
    whole corpus, so the degeneracy has begun rather than being about to."""
    assert "TauExceedsScoreRangeWarning" in _diagnostics(_SPANNING, tau)


def test_a_zero_range_at_zero_tolerance_is_the_exempted_baseline() -> None:
    """An all-tied corpus at tau = 0 is the exact-tie baseline the whole
    tie-break study is normalised against. Warning there would fire on the one
    configuration that is least degenerate, and under `filterwarnings = error`
    it would abort every sweep at its first point."""
    assert _diagnostics((0.5, 0.5, 0.5), 0.0) == []


def test_a_zero_range_at_any_positive_tolerance_reports_an_infinite_ratio() -> None:
    """`tau / span` is `x / 0`. The errata require `inf` rather than the NaN a
    naive `0.0 / 0.0` would give, because the message is read by a human
    deciding whether to trust the point."""
    with pytest.warns(TauExceedsScoreRangeWarning, match=r"tau/span=inf") as caught:
        TieGroupIndex.build((0.5, 0.5), 5e-324)
    assert "span=0.0" in str(caught[0].message)


def test_an_empty_corpus_produces_no_diagnostic_at_all() -> None:
    """There is no span to compare against and no ratio to inflate. Both
    diagnostics are guarded on the corpus being non-empty."""
    assert _diagnostics((), 1.0) == []
    assert _diagnostics((), 0.0) == []


def test_both_diagnostics_can_fire_for_one_index() -> None:
    """They are independent conditions and neither suppresses the other. At tau
    equal to the span the corpus is one chain and one clique, so rho is exactly
    1 -- which a threshold below 1 still exceeds."""
    assert _diagnostics(_SPANNING, 1.0, rho_warn_threshold=0.0) == [
        "ChainInflationWarning",
        "TauExceedsScoreRangeWarning",
    ]


def test_a_nan_inflation_threshold_silences_the_diagnostic_rather_than_raising() -> None:
    """`rho > nan` is false for every rho, so a threshold that arrived as NaN
    disables the check silently. Pinned: it is the one threshold value that
    turns a diagnostic off without saying so, and a sweep computing its
    threshold could produce one.
    """
    assert "ChainInflationWarning" not in _diagnostics(ladder(6), TAU, rho_warn_threshold=math.nan)
    assert "ChainInflationWarning" in _diagnostics(ladder(6), TAU, rho_warn_threshold=2.0)


@pytest.mark.parametrize(("threshold", "fires"), [(0.0, True), (2.0, True), (3.0, False)])
def test_the_inflation_threshold_is_exclusive(threshold: float, fires: bool) -> None:
    """`rho > threshold`, so a ratio exactly at the threshold does not fire. The
    ladder's rho is exactly 3, which makes it the witness for both sides."""
    assert chain_inflation_ratio(ladder(6), TAU) == 3.0
    assert (
        "ChainInflationWarning" in _diagnostics(ladder(6), TAU, rho_warn_threshold=threshold)
    ) is fires


def test_the_diagnostic_is_attributed_to_the_caller_not_to_the_module() -> None:
    """`stacklevel=2`. Without it every warning points at `tie_groups.py`, and a
    sweep emitting thousands of them gives the reader one location that is never
    the one they need. Asserted by filename, since that is what a reader sees.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TieGroupIndex.build(_SPANNING, 2.0)

    assert caught
    assert Path(caught[0].filename).name == Path(__file__).name


# ---------------------------------------------------------------------------
# chain_of
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("j", [-1, 3, 99])
def test_asking_which_chain_holds_a_rank_outside_the_corpus_is_refused(j: int) -> None:
    """The loop falls through rather than returning the last chain, which a
    negative index would otherwise do silently."""
    index = TieGroupIndex.build((1.0, 0.5, 0.0), 0.0)
    with pytest.raises(IndexError, match=f"rank {j} is outside the corpus"):
        index.chain_of(j)


def test_every_rank_belongs_to_exactly_one_chain() -> None:
    """Chains partition the corpus, so the lookup is total on valid ranks and
    the intervals it returns are disjoint."""
    # The threshold is lifted clear of the ladder's rho of 3 so the inflation
    # diagnostic does not fire: `filterwarnings = ["error"]` would turn it into
    # a failure of a test that is not about the diagnostic at all.
    index = TieGroupIndex.build(ladder(6), TAU, rho_warn_threshold=10.0)
    found = [index.chain_of(j) for j in range(6)]

    assert all(lo <= j < hi for j, (lo, hi) in enumerate(found))
    assert set(found) <= set(index.chains)


def test_the_inflation_diagnostic_is_also_attributed_to_the_caller() -> None:
    """The two warnings carry their own `stacklevel`, and testing one says
    nothing about the other.

    The range test above builds a corpus whose tau exceeds its span, which fires
    only the range warning -- so the inflation warning's attribution went
    unchecked. The category is selected explicitly here rather than taken as
    `caught[0]`, so this stays about the inflation warning even if the other one
    starts firing too.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TieGroupIndex.build(ladder(6), TAU)

    inflation = [w for w in caught if w.category is ChainInflationWarning]
    assert len(inflation) == 1
    assert Path(inflation[0].filename).name == Path(__file__).name


def test_the_range_diagnostic_reports_how_far_past_the_range_the_tolerance_is() -> None:
    """`tau / span` is the number that tells a reader whether the point is a
    hair over the edge or far outside it, which decides whether the results are
    worth reading at all.

    The span is deliberately not 1.0: at a span of one the ratio equals tau and
    an assertion on it would pass for any arithmetic that happens to be the
    identity there.
    """
    with pytest.warns(TauExceedsScoreRangeWarning, match=r"tau/span=4\.0") as caught:
        TieGroupIndex.build((1.0, 0.5), 2.0)

    message = str(caught[0].message)
    assert "span=0.5" in message
    assert "tau=2.0" in message


def test_an_empty_corpus_has_no_largest_group_rather_than_a_group_of_one() -> None:
    """`max(..., default=0)`. A default of 1 would report a single-document
    chain in a corpus with no documents, and `rho` would then be 1.0 -- a
    perfectly-behaved ratio computed from nothing.
    """
    empty = TieGroupIndex.build((), 1.0)
    assert empty.largest_chain == 0
    assert empty.largest_clique == 0
    assert math.isnan(empty.rho)
