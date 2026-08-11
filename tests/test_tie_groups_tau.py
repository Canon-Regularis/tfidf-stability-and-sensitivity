"""Tie groups (README section 2.3.3, ``spec_addenda.md#g1``).

G1's central claim is that section 2.3.3's tie group is a **ball**, not an
equivalence class, and that conflating the two is a mistake with consequences.
The claim is made executable here by the *adversarial ladder*
``s_i = i * 2^-20``: every value and every difference is exactly representable
in binary64, so the tests below have no floating-point content at all and
demonstrate structure rather than rounding.

On that ladder the ball relation is visibly non-transitive, a single chain
swallows all six documents, and complete linkage sees only adjacent pairs -- the
three objects disagree maximally, which is exactly what the ``rho`` diagnostic
exists to flag.
"""

from __future__ import annotations

import math

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
    """The predicate exactly as section 2.3.3 and G9 write it."""
    return {i for i in range(len(scores)) if abs(scores[i] - scores[j]) <= tau}


# ---------------------------------------------------------------------------
# The adversarial ladder -- G1's witness
# ---------------------------------------------------------------------------
def test_the_ladder_shows_the_ball_relation_is_not_transitive() -> None:
    """Document 2 is within tau of 1, and 1 is within tau of 0, but 2 is not
    within tau of 0. The relation is reflexive and symmetric but **not
    transitive**, so tie groups genuinely do not partition the corpus."""
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

    A decision discontinuity of exactly the kind the paper studies, arising here
    in the diagnostic rather than in the ranking.
    """
    s = ladder(6)
    just_under = TAU - math.ulp(TAU)
    assert tie_chains(s, TAU) == ((0, 6),)
    assert len(tie_chains(s, just_under)) == 6
    assert chain_inflation_ratio(s, just_under) == 1.0


# ---------------------------------------------------------------------------
# tau = 0 -- the exact-tie baseline
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
# G9 -- the predicate must be the one the paper writes
# ---------------------------------------------------------------------------
def test_the_ball_is_inclusive_at_exactly_tau() -> None:
    """``<=``, not ``<``, and on dyadic values so the boundary is exact."""
    s = (0.5, 0.25)
    assert ball_members(s, 0, 0.25) == {0, 1}
    assert ball_members(s, 0, 0.25 - math.ulp(0.25)) == {0}


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=40),
    st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    st.data(),
)
def test_binary_search_agrees_with_a_linear_scan(
    scores: list[float], tau: float, data: st.DataObject
) -> None:
    """Certifies the monotone-difference search against the literal predicate.

    This is the test that would catch the tempting-but-wrong implementation:
    binary-searching for ``S[j] +/- tau`` evaluates ``S[i] <= fl(S[j] + tau)``,
    which is a *different* test from G9's ``|s_i - s_j| <= tau`` and disagrees
    precisely at the boundary -- the only place tie groups are interesting.

    The rank is drawn *after* the list, from its actual length, rather than
    drawn independently and filtered. Filtering here would reject most examples
    (a rank up to 39 against a list as short as 1), which trips Hypothesis's
    ``filter_too_much`` health check -- intermittently, because the example
    database caches successful draws between runs. An intermittently failing
    test is worse than a failing one.
    """
    s = sorted_scores_desc(scores)
    j = data.draw(st.integers(min_value=0, max_value=len(s) - 1), label="rank")
    assert ball_members(s, j, tau) == brute_force_ball(s, j, tau)


def test_the_naive_bound_shortcut_would_actually_differ() -> None:
    """Evidence that the caution above is warranted, not theoretical.

    A value exists for which ``s - c <= tau`` and ``s <= c + tau`` disagree,
    because ``c + tau`` rounds. If the implementation used the rounded bound it
    would place this document in the wrong group.
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


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=25),
    st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
)
def test_every_clique_has_diameter_at_most_tau(scores: list[float], tau: float) -> None:
    s = sorted_scores_desc(scores)
    for lo, hi in tie_cliques(s, tau):
        assert s[lo] - s[hi - 1] <= tau


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=12),
    st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
)
def test_maximal_cliques_match_brute_force(scores: list[float], tau: float) -> None:
    """Validates the O(N) sweep against an O(N^2) enumerator.

    The sweep is complete only because the near-tie graph is an indifference
    graph, so every maximal clique is a contiguous interval; this checks that
    lemma empirically rather than trusting it.
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
    """A legitimate point at the top of a sweep, so a warning, not an error.

    Note ">=" rather than ">": at tau exactly equal to the range every ball is
    already the whole corpus, so the degeneracy has begun. That is an erratum to
    G3, which says "tau > score range".
    """
    s = (1.0, 0.6, 0.2)
    span = s[0] - s[-1]
    with pytest.warns(TauExceedsScoreRangeWarning, match="entire score range"):
        index = TieGroupIndex.build(s, span)
    assert index.ball_members(0) == (0, 1, 2)
    assert index.n_chains == 1


def test_all_equal_scores_gives_rho_one_not_an_inflation_warning() -> None:
    """An erratum to G3, which says rho "fires" here.

    With every score equal, chain and clique coincide, so ``rho = 1`` -- its
    *minimum*. What actually fires is the score-range warning, because the range
    is zero.
    """
    s = (0.5, 0.5, 0.5, 0.5)
    with pytest.warns(TauExceedsScoreRangeWarning):
        index = TieGroupIndex.build(s, 1e-9)
    assert index.rho == 1.0
    assert index.n_chains == 1
    assert index.largest_clique == 4


def test_zero_range_with_zero_tau_is_the_legitimate_baseline() -> None:
    """The one exclusion from the ">= range" rule: exact ties at tau = 0 are the
    intended baseline, not a degenerate configuration."""
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
    """Documents with no in-vocabulary tokens all score exactly 0.

    They therefore form a single exact-tie group at the bottom of every ranking,
    even at tau = 0 -- which is why section 7.3's "near-tie regime" is, on short
    text, substantially an *exact*-tie regime.
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
