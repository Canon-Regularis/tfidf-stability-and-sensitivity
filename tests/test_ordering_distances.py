"""Ordering distances (README sections 4.5 and 7.3, ``spec_addenda.md#g2``).

FKS is not a metric at any penalty (:func:`test_fks_is_a_near_metric_not_a_metric`); ``K_int``,
Kendall on the intersection, misses section 7.3's membership change and is never
reported alone (:func:`test_intersection_kendall_is_blind_to_membership_change`).
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tfidf_stability.ranking.distances import (
    FKS_PENALTY,
    compare_top_k,
    fks_max,
    inversion_count,
    jaccard_distance,
    kendall_fks,
    kendall_tau_distance,
    top_k_disagreement,
)
from tfidf_stability.utils.numerics import same_bits


# ---------------------------------------------------------------------------
# Inversion counting
# ---------------------------------------------------------------------------
def test_inversion_count_against_brute_force() -> None:
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 12)
        seq = [rng.randrange(5) for _ in range(n)]
        expected = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])
        assert inversion_count(seq) == expected


def test_inversion_count_extremes() -> None:
    assert inversion_count([]) == 0
    assert inversion_count([1, 2, 3, 4]) == 0
    assert inversion_count([4, 3, 2, 1]) == 6  # C(4,2)
    assert inversion_count([1, 1, 1]) == 0, "equal elements are not inversions"


# ---------------------------------------------------------------------------
# Same-set Kendall tau
# ---------------------------------------------------------------------------
def test_kendall_tau_on_identical_and_reversed_orders() -> None:
    a = [1, 2, 3, 4]
    assert kendall_tau_distance(a, a) == 0.0
    assert kendall_tau_distance(a, list(reversed(a))) == 1.0


def test_kendall_tau_is_symmetric_and_normalised() -> None:
    for a in itertools.permutations(range(4)):
        for b in itertools.permutations(range(4)):
            d = kendall_tau_distance(list(a), list(b))
            assert 0.0 <= d <= 1.0
            assert d == kendall_tau_distance(list(b), list(a))


def test_kendall_tau_is_zero_below_two_elements() -> None:
    assert kendall_tau_distance([], []) == 0.0
    assert kendall_tau_distance([7], [7]) == 0.0


def test_kendall_tau_refuses_differing_sets() -> None:
    """Differing sets belong to FKS, so the same-set function raises."""
    with pytest.raises(ValueError, match="same set"):
        kendall_tau_distance([1, 2], [1, 3])


def test_kendall_tau_satisfies_the_triangle_inequality() -> None:
    """Unlike FKS, the same-set distance is a metric."""
    worst = 0.0
    for a, b, c in itertools.product(itertools.permutations(range(4)), repeat=3):
        ab = kendall_tau_distance(list(a), list(b))
        bc = kendall_tau_distance(list(b), list(c))
        ac = kendall_tau_distance(list(a), list(c))
        worst = max(worst, ac - (ab + bc))
    assert worst <= 1e-12


# ---------------------------------------------------------------------------
# FKS generalised Kendall
# ---------------------------------------------------------------------------
def test_fks_is_zero_on_identical_lists() -> None:
    assert kendall_fks([1, 2, 3], [1, 2, 3]) == 0.0
    assert kendall_fks([], []) == 0.0


def test_fks_is_symmetric() -> None:
    rng = random.Random(3)
    for _ in range(300):
        k = rng.randint(1, 4)
        a = rng.sample(range(8), k)
        b = rng.sample(range(8), k)
        assert kendall_fks(a, b) == kendall_fks(b, a)


def test_fks_reduces_to_inversion_count_when_the_sets_match() -> None:
    """With equal sets only case 1 can fire, so FKS must be plain Kendall."""
    for a in itertools.permutations(range(5)):
        for b in itertools.permutations(range(5)):
            position = {v: i for i, v in enumerate(b)}
            expected = inversion_count([position[v] for v in a])
            assert kendall_fks(list(a), list(b), normalise=False) == float(expected)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 50])
def test_disjoint_lists_attain_the_maximum(k: int) -> None:
    a = list(range(k))
    b = list(range(1000, 1000 + k))
    assert kendall_fks(a, b, normalise=False) == fks_max(k)
    assert kendall_fks(a, b) == 1.0


def test_fks_max_matches_the_closed_form() -> None:
    """``k^2 + p*k*(k-1)``, which at p = 1/2 is ``k(3k-1)/2``."""
    for k in range(1, 60):
        assert fks_max(k) == pytest.approx(k * (3 * k - 1) / 2)
    assert fks_max(0) == 0.0


def test_singleton_disjoint_lists_are_maximally_distant() -> None:
    """Two disjoint singletons are maximally distant.

    Two disjoint one-element lists contribute a case-3 pair, so the maximum
    is 1 and the normalised distance is 1.0; a `k < 2` guard returning 0 fails.
    """
    assert fks_max(1) == 1.0
    assert kendall_fks([1], [2]) == 1.0
    assert kendall_fks([1], [1]) == 0.0


@pytest.mark.property
@given(
    st.lists(st.integers(0, 9), min_size=0, max_size=6, unique=True),
    st.lists(st.integers(0, 9), min_size=0, max_size=6, unique=True),
)
def test_fks_stays_within_zero_and_one(a: list[int], b: list[int]) -> None:
    assert 0.0 <= kendall_fks(a, b) <= 1.0 + 1e-12


def test_fks_is_a_near_metric_not_a_metric() -> None:
    """FKS violates the triangle inequality, so it is not a metric (G2).

    ``A`` and ``C`` are disjoint so their distance is maximal, ``B`` shares one
    element with each, and the inequality fails by 4 (12 against 6 + 2). Bounded
    distortion holds, which suffices for disagreement rates but not clustering.
    """
    a, b, c = [3, 1, 0], [5, 3, 4], [5, 4, 2]
    d_ab = kendall_fks(a, b, normalise=False)
    d_bc = kendall_fks(b, c, normalise=False)
    d_ac = kendall_fks(a, c, normalise=False)

    assert set(a).isdisjoint(c), "the premise: A and C share nothing"
    assert d_ac == fks_max(3), "so their distance is maximal"
    assert (d_ab, d_bc, d_ac) == (6.0, 2.0, 12.0)
    assert d_ac > d_ab + d_bc, "the triangle inequality is violated"


@pytest.mark.parametrize("penalty", [0.0, 0.25, 0.5, 1.0])
def test_no_penalty_value_restores_the_triangle_inequality(penalty: float) -> None:
    """The violation is not an artefact of the particular penalty chosen."""
    a, b, c = [3, 1, 0], [5, 3, 4], [5, 4, 2]
    lhs = kendall_fks(a, c, penalty, normalise=False)
    rhs = kendall_fks(a, b, penalty, normalise=False) + kendall_fks(b, c, penalty, normalise=False)
    assert lhs > rhs


def test_the_penalty_is_the_neutral_choice() -> None:
    """``p = 1/2`` sits midway between the optimistic and pessimistic readings.

    ``p = 0`` assumes an unseen pair agrees and understates disagreement, the
    wrong bias for a study of instability; ``p = 1`` overstates it.
    """
    a, b = [1, 2], [3, 4]  # disjoint, so only cases 3 and 4 arise
    optimistic = kendall_fks(a, b, 0.0, normalise=False)
    neutral = kendall_fks(a, b, FKS_PENALTY, normalise=False)
    pessimistic = kendall_fks(a, b, 1.0, normalise=False)
    assert optimistic < neutral < pessimistic
    assert neutral == (optimistic + pessimistic) / 2


def test_fks_case_two_penalises_the_absent_element_ranked_first() -> None:
    """Case 2 counts a pair only when the absent element is ranked first.

    ``a = [x, y]``, ``b = [x]``. In ``b`` the present ``x`` counts as ranked
    above the absent ``y``, and ``a`` agrees, so nothing is counted. Reversing
    ``a`` to ``[y, x]`` puts the absent element first and the pair counts.
    """
    assert kendall_fks([1, 2], [1], normalise=False) == 0.0
    assert kendall_fks([2, 1], [1], normalise=False) == 1.0


def test_fks_case_three_always_counts() -> None:
    """One element from each list alone: the two lists cannot be reconciled."""
    assert kendall_fks([1], [2], normalise=False) == 1.0


# ---------------------------------------------------------------------------
# Set measures
# ---------------------------------------------------------------------------
def test_top_k_disagreement_is_about_the_set_not_the_order() -> None:
    """Section 7.3 asks for set disagreement; reordering is the FKS measure."""
    assert top_k_disagreement([1, 2, 3], [3, 2, 1]) is False
    assert top_k_disagreement([1, 2, 3], [1, 2, 4]) is True


def test_jaccard_distance() -> None:
    """The last case pins the expression, not just the quantity.

    The two algebraic forms of the Jaccard distance are one ulp apart in
    binary64, and the native core reproduces the reference's bits, so the form
    is part of the contract. Rationale: docs/engineering_notes.md#jaccard-form
    """
    assert jaccard_distance([1, 2], [1, 2]) == 0.0
    assert jaccard_distance([1, 2], [3, 4]) == 1.0
    assert jaccard_distance([], []) == 0.0
    assert same_bits(jaccard_distance([1, 2], [2, 3]), 1.0 - 1 / 3)
    assert jaccard_distance([1, 2], [2, 3]) != 2 / 3, (
        "the rearranged form is a different double; pinning 2/3 would admit it"
    )


# ---------------------------------------------------------------------------
# The combined comparison
# ---------------------------------------------------------------------------
def test_compare_top_k_on_identical_lists() -> None:
    c = compare_top_k([1, 2, 3], [1, 2, 3], 3)
    assert c.sets_differ is False
    assert c.fks == 0.0
    assert c.kendall_intersection == 0.0
    assert c.intersection_size == 3
    assert c.jaccard == 0.0
    assert c.swapped == 0


def test_intersection_kendall_is_blind_to_membership_change() -> None:
    """``K_int`` is never reported alone because it misses membership change.

    The lists share one element, so the intersection Kendall is NaN, which can
    be misread as no reordering; the set indicator and FKS both see the change.
    """
    c = compare_top_k([1, 2, 3], [1, 4, 5], 3)
    assert math.isnan(c.kendall_intersection)
    assert c.intersection_size == 1
    assert c.sets_differ is True
    assert c.fks > 0.0
    assert c.swapped == 2


def test_intersection_kendall_reports_its_support() -> None:
    """It is uninterpretable without the support size, so both travel together."""
    c = compare_top_k([1, 2, 3, 4], [4, 3, 9, 8], 4)
    assert c.intersection_size == 2
    assert c.kendall_intersection == 1.0  # 3 and 4 appear in opposite order


def test_compare_top_k_truncates_to_k() -> None:
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 9, 8, 7]
    assert compare_top_k(a, b, 2).sets_differ is False
    assert compare_top_k(a, b, 3).sets_differ is True


@pytest.mark.property
@given(
    st.lists(st.integers(0, 9), min_size=1, max_size=6, unique=True),
    st.lists(st.integers(0, 9), min_size=1, max_size=6, unique=True),
    st.integers(1, 6),
)
def test_compare_top_k_invariants(a: list[int], b: list[int], k: int) -> None:
    c = compare_top_k(a, b, k)
    assert 0.0 <= c.fks <= 1.0 + 1e-12
    assert 0.0 <= c.jaccard <= 1.0
    assert c.intersection_size <= min(len(a[:k]), len(b[:k]))
    assert c.sets_differ == (set(a[:k]) != set(b[:k]))
    if not c.sets_differ:
        assert c.jaccard == 0.0
        assert c.swapped == 0


# ---------------------------------------------------------------------------
# kendall_tau_distance: the multiplicity precondition
# ---------------------------------------------------------------------------
def test_two_lists_of_the_same_set_but_different_multiplicity_are_refused() -> None:
    """Lists with the same set but different multiplicities are refused.

    A guard on length and `set()` alone accepts `[1, 1, 2]` and `[1, 2, 2]`,
    and the last-wins `position` map then returns `0.0`; both languages compare
    multisets. Rationale: docs/engineering_notes.md#kendall-multiplicity-guard
    """
    left, right = [1, 1, 2], [1, 2, 2]
    assert len(left) == len(right), "a length check alone accepts the pair"
    assert set(left) == set(right), "and so does a set comparison"
    with pytest.raises(ValueError, match="same multiplicities"):
        kendall_tau_distance(left, right)


def test_a_repeat_is_still_accepted_when_both_sides_carry_it() -> None:
    """A repeat carried by both sides is accepted and yields a number.

    `Counter` equality asks for the same multiset, which `[1, 1, 2]` and
    `[2, 1, 1]` satisfy, so the last-wins position map produces a value with
    no canonical meaning. Every caller in the package passes a permutation.
    """
    assert kendall_tau_distance([1, 1, 2], [2, 1, 1]) == pytest.approx(2 / 3)
    assert kendall_tau_distance([1, 1, 2], [1, 1, 2]) == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [([1, 2], [1, 2, 3]), ([1, 2, 3], [1, 2]), ([1, 2], [3, 4]), ([], [1])],
)
def test_orderings_over_different_sets_are_refused_towards_the_right_function(
    a: list[int], b: list[int]
) -> None:
    """Different lengths and memberships both raise and name `kendall_fks`.

    A caller comparing top-k lists that may differ in membership has the wrong
    function, not bad data.
    """
    with pytest.raises(ValueError, match="use kendall_fks"):
        kendall_tau_distance(a, b)


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_elements_have_no_pair_to_disagree_about(n: int) -> None:
    """Fewer than two elements give `0.0`, not NaN: no pair can be discordant."""
    items = list(range(n))
    assert kendall_tau_distance(items, items) == 0.0


@pytest.mark.parametrize("n", [2, 3, 8, 9])
def test_a_reversed_ordering_is_exactly_one(n: int) -> None:
    """A full reversal saturates the `C(n, 2)` normaliser bit for bit.

    Odd and even sizes split the merge in the inversion count differently.
    """
    forwards = list(range(n))
    assert same_bits(kendall_tau_distance(forwards, forwards[::-1]), 1.0)


# ---------------------------------------------------------------------------
# inversion_count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 9, 16, 17])
def test_a_reversed_sequence_has_every_pair_inverted(n: int) -> None:
    """A reversed sequence of length `n` has `n(n-1)/2` inversions.

    Odd and even sizes straddle the merge split, where a recursive count can
    lose or double-count a pair.
    """
    assert inversion_count(list(range(n, 0, -1))) == n * (n - 1) // 2


def test_equal_neighbours_are_not_inversions() -> None:
    """The merge keeps equal elements in order, so ties are not inversions.

    Otherwise a ranking with tied scores would disagree with itself.
    """
    assert inversion_count([1, 1, 1]) == 0
    assert inversion_count([2, 1, 1]) == 2


# ---------------------------------------------------------------------------
# fks_max: the closed form across its domain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [-1, -(2**20), 0])
def test_a_non_positive_list_length_has_no_maximum(k: int) -> None:
    """A non-positive `k` answers `0.0` rather than the closed form.

    Only `k = 0` has no pairs. A negative `k` is not a list length, and the
    closed form would give it a positive maximum.
    """
    assert fks_max(k) == 0.0


@pytest.mark.parametrize("penalty", [0.0, 0.25, 0.5, 1.0])
def test_a_single_pair_of_disjoint_singletons_is_one_for_every_admissible_penalty(
    penalty: float,
) -> None:
    """`k = 1` has a maximum of 1 at every admissible penalty.

    Two disjoint singletons contribute exactly one case-3 pair, and case 3 is
    the one the penalty does not touch.
    """
    assert fks_max(1, penalty) == 1.0


@pytest.mark.parametrize("penalty", [math.nan, math.inf, -math.inf, -1.0, -0.5, 1.5, 2.0])
def test_a_penalty_outside_the_domain_is_refused_by_both_entry_points(
    penalty: float,
) -> None:
    """A penalty outside `[0, 1]` is refused by both entry points.

    G2 fixes the domain. Unguarded, `fks_max` is NaN for a NaN penalty and
    negative for a sufficiently negative one; `ceiling > 0.0` is then false and
    `kendall_fks` returns its `k = 0` value, so disjoint lists read as agreeing.
    """
    with pytest.raises(ValueError, match=r"penalty must lie in \[0, 1\]"):
        fks_max(3, penalty)
    with pytest.raises(ValueError, match=r"penalty must lie in \[0, 1\]"):
        kendall_fks([1, 2, 3], [4, 5, 6], penalty)
    with pytest.raises(ValueError, match=r"penalty must lie in \[0, 1\]"):
        kendall_fks([1, 2, 3], [4, 5, 6], penalty, normalise=False)


def test_disjoint_lists_are_maximally_distant_at_every_admissible_penalty() -> None:
    """Disjoint lists normalise to 1.0 at every admissible penalty.

    Disjoint lists attain `fks_max` by construction, so the normalised distance
    is the maximum whatever the penalty, independent of the range guard.
    """
    a, b = [1, 2, 3], [4, 5, 6]
    for penalty in (0.0, 0.25, 0.5, 1.0):
        assert kendall_fks(a, b, penalty) == 1.0, f"disjoint lists at p={penalty}"
        assert kendall_fks(a, b, penalty, normalise=False) == fks_max(3, penalty)


@pytest.mark.parametrize("penalty", [0.0, 0.5, 1.0])
def test_no_pairs_means_the_penalty_term_never_runs(penalty: float) -> None:
    """`k = 0` short-circuits before the closed form.

    At `k = 0` the fall-through computes `0.0 + p * 0.0 * -1.0`, which is
    `+0.0` for every admissible penalty, so the guard is not observable.
    Rationale: docs/engineering_notes.md#fks-max-zero-guard
    """
    assert fks_max(0, penalty) == 0.0
    assert same_bits(fks_max(0, penalty), 0.0), "positive zero, not negative"


@pytest.mark.parametrize(("k", "expected"), [(2, 5.0), (3, 12.0), (50, 3725.0)])
def test_the_closed_form_is_k_times_three_k_minus_one_over_two(k: int, expected: float) -> None:
    """At the pinned penalty of one half the closed form is `k(3k-1)/2`.

    G2 bounds `k <= 50`, so 50 is the largest size the protocol produces.
    """
    assert fks_max(k) == expected
    assert fks_max(k) == k * (3 * k - 1) / 2


def test_the_penalty_scales_only_the_same_list_pairs() -> None:
    """The penalty weights only the case-4 term of `k^2 + p * k * (k-1)`.

    The first term is case 3 and is fixed; at `p = 0` only case 3 survives.
    """
    assert fks_max(2, 0.0) == 4.0, "k^2 alone"
    assert fks_max(2, 1.0) == 6.0, "case 4 at full weight"
    assert fks_max(2, 0.5) == 5.0, "the pinned half"


# ---------------------------------------------------------------------------
# kendall_fks: the degenerate list shapes
# ---------------------------------------------------------------------------
def test_two_empty_lists_are_not_distant() -> None:
    """Two empty lists have no pairs, so the distance is zero.

    The normaliser is also zero, so the guard returns before dividing.
    """
    assert kendall_fks([], []) == 0.0


def test_one_empty_list_is_distant_from_a_non_empty_one() -> None:
    """One empty list is distant from a non-empty one, below the maximum.

    Every pair lies in the non-empty list alone, which is case 4, so only the
    penalty counts: 0.5 against a maximum of 5.
    """
    distance = kendall_fks([1, 2], [])
    assert 0.0 < distance < 1.0


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_disjoint_lists_of_equal_length_attain_exactly_one(k: int) -> None:
    """Disjoint lists of equal length normalise to exactly 1.

    The normaliser is `fks_max(k)`, and disjoint lists are its maximum.
    """
    left = list(range(k))
    right = list(range(k, 2 * k))
    assert same_bits(kendall_fks(left, right), 1.0)


def test_the_unnormalised_form_returns_the_pair_count_itself() -> None:
    """The unnormalised form returns the pair count itself.

    Lists of different k have different maxima, so the raw count is the
    comparable quantity between them.
    """
    assert kendall_fks([0], [1], normalise=False) == fks_max(1)
    assert kendall_fks([0, 1], [2, 3], normalise=False) == fks_max(2)


# ---------------------------------------------------------------------------
# compare_top_k: the prefix length and the swap counter
# ---------------------------------------------------------------------------
def test_comparing_no_documents_at_all_reports_a_wholly_degenerate_row() -> None:
    """`k = 0` is a legitimate grid point that compares nothing.

    The sets agree, FKS is 0.0 and nothing is shared; the intersection Kendall
    is NaN ("no evidence" rather than "no disagreement"), since there are no pairs.
    """
    result = compare_top_k([1, 2, 3], [3, 2, 1], 0)

    assert result.sets_differ is False
    assert result.intersection_size == 0
    assert result.fks == 0.0
    assert math.isnan(result.kendall_intersection)


def test_a_negative_prefix_length_is_refused_rather_than_sliced_from_the_end() -> None:
    """A negative prefix length is refused rather than sliced from the end.

    `a[:-1]` is a legal slice, so an unchecked negative `k` compares almost the
    whole lists. The native binding refuses the argument, so the reference does
    too and both backends answer alike. `Ranking.top_k` refuses in the same words.
    """
    with pytest.raises(
        ValueError, match=r"^compare_top_k\(-1\) is not a prefix; k must be non-negative$"
    ):
        compare_top_k([1, 2, 3], [3, 2, 1], -1)


def test_prefixes_of_different_lengths_report_the_documents_that_differ() -> None:
    """Unequal prefixes report the documents that differ, not half of them.

    Halving the symmetric difference floor-divides a one-document difference to
    zero swaps while `sets_differ` is true. A fold's candidate set can be smaller
    than `k` by protocol (G19), so `swapped` is the larger one-way difference.
    """
    result = compare_top_k([1, 2, 3], [1, 2], 3)

    assert result.sets_differ is True
    assert result.swapped == 1, "document 3 left the top-k"
    assert result.intersection_size == 2

    # Disjoint prefixes of different lengths: three left and two entered, and
    # the count reports the larger rather than rounding their mean down.
    disjoint = compare_top_k([1, 2, 3], [4, 5], 3)
    assert disjoint.swapped == 3
    assert disjoint.intersection_size == 0


def test_a_swap_of_one_document_for_another_counts_as_one() -> None:
    """Equal-length prefixes with an even symmetric difference count one swap."""
    result = compare_top_k([1, 2, 3], [1, 2, 4], 3)
    assert result.swapped == 1
    assert result.sets_differ is True


@pytest.mark.parametrize("k", [1, 2, 3, 4, 99])
def test_a_prefix_longer_than_the_lists_compares_what_there_is(k: int) -> None:
    """A `k` beyond the candidate count compares the whole lists.

    Slicing past the end is not an error, which gives the lenient clamping G3
    describes without a guard.
    """
    result = compare_top_k([1, 2, 3], [1, 2, 3], k)
    assert result.sets_differ is False
    assert result.intersection_size == min(k, 3) if k > 0 else True


# ---------------------------------------------------------------------------
# top_k_disagreement and jaccard_distance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b", "differ"),
    [
        ([1, 2], [2, 1], False),
        ([1, 2], [1, 3], True),
        ([], [], False),
        ([1], [], True),
        ([1, 2, 3], [3, 2, 1], False),
    ],
)
def test_disagreement_is_about_membership_and_not_order(
    a: list[int], b: list[int], differ: bool
) -> None:
    """Reordering the same documents is not a disagreement about the set.

    This keeps section 7.3's set rate and the FKS ordering distance separate
    statistics. The function takes no `k`, so truncation is the caller's job.
    """
    assert top_k_disagreement(a, b) is differ


def test_two_empty_sets_are_not_distant_rather_than_undefined() -> None:
    """`0/0` for the Jaccard is zero, not NaN.

    Two empty candidate sets agree completely, which is a fact rather than an
    absence of evidence.
    """
    assert jaccard_distance([], []) == 0.0


def test_disjoint_sets_are_maximally_distant() -> None:
    assert jaccard_distance([1, 2], [3, 4]) == 1.0
