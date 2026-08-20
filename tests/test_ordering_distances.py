"""Ordering distances (README sections 4.5 and 7.3, ``spec_addenda.md#g2``).

The generalised Kendall distance is not a metric at any penalty. An earlier
draft of G2 claimed ``p = 1/2`` made it one;
:func:`test_fks_is_a_near_metric_not_a_metric` pins the counterexample. The
penalty is chosen on bias grounds: ``1/2`` is the unbiased contribution for a
pair whose relative order a list says nothing about.

``K_int`` (Kendall restricted to the intersection) is blind to membership
change, which is the effect section 7.3 measures, so it is never reported alone.
See :func:`test_intersection_kendall_is_blind_to_membership_change`.
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
    """Differing sets are the signal to reach for FKS, so this raises."""
    with pytest.raises(ValueError, match="same set"):
        kendall_tau_distance([1, 2], [1, 3])


def test_kendall_tau_satisfies_the_triangle_inequality() -> None:
    """Unlike FKS, the same-set version genuinely is a metric."""
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
    """A case the obvious ``if k < 2: return 0`` guard gets wrong.

    Two disjoint one-element lists still contribute a case-3 pair, so the
    maximum is 1 and the normalised distance is 1.0.
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
    """Pins the correction to G2.

    An earlier draft claimed ``p = 1/2`` makes ``K^(p)`` a metric. It does not,
    at any penalty. Witness: ``A`` and ``C`` are disjoint so their distance is
    maximal, ``B`` shares one element with each, and the triangle inequality
    fails by 4 (12 against 6 + 2).

    Bounded distortion does hold, which suffices for reporting disagreement
    rates, but the quantity must never be clustered on or treated as a norm.
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
    """Case 2, worked by hand.

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
    assert jaccard_distance([1, 2], [1, 2]) == 0.0
    assert jaccard_distance([1, 2], [3, 4]) == 1.0
    assert jaccard_distance([], []) == 0.0
    assert jaccard_distance([1, 2], [2, 3]) == pytest.approx(2 / 3)


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
    """Why ``K_int`` is never reported alone.

    These two lists share only their first element, so the intersection Kendall
    is undefined (fewer than two shared elements) and reads as "no reordering"
    to a careless consumer. The set indicator and the FKS distance both see the
    change.
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
# kendall_tau_distance: the precondition the guard does not quite cover
# ---------------------------------------------------------------------------
def test_two_orderings_of_the_same_multiset_but_different_multiplicity_are_not_caught() -> None:
    """A latent defect, pinned rather than repaired.

    The guard compares lengths and `set()`s. `[1, 1, 2]` and `[1, 2, 2]` pass
    both -- same length, same set -- and then the `position` dict keeps only the
    last index of each repeated element, so the inversion count is computed
    against a mapping that lost a document. The result is `0.0`: two orderings
    reported as identical when they are not.

    Not reachable from this package, where both arguments come from a `Ranking`
    whose order is a permutation and therefore duplicate-free. A `Counter`
    comparison would close it; the guard as written documents "same set" and
    delivers it, so the gap is in the precondition rather than in the code.
    """
    left, right = [1, 1, 2], [1, 2, 2]
    assert len(left) == len(right), "the length half of the guard passes"
    assert set(left) == set(right), "and so does the membership half"
    assert kendall_tau_distance(left, right) == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [([1, 2], [1, 2, 3]), ([1, 2, 3], [1, 2]), ([1, 2], [3, 4]), ([], [1])],
)
def test_orderings_over_different_sets_are_refused_towards_the_right_function(
    a: list[int], b: list[int]
) -> None:
    """Different lengths and different memberships both refuse, and the message
    names `kendall_fks` -- because a caller comparing top-k lists that may
    differ in membership has the wrong function, not bad data."""
    with pytest.raises(ValueError, match="use kendall_fks"):
        kendall_tau_distance(a, b)


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_elements_have_no_pair_to_disagree_about(n: int) -> None:
    """`0.0` rather than NaN: there is genuinely no discordance, which is
    different from the quantity being undefined."""
    items = list(range(n))
    assert kendall_tau_distance(items, items) == 0.0


@pytest.mark.parametrize("n", [2, 3, 8, 9])
def test_a_reversed_ordering_is_exactly_one(n: int) -> None:
    """The normaliser is `C(n, 2)`, which is the number of pairs, so a full
    reversal saturates it exactly -- bit for bit, at both odd and even sizes
    where the merge in the inversion count splits differently."""
    forwards = list(range(n))
    assert same_bits(kendall_tau_distance(forwards, forwards[::-1]), 1.0)


# ---------------------------------------------------------------------------
# inversion_count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 9, 16, 17])
def test_a_reversed_sequence_has_every_pair_inverted(n: int) -> None:
    """`n(n-1)/2`. Odd and even sizes straddle the merge split, which is where a
    recursive count most easily loses or double-counts a pair."""
    assert inversion_count(list(range(n, 0, -1))) == n * (n - 1) // 2


def test_equal_neighbours_are_not_inversions() -> None:
    """The merge keeps equal elements in order, so a run of ties contributes
    nothing. Otherwise a ranking with tied scores would report disagreement with
    itself."""
    assert inversion_count([1, 1, 1]) == 0
    assert inversion_count([2, 1, 1]) == 2


# ---------------------------------------------------------------------------
# fks_max: the closed form across its domain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [-1, -(2**20), 0])
def test_a_non_positive_list_length_has_no_maximum(k: int) -> None:
    """Only `k = 0` genuinely has no pairs; a negative k is nonsense, and both
    answer `0.0` rather than producing a negative maximum that would make every
    normalised distance negative."""
    assert fks_max(k) == 0.0


@pytest.mark.parametrize("penalty", [0.0, 0.5, 1.0, -1.0])
def test_a_single_pair_of_disjoint_singletons_is_one_whatever_the_penalty(
    penalty: float,
) -> None:
    """`k = 1` is not degenerate: two disjoint singletons contribute exactly one
    case-3 pair, and case 3 is the one the penalty does not touch. So the
    maximum is 1 for every penalty, including the absurd ones.
    """
    assert fks_max(1, penalty) == 1.0


def test_an_infinite_penalty_makes_the_singleton_maximum_undefined() -> None:
    """The one penalty that breaks the rule above. At `k = 1` the case-4 term is
    `p * 1 * 0`, and `inf * 0` is NaN rather than zero -- so the maximum comes
    out undefined and every normalised distance against it would follow.

    Pinned as the boundary of the "any penalty" claim. `FKS_PENALTY` is a pinned
    constant and no caller computes one, so this is unreachable in practice.
    """
    assert math.isnan(fks_max(1, math.inf))
    assert fks_max(2, math.inf) == math.inf, "at k >= 2 the term is finite times inf"


@pytest.mark.parametrize(("k", "expected"), [(2, 5.0), (3, 12.0), (50, 3725.0)])
def test_the_closed_form_is_k_times_three_k_minus_one_over_two(k: int, expected: float) -> None:
    """At the pinned penalty of one half. G2 notes `k <= 50` gives at most 1225
    pairs, so 50 is the largest size the protocol produces."""
    assert fks_max(k) == expected
    assert fks_max(k) == k * (3 * k - 1) / 2


def test_the_penalty_scales_only_the_same_list_pairs() -> None:
    """`k^2 + p * k * (k-1)`: the first term is case 3 and is fixed, the second
    is case 4 and is what the penalty weights. At `p = 0` only case 3 survives.
    """
    assert fks_max(2, 0.0) == 4.0, "k^2 alone"
    assert fks_max(2, 1.0) == 6.0, "case 4 at full weight"
    assert fks_max(2, 0.5) == 5.0, "the pinned half"


# ---------------------------------------------------------------------------
# kendall_fks: the degenerate list shapes
# ---------------------------------------------------------------------------
def test_two_empty_lists_are_not_distant() -> None:
    """No union, no pairs, so no disagreement -- and the normaliser is zero, so
    the guard has to return before dividing."""
    assert kendall_fks([], []) == 0.0


def test_one_empty_list_is_distant_from_a_non_empty_one() -> None:
    """Every element of the non-empty list is absent from the other, which is
    case 2 rather than case 3, so the distance is positive but well below the
    disjoint maximum."""
    distance = kendall_fks([1, 2], [])
    assert 0.0 < distance < 1.0


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_disjoint_lists_of_equal_length_attain_exactly_one(k: int) -> None:
    """The normaliser is `fks_max(k)`, and disjoint lists are what it is the
    maximum over -- so the normalised distance is exactly 1, not merely close.
    """
    left = list(range(k))
    right = list(range(k, 2 * k))
    assert same_bits(kendall_fks(left, right), 1.0)


def test_the_unnormalised_form_returns_the_pair_count_itself() -> None:
    """Useful when comparing lists of different k, where dividing by a maximum
    that differs between the two would not be meaningful."""
    assert kendall_fks([0], [1], normalise=False) == fks_max(1)
    assert kendall_fks([0, 1], [2, 3], normalise=False) == fks_max(2)


# ---------------------------------------------------------------------------
# compare_top_k: the prefix length, and a counter that floor-divides
# ---------------------------------------------------------------------------
def test_comparing_no_documents_at_all_reports_a_wholly_degenerate_row() -> None:
    """`k = 0` is a legitimate grid point with nothing in it. Every field has to
    take the value that means "no evidence" rather than "no disagreement":
    notably the intersection Kendall is NaN, since there are no pairs.
    """
    result = compare_top_k([1, 2, 3], [3, 2, 1], 0)

    assert result.sets_differ is False
    assert result.intersection_size == 0
    assert result.fks == 0.0
    assert math.isnan(result.kendall_intersection)


def test_a_negative_prefix_length_silently_drops_from_the_end() -> None:
    """`a[:-1]` is a legal slice, so a `k` that arrived negative compares almost
    the whole lists and reports a `k` of -1 alongside. The third instance of the
    same trap in this package, after `Ranking.top_k` and `short`.
    """
    result = compare_top_k([1, 2, 3], [3, 2, 1], -1)
    assert result.k == -1
    assert result.intersection_size == 1, "it compared the first two of each"


def test_an_odd_symmetric_difference_reports_no_swaps_at_all() -> None:
    """A latent defect, pinned.

    `swapped` is `len(sa ^ sb) // 2`, which assumes membership changes come in
    pairs -- one document leaving as another arrives. When the two prefixes have
    different lengths the symmetric difference is odd, and the floor division
    reports zero swaps for a comparison that simultaneously reports the sets as
    differing.

    Reachable whenever a fold's candidate set is smaller than `k`, which G19
    says happens by protocol rather than by accident.
    """
    result = compare_top_k([1, 2, 3], [1, 2], 3)

    assert result.sets_differ is True
    assert result.swapped == 0, "one document differs, and half of one is none"
    assert result.intersection_size == 2


def test_a_swap_of_one_document_for_another_counts_as_one() -> None:
    """The case the counter is written for: equal-length prefixes where the
    symmetric difference is even."""
    result = compare_top_k([1, 2, 3], [1, 2, 4], 3)
    assert result.swapped == 1
    assert result.sets_differ is True


@pytest.mark.parametrize("k", [1, 2, 3, 4, 99])
def test_a_prefix_longer_than_the_lists_compares_what_there_is(k: int) -> None:
    """Slicing past the end is not an error, so a k beyond the candidate count
    compares the whole lists -- which is the lenient clamping G3 describes,
    arriving here by way of Python rather than by a guard."""
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
    """Reordering the same documents is not a disagreement about the set. That
    is what makes section 7.3's set rate and the FKS ordering distance separate
    statistics rather than two views of one number.

    The lists are already prefixes: the function takes no `k` and compares what
    it is given, so truncation is the caller's job.
    """
    assert top_k_disagreement(a, b) is differ


def test_two_empty_sets_are_not_distant_rather_than_undefined() -> None:
    """`0/0` for the Jaccard. Zero rather than NaN: two empty candidate sets
    agree completely, which is a fact rather than an absence of evidence."""
    assert jaccard_distance([], []) == 0.0


def test_disjoint_sets_are_maximally_distant() -> None:
    assert jaccard_distance([1, 2], [3, 4]) == 1.0
