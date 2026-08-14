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
