"""Reference vs native for the ordering distances: bit-exact equivalence.

The claim is the usual one -- the C++ mirror performs *the same floating-point
operations in the same order* as the normative Python reference -- but the two
halves of this module reach it by different routes, and the distinction matters
when a failure has to be diagnosed.

``inversion_count`` and the set measures are **integer** quantities dressed up as
floats only at the last division, so agreement there is combinatorial and any
divergence is a logic bug.

``kendall_fks`` is a **float accumulation**, and that is where a plausible
implementation goes quietly wrong. Its total is a sum of ones and penalties over
the pairs of the union, so it depends on the order in which the union is walked.
At ``p = 1/2`` every addend is dyadic and the sum is exact whatever the order --
which means the default penalty *cannot* expose a mis-ordered enumeration. The
tests therefore also drive penalties that are not dyadic (``1/3``, ``0.1``),
where the additions genuinely round and the reference's first-appearance
enumeration order is the only one that reproduces its bits.

NaN is compared as NaN, not as a bit pattern. ``kendall_intersection`` is
undefined below two shared elements, and "undefined" is the contract; asserting a
particular quiet-NaN payload would be asserting something the reference does not
promise.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from tfidf_stability._native import native_available, unavailable_reason
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

pytestmark = [
    pytest.mark.native,
    pytest.mark.differential,
    pytest.mark.skipif(not native_available(), reason=unavailable_reason() or "no native backend"),
]

if native_available():
    from tfidf_stability._native import _tfidf_native as nat  # type: ignore[attr-defined]

#: Penalties that are *not* dyadic rationals, so the accumulation rounds and the
#: enumeration order becomes observable. Without these the suite would pass on an
#: implementation that walked the union in sorted order instead.
ROUNDING_PENALTIES = (1.0 / 3.0, 0.1, 0.7)


def ids(seq: list[int]) -> np.ndarray:
    """Document identifiers as the int32 array the binding takes."""
    return np.array(seq, dtype=np.int32)


def top_k_pair(rng: random.Random, pool: int = 12) -> tuple[list[int], list[int]]:
    """Two overlapping top-k lists over a small pool.

    The pool is deliberately tight. Two lists sampled from thousands of documents
    would be disjoint almost every time, exercising only FKS case 3 and 4 and
    never case 1 or 2 -- and case 2 is where the "present outranks absent"
    convention lives.
    """
    a = rng.sample(range(pool), rng.randint(0, 6))
    b = rng.sample(range(pool), rng.randint(0, 6))
    return a, b


# ---------------------------------------------------------------------------
# Integer core
# ---------------------------------------------------------------------------
def test_inversion_count_is_identical() -> None:
    rng = random.Random(11)
    for _ in range(200):
        n = rng.randint(0, 60)
        # A small alphabet, so equal elements -- which are NOT inversions -- are
        # the rule rather than the exception.
        seq = [rng.randrange(5) for _ in range(n)]
        assert nat.inversion_count(ids(seq)) == inversion_count(seq)


def test_inversion_count_on_a_large_tie_group() -> None:
    """Tie groups are not bounded by k, so this path really does run long."""
    rng = random.Random(12)
    seq = [rng.randrange(3) for _ in range(20_000)]
    assert nat.inversion_count(ids(seq)) == inversion_count(seq)


def test_set_measures_are_identical() -> None:
    rng = random.Random(13)
    for _ in range(300):
        a, b = top_k_pair(rng)
        assert bool(nat.top_k_disagreement(ids(a), ids(b))) is top_k_disagreement(a, b)
        assert same_bits(nat.jaccard_distance(ids(a), ids(b)), jaccard_distance(a, b))


# ---------------------------------------------------------------------------
# Same-set Kendall tau
# ---------------------------------------------------------------------------
def test_kendall_tau_is_bit_exact() -> None:
    rng = random.Random(14)
    for _ in range(300):
        n = rng.randint(0, 40)
        a = rng.sample(range(100), n)
        b = a[:]
        rng.shuffle(b)
        assert same_bits(nat.kendall_tau_distance(ids(a), ids(b)), kendall_tau_distance(a, b))


def test_kendall_tau_refuses_differing_sets_on_both_sides() -> None:
    """The refusal is part of the contract, so it is mirrored, not just the value."""
    with pytest.raises(ValueError, match="same set"):
        kendall_tau_distance([1, 2], [1, 3])
    with pytest.raises(ValueError, match="same set"):
        nat.kendall_tau_distance(ids([1, 2]), ids([1, 3]))


# ---------------------------------------------------------------------------
# FKS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("penalty", [0.0, FKS_PENALTY, 1.0, *ROUNDING_PENALTIES])
def test_kendall_fks_is_bit_exact(penalty: float) -> None:
    rng = random.Random(15)
    checked = 0
    for _ in range(400):
        a, b = top_k_pair(rng)
        for normalise in (False, True):
            got = nat.kendall_fks(ids(a), ids(b), penalty, normalise)
            expected = kendall_fks(a, b, penalty, normalise=normalise)
            assert same_bits(got, expected), f"a={a} b={b} p={penalty} norm={normalise}"
            checked += 1
    assert checked == 800


def test_the_inputs_really_do_exercise_every_fks_case() -> None:
    """Guards the guard.

    If the generator drifted towards disjoint lists, the tests above would keep
    passing while never reaching case 1 or case 2 -- the only cases where the
    two implementations could disagree about *ordering* rather than membership.
    """
    rng = random.Random(15)
    seen = {"shared_pair": 0, "case_two": 0, "case_three": 0, "case_four": 0}
    for _ in range(400):
        a, b = top_k_pair(rng)
        sa, sb = set(a), set(b)
        both = sa & sb
        seen["shared_pair"] += len(both) >= 2
        seen["case_two"] += bool(len(sa) >= 2 and len(both) == 1)
        seen["case_three"] += bool(sa - sb and sb - sa)
        seen["case_four"] += len(sa - sb) >= 2 or len(sb - sa) >= 2
    assert all(count > 20 for count in seen.values()), seen


def test_fks_max_is_bit_exact_including_the_k_equals_one_trap() -> None:
    for k in range(0, 200):
        for penalty in (0.0, FKS_PENALTY, 1.0, *ROUNDING_PENALTIES):
            assert same_bits(nat.fks_max(k, penalty), fks_max(k, penalty))
    # Pinned separately because an early guard returned 0.0 here, which
    # normalised two entirely disjoint singleton lists to distance zero.
    assert nat.fks_max(1, FKS_PENALTY) == 1.0
    assert nat.kendall_fks(ids([1]), ids([2]), FKS_PENALTY, True) == 1.0


def test_the_near_metric_violation_survives_the_language_boundary() -> None:
    """G2's witness, evaluated natively. Not a bug: do not "fix" it."""
    a, b, c = ids([3, 1, 0]), ids([5, 3, 4]), ids([5, 4, 2])
    d_ab = nat.kendall_fks(a, b, FKS_PENALTY, False)
    d_bc = nat.kendall_fks(b, c, FKS_PENALTY, False)
    d_ac = nat.kendall_fks(a, c, FKS_PENALTY, False)
    assert (d_ab, d_bc, d_ac) == (6.0, 2.0, 12.0)
    assert d_ac > d_ab + d_bc


# ---------------------------------------------------------------------------
# The combined comparison
# ---------------------------------------------------------------------------
def test_compare_top_k_is_field_for_field_identical() -> None:
    rng = random.Random(16)
    undefined = 0
    for _ in range(400):
        a, b = top_k_pair(rng)
        for k in (0, 1, 2, 3, 6, 20):
            expected = compare_top_k(a, b, k)
            k_out, differ, fks, k_int, size, jac, swapped = nat.compare_top_k(ids(a), ids(b), k)

            assert k_out == expected.k
            assert bool(differ) is expected.sets_differ
            assert same_bits(fks, expected.fks)
            assert size == expected.intersection_size
            assert same_bits(jac, expected.jaccard)
            assert swapped == expected.swapped

            # NaN is the contract, not a payload: compare definedness, and bits
            # only where a value is actually claimed.
            if math.isnan(expected.kendall_intersection):
                assert math.isnan(k_int)
                undefined += 1
            else:
                assert same_bits(k_int, expected.kendall_intersection)
    assert undefined > 100, "the undefined branch must actually be reached"


def test_compare_top_k_rejects_a_negative_k() -> None:
    """Python would slice from the end; the binding refuses rather than guess."""
    with pytest.raises(ValueError, match="non-negative"):
        nat.compare_top_k(ids([1, 2]), ids([2, 1]), -1)
