"""Reference vs native for the ranking layer: permutation identity.

A ranking is a sequence of ``int32``, so the claim here is element-by-element
equality of the two orders. Margins and sorted scores are floats again and go
back to the bitwise standard.

Permutation identity holds under four separately testable conditions:

1. the sort key is injective: identifier ranks are a bijection, which the native
   constructor validates and refuses to proceed without;
2. the key inputs are identical: scores are bit-exact by the scoring
   differential tests, and the integer ranks cross the boundary as data with no
   re-derivation on the native side;
3. the comparison relation is the same in both languages: IEEE ``<`` on finite
   doubles, with negation (a sign-bit flip) the only arithmetic applied;
4. the build is not fast-math, per ``test_build_is_reproducible``.

A uniform-random double vector contains a tie with probability near zero, so a
differential test over random scores exercises none of the tie-break and passes
with a completely broken attribute table. Scores here come from a small discrete
alphabet, with a large all-zero block standing in for the zero-norm documents
short-text corpora produce in bulk.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from tfidf_stability._native import native_available, unavailable_reason
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.margins import boundary_margin, min_adjacent_margin_top
from tfidf_stability.ranking.ranker import rank, rank_top_k, sorted_scores_desc
from tfidf_stability.ranking.sort_keys import SortKeySpec
from tfidf_stability.ranking.tie_groups import (
    chain_inflation_ratio,
    tie_ball_interval,
    tie_chains,
    tie_cliques,
)
from tfidf_stability.utils.numerics import same_bits
from tfidf_stability.utils.validation import KOutOfRangeError, StrictMode

pytestmark = [
    pytest.mark.native,
    pytest.mark.differential,
    pytest.mark.skipif(not native_available(), reason=unavailable_reason() or "no native backend"),
]

if native_available():
    from tfidf_stability._native import _tfidf_native as nat  # type: ignore[attr-defined]

ALPHABET = (0.0, 0.25, 0.5, 0.75)
ATTRS = ("popularity", "rating", "engagement")
LENIENT = StrictMode.LENIENT


def tie_heavy(rng: random.Random, n: int) -> tuple[list[float], AttributeTable]:
    """Scores from a discrete alphabet plus a zero block, and a matching table."""
    zeros = n // 5
    scores = [0.0] * zeros + [rng.choice(ALPHABET) for _ in range(n - zeros)]
    rng.shuffle(scores)
    records = [
        {
            "doc_id": f"d{i:05d}",
            "popularity": rng.randrange(4),
            "rating_sum2": rng.randrange(2, 11),
            "rating_count": rng.randrange(1, 4),
            "engagement": rng.randrange(3),
        }
        for i in range(n)
    ]
    return scores, AttributeTable.from_records(records)


def native_ranker(table: AttributeTable, priority: tuple[str, ...]):  # type: ignore[no-untyped-def]
    """Build a NativeRanker from the same ranks the reference will use.

    Nothing is recomputed natively: the rank encoding turns a question about
    comparing rationals into integer equality.
    """
    flat: list[int] = []
    for name in ATTRS:
        flat.extend(table.column(name).ranks)
    return nat.NativeRanker(
        np.array(flat, dtype=np.int32),
        np.array(table.id_ranks, dtype=np.int32),
        np.array([ATTRS.index(p) for p in priority], dtype=np.int32),
        len(ATTRS),
    )


# ---------------------------------------------------------------------------
# The headline claim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "priority",
    [ATTRS, (), ("engagement", "rating", "popularity")],
    ids=["pi", "pi_score", "pi_alt"],
)
def test_native_permutation_is_identical(priority: tuple[str, ...]) -> None:
    rng = random.Random(4242)
    spec = SortKeySpec("under_test", priority)
    compared = 0
    for _ in range(40):
        n = rng.randint(1, 120)
        scores, table = tie_heavy(rng, n)
        expected = rank(scores, table, spec).order
        got = native_ranker(table, priority).rank(
            np.array(scores, dtype=np.float64), int(nat.SELECTION["full_sort"])
        )
        assert tuple(int(x) for x in got) == expected
        compared += n
    assert compared > 1000


def test_the_inputs_really_do_contain_ties() -> None:
    """Guards the guard: if the generator stopped producing ties, every test in
    this file would keep passing while testing nothing."""
    rng = random.Random(4242)
    scores, _ = tie_heavy(rng, 200)
    assert len(set(scores)) < len(scores) / 10, "expected heavy tying"
    assert scores.count(0.0) > 20, "expected a substantial zero block"


def test_all_native_selection_strategies_agree() -> None:
    """The ranking analogue of ``TAAT == DAAT``: unrelated algorithms, one answer.

    Sound because the key is injective: no two documents compare equal, so the
    "stable" clause is vacuous.
    """
    rng = random.Random(777)
    scores, table = tie_heavy(rng, 90)
    ranker = native_ranker(table, ATTRS)
    arr = np.array(scores, dtype=np.float64)
    reference = tuple(int(x) for x in ranker.rank(arr, int(nat.SELECTION["full_sort"])))
    for name in ("stable_sort", "bounded_heap"):
        assert tuple(int(x) for x in ranker.rank(arr, int(nat.SELECTION[name]))) == reference


def test_native_top_k_matches_the_reference_prefix() -> None:
    rng = random.Random(31)
    scores, table = tie_heavy(rng, 80)
    ranker = native_ranker(table, ATTRS)
    arr = np.array(scores, dtype=np.float64)
    spec = SortKeySpec("under_test", ATTRS)
    for k in (1, 5, 20, 79):
        expected = rank_top_k(scores, table, spec, k=k).order
        for name in ("partial_sort", "nth_element"):
            got = ranker.top_k(arr, len(expected), int(nat.SELECTION[name]))
            assert tuple(int(x) for x in got) == expected


# ---------------------------------------------------------------------------
# Floats: back to bit-for-bit
# ---------------------------------------------------------------------------
def test_native_sorted_scores_are_bit_exact() -> None:
    rng = random.Random(5)
    scores, _ = tie_heavy(rng, 200)
    got = nat.sorted_scores_desc(np.array(scores, dtype=np.float64))
    assert all(same_bits(a, b) for a, b in zip(sorted_scores_desc(scores), got, strict=True))


def test_native_margins_are_bit_exact() -> None:
    rng = random.Random(6)
    for _ in range(25):
        scores, _ = tie_heavy(rng, rng.randint(2, 50))
        s = sorted_scores_desc(scores)
        arr = np.array(s, dtype=np.float64)
        for k in range(1, len(s) + 1):
            ref = boundary_margin(s, k, mode=LENIENT)
            value, defined, k_eff = nat.boundary_margin(arr, k)
            assert bool(defined) == ref.defined
            assert k_eff == ref.k_effective
            if ref.defined:
                assert same_bits(value, ref.value)
            else:
                assert math.isnan(value)

            ref_min = min_adjacent_margin_top(s, k, mode=LENIENT)
            v2, d2, _ = nat.min_adjacent_margin_top(arr, k)
            assert bool(d2) == ref_min.defined
            if ref_min.defined:
                assert same_bits(v2, ref_min.value)


# ---------------------------------------------------------------------------
# Tie groups
# ---------------------------------------------------------------------------
def test_native_tie_groups_are_identical() -> None:
    """Balls, chains, cliques and rho, at tau values that straddle the gaps."""
    rng = random.Random(909)
    for _ in range(20):
        scores, _ = tie_heavy(rng, rng.randint(2, 50))
        s = sorted_scores_desc(scores)
        arr = np.array(s, dtype=np.float64)
        for tau in (0.0, 1e-12, 0.1, 0.25, 0.3):
            for j in range(len(s)):
                got = tuple(int(x) for x in nat.tie_ball_interval(arr, j, tau))
                assert got == tie_ball_interval(s, j, tau)

            flat = [int(x) for x in nat.tie_chains(arr, tau)]
            assert tuple(zip(flat[::2], flat[1::2], strict=True)) == tie_chains(s, tau)

            flat = [int(x) for x in nat.tie_cliques(arr, tau)]
            assert tuple(zip(flat[::2], flat[1::2], strict=True)) == tie_cliques(s, tau)

            assert same_bits(nat.chain_inflation_ratio(arr, tau), chain_inflation_ratio(s, tau))


def test_native_ladder_reproduces_the_non_transitivity() -> None:
    """G1's witness, across the language boundary."""
    tau = 2.0**-20
    s = np.array([(5 - i) * tau for i in range(6)], dtype=np.float64)
    assert tuple(int(x) for x in nat.tie_ball_interval(s, 1, tau)) == (0, 3)
    assert tuple(int(x) for x in nat.tie_ball_interval(s, 0, tau)) == (0, 2)
    assert nat.chain_inflation_ratio(s, tau) == 3.0


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------
def test_native_ranker_rejects_non_finite_scores() -> None:
    """G3 requires a re-check here: it is the last line of defence against
    undefined behaviour inside ``std::sort``."""
    rng = random.Random(1)
    scores, table = tie_heavy(rng, 8)
    ranker = native_ranker(table, ATTRS)
    bad = np.array(scores, dtype=np.float64)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ranker.rank(bad, 0)


def test_native_ranker_rejects_non_bijective_identifier_ranks() -> None:
    """Injectivity of the key is the precondition of permutation identity, so
    the constructor refuses to build a ranker that cannot deliver it."""
    with pytest.raises(ValueError, match="bijection"):
        nat.NativeRanker(
            np.array([0, 0], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),  # not a bijection
            np.array([0], dtype=np.int32),
            1,
        )


def test_native_ranker_rejects_a_mismatched_score_count() -> None:
    rng = random.Random(2)
    _, table = tie_heavy(rng, 6)
    ranker = native_ranker(table, ATTRS)
    with pytest.raises(ValueError, match="does not match"):
        ranker.rank(np.array([0.1, 0.2], dtype=np.float64), 0)


def test_native_ranker_rejects_an_unknown_attribute() -> None:
    rng = random.Random(3)
    _, table = tie_heavy(rng, 4)
    flat: list[int] = []
    for name in ATTRS:
        flat.extend(table.column(name).ranks)
    with pytest.raises(ValueError, match="does not exist"):
        nat.NativeRanker(
            np.array(flat, dtype=np.int32),
            np.array(table.id_ranks, dtype=np.int32),
            np.array([99], dtype=np.int32),
            len(ATTRS),
        )


def test_the_two_backends_disagree_only_on_invalid_k_and_only_in_kind() -> None:
    """Pin the one place the backends part company: ``k = 0``.

    For every valid k they agree bit-for-bit, per the tests above. At ``k = 0``
    they differ in kind: the reference raises ``KOutOfRangeError`` (``resolve_k``
    rejects non-positive k in strict and lenient modes alike, treating zero as a
    nonsensical rank), while the native margin functions return an undefined
    margin.

    Nothing in the package passes k = 0, so the divergence is latent. Pinned
    because if it widened, the reference would refuse while the native path
    returned a NaN that serialises to ``null`` in a results file.
    """
    scores = np.array([1.0, 0.5, 0.25], dtype=np.float64)

    for native_fn, reference_fn in (
        (nat.boundary_margin, boundary_margin),
        (nat.min_adjacent_margin_top, min_adjacent_margin_top),
    ):
        value, defined, _ = native_fn(scores, 0)
        assert math.isnan(value)
        assert not defined

        for mode in (StrictMode.STRICT, StrictMode.LENIENT):
            with pytest.raises(KOutOfRangeError, match="k must be positive, got 0"):
                reference_fn([1.0, 0.5, 0.25], 0, mode=mode)


# ---------------------------------------------------------------------------
# Signed zeros: the one case where sort stability is observable
# ---------------------------------------------------------------------------
# `-0.0 == 0.0` is true while the two differ in bits, so an unstable sort may
# return them in either order. The reference is `sorted(..., reverse=True)`,
# which is stable; the core used `std::sort`, which is not, and the two
# disagreed.
#
# Every fixture in this file was under sixteen elements, and libstdc++'s
# introsort falls back to insertion sort below that -- incidentally stable. So
# the suite agreed everywhere and could not have caught it. Seventeen is where
# it started.
_SIGNED_ZERO_SIZES = (17, 33, 64, 129)


@pytest.mark.parametrize("size", _SIGNED_ZERO_SIZES)
def test_sorting_signed_zeros_agrees_bit_for_bit_with_the_reference(size: int) -> None:
    """Stability is observable here, so the two sorts must make the same choice.

    Asserted on bit patterns rather than values, because every value in this
    array is zero and `==` cannot tell the two apart -- which is the whole
    reason an unstable sort could diverge here unnoticed.
    """
    scores = [(-0.0 if i % 2 else 0.0) for i in range(size)]

    reference = sorted_scores_desc(scores)
    native = list(nat.sorted_scores_desc(np.asarray(scores, dtype=np.float64)))

    assert len(native) == len(reference)
    for i, (a, b) in enumerate(zip(reference, native, strict=True)):
        assert same_bits(a, b), (
            f"position {i}: reference {a!r} and native {b!r} are equal but not "
            f"bit-identical, so the two sorts ordered the signed zeros differently"
        )


def test_the_margin_over_signed_zeros_agrees_bit_for_bit() -> None:
    """The consequence, and why the sort mattered rather than being cosmetic.

    ``m_k`` is a difference of adjacent sorted scores. Over zeros, ``0.0 - 0.0``
    is ``+0.0`` while ``-0.0 - 0.0`` is ``-0.0``, so a different arrangement
    produces a differently-signed zero. Measured before the fix: the two
    backends disagreed in bits at k = 2, 3 and 4.
    """
    scores = [(-0.0 if i % 2 else 0.0) for i in range(17)]
    reference = list(sorted_scores_desc(scores))
    native = list(nat.sorted_scores_desc(np.asarray(scores, dtype=np.float64)))

    checked = 0
    for k in (1, 2, 3, 4):
        a = boundary_margin(reference, k, mode=StrictMode.LENIENT).value
        b = boundary_margin(native, k, mode=StrictMode.LENIENT).value
        assert same_bits(a, b), f"m_{k} differs in bits: {a!r} against {b!r}"
        checked += 1

    assert checked == 4, "every k was compared"
