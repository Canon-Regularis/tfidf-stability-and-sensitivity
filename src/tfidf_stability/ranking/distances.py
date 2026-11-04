"""Ordering distances (README sections 4.5 and 7.3, ``spec_addenda.md#g2``).

Tie groups (4.5) use :func:`kendall_tau_distance`; top-k lists (7.3) use :func:`compare_top_k`,
FKS ``K^(1/2)`` (``spec_addenda.md#g2``). Rationale: docs/engineering_notes.md#fks-distance.
Near-metric test: :func:`~tests.test_ordering_distances.test_fks_is_a_near_metric_not_a_metric`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

__all__ = [
    "FKS_PENALTY",
    "TopKComparison",
    "compare_top_k",
    "fks_max",
    "inversion_count",
    "jaccard_distance",
    "kendall_fks",
    "kendall_tau_distance",
    "top_k_disagreement",
]

#: The Fagin-Kumar-Sivakumar case-4 penalty: both elements of a pair in one top-k list, neither in
#: the other. ``p = 1/2`` is the unbiased estimate for a pair a list says nothing about; ``p = 0``
#: biases downwards and ``p = 1`` upwards. ``K^(p)`` is a near-metric; the triangle inequality
#: fails at every penalty tested (G2). Rationale: docs/engineering_notes.md#fks-penalty
FKS_PENALTY: Final[float] = 0.5


# ---------------------------------------------------------------------------
# Same-set Kendall tau
# ---------------------------------------------------------------------------
def inversion_count(sequence: Sequence[int]) -> int:
    """Number of pairs ``i < j`` with ``sequence[i] > sequence[j]``.

    Merge sort, ``O(n log n)``. Tie groups are not bounded by ``k``: the zero-score
    block of a short-text corpus can be a large fraction of it, so the asymptotics
    matter here in a way they do not in :func:`kendall_fks`.
    """
    work = list(sequence)
    buffer = [0] * len(work)

    def sort_count(lo: int, hi: int) -> int:
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        total = sort_count(lo, mid) + sort_count(mid, hi)
        i, j, out = lo, mid, lo
        while i < mid and j < hi:
            if work[i] <= work[j]:
                buffer[out] = work[i]
                i += 1
            else:
                # work[i..mid) all exceed work[j], so each is an inversion.
                total += mid - i
                buffer[out] = work[j]
                j += 1
            out += 1
        while i < mid:
            buffer[out] = work[i]
            i += 1
            out += 1
        while j < hi:
            buffer[out] = work[j]
            j += 1
            out += 1
        work[lo:hi] = buffer[lo:hi]
        return total

    return sort_count(0, len(work))


def kendall_tau_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """Normalised Kendall tau distance between two orderings of the *same* set.

    ``a`` and ``b`` are orderings, best first, of the same multiset. Returns
    ``#discordant / C(n, 2)`` in ``[0, 1]``, or ``0.0`` below two elements. Raises
    ValueError if the multisets differ; :func:`kendall_fks` handles differing membership.
    """
    # Counter, not set: a set ignores multiplicity, so [1, 1, 2] and [1, 2, 2]
    # would pass a set check, and the position dict below keeps only the last
    # index of each repeat, giving 0.0 for inputs that are not orderings of the
    # same multiset.
    if len(a) != len(b) or Counter(a) != Counter(b):
        raise ValueError(
            "kendall_tau_distance requires two orderings of the same set, with the "
            "same multiplicities; for top-k lists that may differ in membership use "
            "kendall_fks (spec_addenda G2)"
        )
    n = len(a)
    if n < 2:
        return 0.0

    position = {item: i for i, item in enumerate(b)}
    return inversion_count([position[item] for item in a]) / (n * (n - 1) / 2)


# ---------------------------------------------------------------------------
# Fagin-Kumar-Sivakumar generalised Kendall distance
# ---------------------------------------------------------------------------
def _checked_penalty(penalty: float) -> float:
    """Reject a case-4 penalty outside ``[0, 1]``.

    ``not 0.0 <= p <= 1.0`` rather than ``p < 0.0 or p > 1.0``: every comparison
    with NaN is false, so the decomposed form admits NaN. G2 fixes the domain,
    and the normalised ``K-bar`` lies in ``[0, 1]`` only because of it.
    """
    if not 0.0 <= penalty <= 1.0:
        raise ValueError(f"the case-4 penalty must lie in [0, 1], got {penalty!r}")
    return penalty


def fks_max(k: int, penalty: float = FKS_PENALTY) -> float:
    """The maximum value of ``K^(p)`` over two top-k lists, attained when disjoint.

    Disjoint lists give ``k^2`` case-3 pairs at 1 and ``2 * C(k, 2)`` case-4 pairs
    at ``p``, so the maximum is ``k^2 + p * k * (k - 1)``, or ``k(3k - 1) / 2`` at
    ``p = 1/2``. ``k = 1`` gives 1 (one case-3 pair); only ``k = 0`` has no pairs.
    """
    _checked_penalty(penalty)
    if k < 1:
        return 0.0
    return float(k * k) + penalty * float(k) * float(k - 1)


def kendall_fks(
    a: Sequence[int],
    b: Sequence[int],
    penalty: float = FKS_PENALTY,
    *,
    normalise: bool = True,
) -> float:
    """Generalised Kendall distance between two top-k lists (G2b).

    ``a``, ``b`` are best first and may rank different sets; union pairs score by FKS cases 1-4,
    case 4 at ``penalty`` in ``[0, 1]`` (else ValueError). ``normalise`` divides by :func:`fks_max`
    into ``[0, 1]``; equal lists give 0.0. Rationale: docs/engineering_notes.md#fks-cases
    """
    _checked_penalty(penalty)
    union = list(dict.fromkeys([*a, *b]))
    pos_a = {item: i for i, item in enumerate(a)}
    pos_b = {item: i for i, item in enumerate(b)}

    def case_two(both: dict[int, int], other: dict[int, int], x: int, y: int) -> float:
        """`both` ranks x and y; `other` ranks exactly one of them.

        An element present in a list is taken to rank above one absent from it,
        so the two lists disagree when `both` puts the element `other` lacks
        first.
        """
        absent = y if x in other else x
        present = x if absent == y else y
        return 1.0 if both[absent] < both[present] else 0.0

    total = 0.0
    for x, y in combinations(union, 2):
        x_in_a, y_in_a = x in pos_a, y in pos_a
        x_in_b, y_in_b = x in pos_b, y in pos_b

        if x_in_a and y_in_a and x_in_b and y_in_b:  # case 1
            if (pos_a[x] < pos_a[y]) != (pos_b[x] < pos_b[y]):
                total += 1.0
        elif x_in_a and y_in_a and (x_in_b or y_in_b):  # case 2, `a` holds both
            total += case_two(pos_a, pos_b, x, y)
        elif x_in_b and y_in_b and (x_in_a or y_in_a):  # case 2, `b` holds both
            total += case_two(pos_b, pos_a, x, y)
        elif (x_in_a and y_in_a) or (x_in_b and y_in_b):  # case 4
            total += penalty
        else:  # case 3: one element from each list alone
            total += 1.0

    if not normalise:
        return total
    ceiling = fks_max(max(len(a), len(b)), penalty)
    return total / ceiling if ceiling > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Set-level measures
# ---------------------------------------------------------------------------
def top_k_disagreement(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether the two top-k sets differ: section 7.3's headline indicator.

    About the set alone. Section 7.3 measures "the fraction of queries for which
    the top-k set differs", and a pure reordering within an unchanged set is a
    separate phenomenon, captured by the FKS distance.
    """
    return set(a) != set(b)


def jaccard_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """``1 - |A n B| / |A u B|``; ``0.0`` when both sets are empty."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - len(sa & sb) / len(union)


@dataclass(frozen=True, slots=True)
class TopKComparison:
    """Every ordering measure for one pair of top-k lists at one ``k``.

    Each measure is blind to something the others see: ``kendall_intersection``
    covers only shared elements, cannot detect membership change, and alone would
    understate the effect section 7.3 measures.
    """

    k: int
    #: 1[topk(a) != topk(b)]: the headline disagreement indicator.
    sets_differ: bool
    #: Normalised FKS K^(1/2) in [0, 1]: the headline reordering measure.
    fks: float
    #: Plain Kendall tau on the intersection, or NaN when fewer than two
    #: elements are shared.
    kendall_intersection: float
    #: |intersection|. Always reported: ``kendall_intersection`` is
    #: uninterpretable without it.
    intersection_size: int
    jaccard: float
    #: The larger of the two one-way differences: documents that left the top-k,
    #: or entered it, whichever is greater. On equal-length prefixes the two are
    #: equal and this counts swaps. On unequal ones, halving their sum rounds a
    #: one-document difference down to zero while ``sets_differ`` stays true.
    swapped: int


def compare_top_k(a: Sequence[int], b: Sequence[int], k: int) -> TopKComparison:
    """Compare two top-k lists under every measure of G2(b) and G2(c).

    ``k = 0`` compares nothing. A negative ``k`` raises ValueError: ``a[:k]`` would
    drop from the end instead of taking a prefix, and the native binding refuses it.
    ``kendall_intersection`` is :func:`kendall_tau_distance` on the shared elements.
    """
    if k < 0:
        raise ValueError(f"compare_top_k({k}) is not a prefix; k must be non-negative")
    prefix_a, prefix_b = list(a[:k]), list(b[:k])
    sa, sb = set(prefix_a), set(prefix_b)
    shared = sa & sb

    if len(shared) >= 2:
        restricted_a = [d for d in prefix_a if d in shared]
        restricted_b = [d for d in prefix_b if d in shared]
        k_int = kendall_tau_distance(restricted_a, restricted_b)
    else:
        k_int = math.nan

    return TopKComparison(
        k=k,
        sets_differ=sa != sb,
        fks=kendall_fks(prefix_a, prefix_b),
        kendall_intersection=k_int,
        intersection_size=len(shared),
        jaccard=jaccard_distance(prefix_a, prefix_b),
        swapped=max(len(sa), len(sb)) - len(shared),
    )
