"""Ordering distances (README sections 4.5 and 7.3, ``spec_addenda.md#g2``).

The paper asks for an ordering distance in two places, and they are different
problems:

* §4.5 wants "a distance between orderings restricted to **tie groups**";
* §7.3 wants "**within-top-k** reordering ... restricted to tie-affected subsets".

Conflating them produces a quantity that looks defined and is not.

Restricted to a tie group the problem is easy. A tie group is defined on the
score vector, which pi, pi_score and pi_alt share, so the two orderings contain
the same elements; :func:`kendall_tau_distance` is plain normalised Kendall tau
by merge-sort inversion count, ``O(n log n)``.

Restricted to top-k it is ill-posed. When ``topk(pi) != topk(pi')``, the case
section 7.3 exists to measure, the two lists rank different sets and "the number
of discordant pairs" has no meaning. Resolved here with the
Fagin-Kumar-Sivakumar generalised Kendall distance (*Comparing Top k Lists*,
SIAM J. Discrete Math. 17(1), 2003) at penalty ``p = 1/2``.

``K^(p)`` is a near-metric. Measured against this module's own implementation the
triangle inequality fails at every penalty tested, with the violation growing in
``p``, so ``p = 1/2`` is chosen for being the unbiased contribution when a list
says nothing about a pair. See ``spec_addenda.md#g2`` for the witness and
:func:`~tests.test_ordering_distances.test_fks_is_a_near_metric_not_a_metric`
for the regression guard.

Nothing here should be reported alone. :func:`compare_top_k` returns the whole
set (the set-disagreement indicator section 7.3 asks for, normalised FKS,
intersection Kendall with its support size, Jaccard) because each is blind to
something the others see. ``K_int`` cannot see membership change at all, which is
the effect under study.
"""

from __future__ import annotations

import math
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

#: The Fagin-Kumar-Sivakumar penalty for a pair that appears in one top-k list
#: and in neither position of the other.
#:
#: ``p = 1/2`` is the neutral choice: knowing nothing about the relative order of
#: two elements absent from a list, they disagree with probability one half, so
#: ``1/2`` is the unbiased estimate of the contribution. ``p = 0`` assumes the
#: unseen pair agrees, biasing every measurement downwards, the wrong direction
#: for a study of instability; ``p = 1`` biases upwards.
#:
#: It does not make ``K^(p)`` a metric. Measured on this project's own
#: implementation, the triangle inequality is violated at every penalty tested
#: and the violation grows with ``p`` (``spec_addenda.md#g2``). A near-metric has
#: bounded distortion without the triangle inequality, which suffices for
#: reporting disagreement rates and is why this module never clusters on the
#: quantity or treats it as a norm.
FKS_PENALTY: Final[float] = 0.5


# ---------------------------------------------------------------------------
# Same-set Kendall tau
# ---------------------------------------------------------------------------
def inversion_count(sequence: Sequence[int]) -> int:
    """Number of pairs ``i < j`` with ``sequence[i] > sequence[j]``.

    Merge sort, ``O(n log n)``. The naive ``O(n^2)`` enumeration would do for a
    top-k list of 50, but tie groups are not bounded by ``k``: on short-text
    corpora the zero-score block alone can reach a large fraction of the corpus,
    so the asymptotics matter here in a way they do not in :func:`kendall_fks`.
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

    Args:
        a, b: Orderings, best first. Must contain the same elements.

    Returns:
        ``#discordant / C(n, 2)`` in ``[0, 1]``; ``0.0`` for fewer than two
        elements, where no pair exists to disagree about.

    Raises:
        ValueError: If the two orderings do not rank the same set, which signals
            that :func:`kendall_fks` is the function wanted.
    """
    if len(a) != len(b) or set(a) != set(b):
        raise ValueError(
            "kendall_tau_distance requires two orderings of the same set; for top-k "
            "lists that may differ in membership use kendall_fks (spec_addenda G2)"
        )
    n = len(a)
    if n < 2:
        return 0.0

    position = {item: i for i, item in enumerate(b)}
    return inversion_count([position[item] for item in a]) / (n * (n - 1) / 2)


# ---------------------------------------------------------------------------
# Fagin-Kumar-Sivakumar generalised Kendall distance
# ---------------------------------------------------------------------------
def fks_max(k: int, penalty: float = FKS_PENALTY) -> float:
    """The maximum value of ``K^(p)`` over two top-k lists, attained when disjoint.

    With disjoint lists the union has ``2k`` elements and every pair falls into
    case 3 or case 4:

        case 3 (one element from each list): k^2 pairs, contributing 1 each
        case 4 (both from the same list):    2 * C(k, 2) pairs, contributing p

    so the maximum is ``k^2 + p * k * (k - 1)``, which at ``p = 1/2`` is
    ``k(3k - 1) / 2``.

    ``k = 1`` is not degenerate: two disjoint singleton lists still contribute
    one case-3 pair, and the formula gives 1. Only ``k = 0`` has no pairs.
    """
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

    Every unordered pair drawn from the union of the two lists contributes
    according to how it is witnessed:

    ===== ================================================ ============
    case  condition                                        contribution
    ===== ================================================ ============
    1     both elements appear in both lists               1 if oppositely ordered, else 0
    2     both in one list, exactly one in the other       1 if the list holding both ranks
                                                       the element the other is missing
                                                       first, else 0
    3     one element in each list only                    1
    4     both in one list, neither in the other           ``p``
    ===== ================================================ ============

    Cases 2 and 3 carry the generalisation. Both encode the reading that an
    element present in a list is ranked above one absent from it: case 3 then
    disagrees unavoidably, and case 2 disagrees when the list holding both puts
    the absent element first.

    Direct ``O(k^2)`` enumeration over the union: at ``k <= 50`` that is at most
    4950 pairs, so legibility beats the asymptotics here. The test suite
    cross-checks it against :func:`kendall_tau_distance` where both apply.

    Args:
        a, b: Two top-k lists, best first. May rank different sets.
        penalty: The case-4 penalty ``p``.
        normalise: Divide by :func:`fks_max` to land in ``[0, 1]``.

    Returns:
        The distance. ``0.0`` when both lists are empty or identical.
    """
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

    Reported together because each is blind to something the others see:
    ``kendall_intersection`` restricts to the elements the two lists share and so
    cannot detect membership change, and quoting it alone would understate the
    effect section 7.3 measures.
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
    #: Documents that entered or left the top-k, halved: the number of swaps.
    swapped: int


def compare_top_k(a: Sequence[int], b: Sequence[int], k: int) -> TopKComparison:
    """Compare two top-k lists under every measure of G2(b) and G2(c)."""
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
        swapped=len(sa ^ sb) // 2,
    )
