"""Tie groups and near-tie structure (README section 2.3.3, ``spec_addenda.md#g1``).

Section 2.3.3 defines, for a tolerance ``tau > 0`` and a rank ``j``,

    G_tau(j) = { i : |s_i - score(r_j)| <= tau }

and says documents inside one are "indistinguishable at the level of similarity
scores". The definition is a **ball**, not an equivalence class, and the
difference is not pedantic:

* the relation ``|s_i - s_j| <= tau`` is reflexive and symmetric but **not
  transitive** -- for scores ``{0, tau, 2*tau}`` the first and second are
  related, the second and third are related, the first and third are not;
* balls therefore overlap and do **not** partition the corpus;
* "the tie group of document i" is not well defined;
* and two members of one ball can differ by as much as ``2*tau``, so the quoted
  sentence is not strictly true as written.

G1's resolution is to implement three separately named objects and never
conflate them:

===============  ==========================================  =========  ==========
object           definition                                  cost       partition?
===============  ==========================================  =========  ==========
:func:`tie_ball`  verbatim section 2.3.3                     O(log N)   no
:func:`tie_chains` single-linkage: adjacent gaps <= tau      O(N)       **yes**
:func:`tie_cliques` complete-linkage: diameter <= tau        O(N)       no
===============  ==========================================  =========  ==========

The ball stays the primary reported object, because it is what the paper
defines. Chains are reported alongside it wherever a partition is required, and
the **chain-inflation ratio** ``rho(tau)`` flags when the two have diverged.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from tfidf_stability.ranking.margins import adjacent_gaps
from tfidf_stability.utils.validation import (
    ChainInflationWarning,
    TauExceedsScoreRangeWarning,
)

__all__ = [
    "DEFAULT_RHO_WARN",
    "Interval",
    "TieGroupIndex",
    "chain_inflation_ratio",
    "tie_ball_interval",
    "tie_chains",
    "tie_cliques",
]

#: G1 defines rho but names no threshold. A chain twice the width of any
#: mutually-indistinguishable set means the reported group is held together by a
#: sequence of small steps rather than by indistinguishability.
DEFAULT_RHO_WARN: float = 2.0

#: A half-open ``[lo, hi)`` range of *ranks* (0-indexed positions in the sorted
#: score array), which every group in this module is, because all three objects
#: are contiguous there.
Interval = tuple[int, int]


# ---------------------------------------------------------------------------
# Balls (section 2.3.3, G9)
# ---------------------------------------------------------------------------
def tie_ball_interval(sorted_scores: Sequence[float], j: int, tau: float) -> Interval:
    """``[lo, hi)`` in rank space: ``{i : |S[i] - S[j]| <= tau}``.

    Args:
        sorted_scores: Non-increasing scores.
        j: A **0-indexed** position in that array.
        tau: The tolerance. ``tau = 0`` is legal and recovers exact ties.

    Returns:
        The half-open rank range. Contiguous, because the array is sorted.

    Note:
        The obvious implementation binary-searches for ``S[j] + tau`` and
        ``S[j] - tau``. **That is wrong**, and wrong exactly where it matters.
        Those bounds are themselves rounded, so the predicate actually evaluated
        becomes ``S[i] <= fl(S[j] + tau)``, which is a *different* test from
        ``spec_addenda.md#g9``'s pinned ``|s_i - s_{r_j}| <= tau``. The two
        disagree only at the boundary -- which is the sole place tie groups are
        interesting.

        So the search runs on the difference itself. On a non-increasing array,
        ``S[i] - S[j]`` is non-increasing in ``i``, and ``S[j] - S[i]`` is
        non-decreasing in ``i``; both are therefore still binary-searchable, and
        both evaluate exactly the subtraction G9 specifies. The monotonicity
        holds in binary64 and not merely in the reals, because IEEE subtraction
        is monotone: ``a >= b`` implies ``fl(a - c) >= fl(b - c)``.
    """
    n = len(sorted_scores)
    if not 0 <= j < n:
        raise IndexError(f"rank index {j} out of range 0..{n - 1}")
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")

    centre = sorted_scores[j]

    # lo: first i in [0, j] with S[i] - centre <= tau.
    lo, hi_search = 0, j
    while lo < hi_search:
        mid = (lo + hi_search) // 2
        if sorted_scores[mid] - centre <= tau:
            hi_search = mid
        else:
            lo = mid + 1

    # hi: first i in [j, n) with centre - S[i] > tau.
    lo_search, hi = j, n
    while lo_search < hi:
        mid = (lo_search + hi) // 2
        if centre - sorted_scores[mid] > tau:
            hi = mid
        else:
            lo_search = mid + 1

    return lo, hi


# ---------------------------------------------------------------------------
# Chains -- single linkage, a genuine partition
# ---------------------------------------------------------------------------
def tie_chains(sorted_scores: Sequence[float], tau: float) -> tuple[Interval, ...]:
    """The transitive closure of the near-tie relation: a partition.

    Cut wherever an adjacent gap exceeds ``tau``.

    Why that is exactly the closure: on a linearly ordered set, any sequence of
    "within tau" steps between two points can be replaced by the monotone path
    through the points between them, because gaps only shrink along the way. So
    two documents are connected precisely when every adjacent gap separating
    them is ``<= tau``.
    """
    n = len(sorted_scores)
    if n == 0:
        return ()
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")

    out: list[Interval] = []
    start = 0
    for i, gap in enumerate(adjacent_gaps(sorted_scores), start=1):
        if gap > tau:
            out.append((start, i))
            start = i
    out.append((start, n))
    return tuple(out)


# ---------------------------------------------------------------------------
# Cliques -- complete linkage, overlapping
# ---------------------------------------------------------------------------
def tie_cliques(sorted_scores: Sequence[float], tau: float) -> tuple[Interval, ...]:
    """Maximal sets in which *every pair* is within ``tau``.

    These are the sets for which "mutually indistinguishable" is actually true,
    as opposed to chains, where only neighbours need be close.

    The O(N) sweep is not merely cheap, it is **complete**, and the reason is a
    small lemma. The graph ``|s_i - s_j| <= tau`` is an indifference graph, so
    every maximal clique is a contiguous interval of the sorted order: if
    ``i < m < j`` and ``i, j`` are both in a clique, then
    ``S[i] >= S[m] >= S[j]`` gives ``|S[i] - S[m]| <= |S[i] - S[j]| <= tau`` and
    likewise for ``m, j``, so ``m`` belongs too. An interval graph on ``N``
    vertices has at most ``N`` maximal cliques, so enumerating one per left
    endpoint misses none.

    Let ``R(a)`` be the largest ``b`` with ``S[a] - S[b] <= tau``. ``R`` is
    non-decreasing, so a two-pointer pass computes all of it; and ``[a, R(a)]``
    is maximal exactly when ``a == 0`` or ``R(a) > R(a - 1)``, since otherwise
    the previous interval strictly contains it.
    """
    n = len(sorted_scores)
    if n == 0:
        return ()
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")

    out: list[Interval] = []
    right = 0
    previous_right = -1
    for a in range(n):
        right = max(right, a)
        while right + 1 < n and sorted_scores[a] - sorted_scores[right + 1] <= tau:
            right += 1
        if right > previous_right:
            out.append((a, right + 1))
            previous_right = right
    return tuple(out)


def chain_inflation_ratio(sorted_scores: Sequence[float], tau: float) -> float:
    """``rho(tau) = |largest chain| / |largest clique|``.

    Always ``>= 1``: a clique is an interval whose adjacent gaps are all
    ``<= tau``, so it lies inside some chain. Large values mean the reported
    tie groups are held together by chaining rather than by indistinguishability.
    """
    chains = tie_chains(sorted_scores, tau)
    cliques = tie_cliques(sorted_scores, tau)
    if not chains or not cliques:
        return math.nan
    return max(hi - lo for lo, hi in chains) / max(hi - lo for lo, hi in cliques)


# ---------------------------------------------------------------------------
# The index -- computes all three once, and owns the diagnostics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TieGroupIndex:
    """All three tie-group objects for one ``(sorted_scores, tau)`` pair.

    Diagnostics are emitted **from the constructor, exactly once**. That is not
    tidiness: ``pyproject.toml`` sets ``filterwarnings = ["error"]``, so a
    warning raised per :func:`tie_ball_interval` call would abort a tau sweep on
    its first query.
    """

    sorted_scores: tuple[float, ...]
    tau: float
    chains: tuple[Interval, ...]
    cliques: tuple[Interval, ...]

    @classmethod
    def build(
        cls,
        sorted_scores: Sequence[float],
        tau: float,
        *,
        rho_warn_threshold: float = DEFAULT_RHO_WARN,
    ) -> TieGroupIndex:
        """Compute chains and cliques, and emit the two diagnostics."""
        scores = tuple(sorted_scores)
        index = cls(
            sorted_scores=scores,
            tau=tau,
            chains=tie_chains(scores, tau),
            cliques=tie_cliques(scores, tau),
        )

        if scores:
            span = scores[0] - scores[-1]
            # ">=", not ">": at tau exactly equal to the range every ball is
            # already the whole corpus, so the degeneracy has begun. The
            # exception is span == 0 with tau == 0, which is the legitimate
            # exact-tie baseline rather than a degenerate configuration.
            if tau >= span and not (span == 0.0 and tau == 0.0):
                ratio = math.inf if span == 0.0 else tau / span
                warnings.warn(
                    f"tau={tau!r} covers the entire score range (span={span!r}, "
                    f"tau/span={ratio!r}): every tie ball is the whole corpus. "
                    f"This is a legitimate point at the top of a sweep, but results "
                    f"here are degenerate and should be marked as such.",
                    TauExceedsScoreRangeWarning,
                    stacklevel=2,
                )

        rho = index.rho
        if math.isfinite(rho) and rho > rho_warn_threshold:
            warnings.warn(
                f"chain inflation rho={rho:.3g} exceeds {rho_warn_threshold}: the "
                f"largest tie chain ({index.largest_chain} documents) is far wider "
                f"than the largest mutually-indistinguishable set "
                f"({index.largest_clique}). Tie-group statistics at tau={tau!r} are "
                f"dominated by transitive chaining.",
                ChainInflationWarning,
                stacklevel=2,
            )
        return index

    # -- the paper's object ---------------------------------------------------
    def ball(self, j: int) -> Interval:
        """``G_tau(j)`` as a half-open rank range, for 0-indexed ``j``."""
        return tie_ball_interval(self.sorted_scores, j, self.tau)

    def ball_members(self, j: int) -> tuple[int, ...]:
        """``G_tau(j)`` as explicit rank positions."""
        lo, hi = self.ball(j)
        return tuple(range(lo, hi))

    # -- diagnostics ----------------------------------------------------------
    @property
    def largest_chain(self) -> int:
        return max((hi - lo for lo, hi in self.chains), default=0)

    @property
    def largest_clique(self) -> int:
        return max((hi - lo for lo, hi in self.cliques), default=0)

    @property
    def rho(self) -> float:
        """The chain-inflation ratio; ``NaN`` on an empty corpus."""
        if not self.chains or not self.cliques:
            return math.nan
        return self.largest_chain / self.largest_clique

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    def chain_of(self, j: int) -> Interval:
        """The unique chain containing rank ``j`` -- chains do partition."""
        for lo, hi in self.chains:
            if lo <= j < hi:
                return lo, hi
        raise IndexError(f"rank {j} is outside the corpus")

    def report(self) -> dict[str, object]:
        """Both the paper-faithful and the partition statistics, side by side.

        G1 requires reporting them together: the ball is what section 2.3.3
        defines, the chain is the object with a well-defined notion of "the
        group containing document i", and rho says how far apart they are.
        """
        return {
            "tau": self.tau,
            "n_documents": len(self.sorted_scores),
            "n_chains": self.n_chains,
            "largest_chain": self.largest_chain,
            "n_cliques": len(self.cliques),
            "largest_clique": self.largest_clique,
            "rho": self.rho,
        }
