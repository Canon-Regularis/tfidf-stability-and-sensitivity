"""Tie groups and near-ties (README section 2.3.3, ``spec_addenda.md#g1``).

The ball ``G_tau(j) = {i : |s_i - score(r_j)| <= tau}`` is not a partition, as near-tie is not
transitive. G1 reports the ball :func:`tie_ball_interval`, the partition :func:`tie_chains`
and the complete-linkage :func:`tie_cliques`. Rationale: docs/engineering_notes.md#tie-groups
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
    check_tau,
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

#: A half-open ``[lo, hi)`` range of ranks (0-indexed positions in the sorted
#: score array). Every group in this module is one, since all three objects are
#: contiguous there.
Interval = tuple[int, int]


# ---------------------------------------------------------------------------
# Balls (section 2.3.3, G9)
# ---------------------------------------------------------------------------
def tie_ball_interval(sorted_scores: Sequence[float], j: int, tau: float) -> Interval:
    """``[lo, hi)`` in rank space: ``{i : |S[i] - S[j]| <= tau}``.

    ``j`` is a 0-indexed position in the non-increasing ``sorted_scores``; ``tau = 0`` gives
    exact ties. Both searches evaluate the ``spec_addenda.md#g9`` difference itself, never a
    rounded bound ``fl(S[j] + tau)``. Rationale: docs/engineering_notes.md#tie-ball-search
    """
    n = len(sorted_scores)
    if not 0 <= j < n:
        raise IndexError(f"rank index {j} out of range 0..{n - 1}")
    check_tau(tau)

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
# Chains: single linkage, a genuine partition
# ---------------------------------------------------------------------------
def tie_chains(sorted_scores: Sequence[float], tau: float) -> tuple[Interval, ...]:
    """The transitive closure of the near-tie relation: a partition.

    Cut wherever an adjacent gap exceeds ``tau``. On a linearly ordered set any sequence of
    "within tau" steps reduces to the monotone path between its ends, so two documents are
    connected when every adjacent gap separating them is ``<= tau``.
    """
    n = len(sorted_scores)
    if n == 0:
        return ()
    check_tau(tau)

    out: list[Interval] = []
    start = 0
    for i, gap in enumerate(adjacent_gaps(sorted_scores), start=1):
        if gap > tau:
            out.append((start, i))
            start = i
    out.append((start, n))
    return tuple(out)


# ---------------------------------------------------------------------------
# Cliques: complete linkage, overlapping
# ---------------------------------------------------------------------------
def tie_cliques(sorted_scores: Sequence[float], tau: float) -> tuple[Interval, ...]:
    """Maximal sets in which every pair is within ``tau``.

    The near-tie graph is an indifference graph, so maximal cliques are sorted-order intervals.
    With ``R(a)`` the largest ``b`` where ``S[a] - S[b] <= tau``, ``[a, R(a)]`` is maximal iff
    ``a == 0`` or ``R(a) > R(a - 1)``. Rationale: docs/engineering_notes.md#tie-clique-sweep
    """
    n = len(sorted_scores)
    if n == 0:
        return ()
    check_tau(tau)

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

    ``rho >= 1`` on non-increasing input; the order is unchecked and unsorted input can give
    ``rho < 1``. Callers sort via :func:`~tfidf_stability.ranking.ranker.sorted_scores_desc`.
    Large rho flags chaining. Rationale: docs/engineering_notes.md#rho-order
    """
    chains = tie_chains(sorted_scores, tau)
    cliques = tie_cliques(sorted_scores, tau)
    if not chains or not cliques:
        return math.nan
    return max(hi - lo for lo, hi in chains) / max(hi - lo for lo, hi in cliques)


# ---------------------------------------------------------------------------
# The index: computes all three once, and owns the diagnostics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TieGroupIndex:
    """All three tie-group objects for one ``(sorted_scores, tau)`` pair.

    Diagnostics are emitted from the constructor, once: ``pyproject.toml`` sets
    ``filterwarnings = ["error"]``, so a warning raised per
    :func:`tie_ball_interval` call would abort a tau sweep on its first query.
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
            # ">=" rather than ">": at tau equal to the range, every ball is
            # already the whole corpus, so the degeneracy has begun. The
            # exception is span == 0 with tau == 0, the legitimate exact-tie
            # baseline.
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

    # --- the paper's object --------------------------------------------------
    def ball(self, j: int) -> Interval:
        """``G_tau(j)`` as a half-open rank range, for 0-indexed ``j``."""
        return tie_ball_interval(self.sorted_scores, j, self.tau)

    def ball_members(self, j: int) -> tuple[int, ...]:
        """``G_tau(j)`` as explicit rank positions."""
        lo, hi = self.ball(j)
        return tuple(range(lo, hi))

    # --- diagnostics ---------------------------------------------------------
    @property
    def largest_chain(self) -> int:
        return max((hi - lo for lo, hi in self.chains), default=0)

    @property
    def largest_clique(self) -> int:
        return max((hi - lo for lo, hi in self.cliques), default=0)

    @property
    def largest_ball(self) -> int:
        """The widest ball over every centre.

        G1 makes the ball the primary object; it lies between the clique and the chain:
        ``largest_clique <= largest_ball <= largest_chain``. The bounds are loose: on the
        adversarial ladder the chain spans the whole corpus and the widest ball spans three.
        """
        return max(
            (hi - lo for lo, hi in (self.ball(j) for j in range(len(self.sorted_scores)))),
            default=0,
        )

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
        """The unique chain containing rank ``j``; chains do partition."""
        for lo, hi in self.chains:
            if lo <= j < hi:
                return lo, hi
        raise IndexError(f"rank {j} is outside the corpus")

    def report(self) -> dict[str, object]:
        """Both the paper-faithful and the partition statistics, side by side.

        G1 requires reporting them together: the ball is what section 2.3.3
        defines, the chain is the object with a well-defined "group containing
        document i", and rho says how far apart the two are.
        """
        return {
            "tau": self.tau,
            "n_documents": len(self.sorted_scores),
            "n_chains": self.n_chains,
            "largest_chain": self.largest_chain,
            "n_cliques": len(self.cliques),
            "largest_clique": self.largest_clique,
            "largest_ball": self.largest_ball,
            "rho": self.rho,
        }
