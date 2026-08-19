"""Score-separation margins and flip radii (README sections 2.3.2 and 4.4).

    m_k         = score(r_k) - score(r_{k+1})
    m_min^top   = min over 1 <= j < k of (score(r_j) - score(r_{j+1}))
    eps_k^flip  = m_k / 2

Every function here takes a sorted score array rather than a ``Ranking``, which
makes the following structural:

    m_k depends only on the score multiset. The non-increasing rearrangement of
    a multiset is unique, and all three ranking operators use score-descending as
    their primary key, so pi, pi_score and pi_alt have identical score sequences
    with differing document sequences.

Research questions A1 (margins) and A2 (tie-breaking) are therefore
independent. The paper never states it; proposed as a note under section 2.3.2.

Undefined margins follow ``spec_addenda.md#g3``: ``NaN`` plus an explicit
validity flag, never coerced to 0 or infinity, and counted rather than silently
dropped. Whether a degenerate query then enters a reported distribution is a
policy question for ``analysis/stability_profile.py``; this module reports the
truth so the raw margins stay auditable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from tfidf_stability.utils.validation import StrictMode, resolve_k

__all__ = [
    "Margin",
    "MarginSummary",
    "adjacent_gaps",
    "boundary_margin",
    "margin_profile",
    "min_adjacent_margin_top",
    "summarise",
]


@dataclass(frozen=True, slots=True)
class Margin:
    """One margin, with the validity flag G3 requires.

    Attributes:
        kind: ``"boundary"`` or ``"min_adjacent_top"``.
        k: The requested ``k``.
        k_effective: After lenient clamping; equal to ``k`` in strict mode.
        value: The margin, or ``NaN`` when undefined.
        defined: Whether ``value`` is meaningful. An undefined margin is excluded
            from distributions but counted in ``n_undefined``; coercing it to 0
            would look like an exact tie, and to infinity like perfect stability.
        reason: Why it is undefined; empty when defined.
    """

    kind: str
    k: int
    k_effective: int
    value: float
    defined: bool
    reason: str = ""

    @property
    def flip_radius(self) -> float:
        """``eps_k^flip = m_k / 2`` (section 2.3.2).

        Exact: division by a power of two only shifts the exponent, so
        ``2 * flip_radius`` recovers ``value`` bit for bit.
        """
        return self.value / 2.0

    @property
    def is_exact_tie(self) -> bool:
        """A defined margin of exactly zero.

        The interesting case: top-k membership is then decided purely by the
        tie-break, the subject of section 4.5, and ``P(m_k = 0)`` is a headline
        statistic.
        """
        return self.defined and self.value == 0.0


def adjacent_gaps(sorted_scores: Sequence[float]) -> tuple[float, ...]:
    """Gaps between consecutive ranks: ``S[j] - S[j+1]`` for each ``j``.

    Length ``N - 1``. Shared with :mod:`~tfidf_stability.ranking.tie_groups`,
    where single-linkage chains are the runs of gaps ``<= tau``.
    """
    return tuple(a - b for a, b in pairwise(sorted_scores))


def boundary_margin(
    sorted_scores: Sequence[float],
    k: int,
    *,
    mode: StrictMode = StrictMode.STRICT,
) -> Margin:
    """``m_k = score(r_k) - score(r_{k+1})``, governing top-k membership.

    Args:
        sorted_scores: Scores in non-increasing order.
        k: The 1-indexed boundary rank.
        mode: Strict raises when ``k > N``; lenient clamps and records it.

    Returns:
        The :class:`Margin`. Undefined when ``k >= N``, since ``r_{k+1}`` does
        not exist.

    Raises:
        KOutOfRangeError: In strict mode when ``k > N`` (or ``k <= 0`` in either).
    """
    n = len(sorted_scores)
    k_eff = resolve_k(k, n, mode)

    if k_eff >= n:
        reason = "k == N: r_{k+1} does not exist" if k == n else "k clamped to N"
        return Margin("boundary", k, k_eff, math.nan, False, reason)

    return Margin("boundary", k, k_eff, sorted_scores[k_eff - 1] - sorted_scores[k_eff], True)


def min_adjacent_margin_top(
    sorted_scores: Sequence[float],
    k: int,
    *,
    mode: StrictMode = StrictMode.STRICT,
) -> Margin:
    """``m_min^top``, governing the ordering within the top-k.

    Undefined at ``k = 1``: the minimum is over an empty set. G3 does not cover
    this case, so ``NaN`` is adopted (proposed as addendum G16). ``+inf`` would
    claim "no constraint" and pollute any percentile summary it entered.
    """
    n = len(sorted_scores)
    k_eff = resolve_k(k, n, mode)

    if k_eff < 2:
        return Margin("min_adjacent_top", k, k_eff, math.nan, False, "k == 1: vacuous minimum")
    # Unreachable while `resolve_k` either returns k (having checked k <= n) or
    # clamps to n, both of which put n at or above k_eff: the guard above has
    # already returned for every corpus of fewer than two documents. Kept
    # because the loop below indexes a second element on the strength of it.
    if n < 2:  # pragma: no cover - defensive
        return Margin("min_adjacent_top", k, k_eff, math.nan, False, "N == 1: no adjacent pair")

    gaps = [sorted_scores[j] - sorted_scores[j + 1] for j in range(k_eff - 1)]
    return Margin("min_adjacent_top", k, k_eff, min(gaps), True)


def margin_profile(
    sorted_scores: Sequence[float],
    ks: Sequence[int] = (5, 10, 20, 50),
    *,
    mode: StrictMode = StrictMode.LENIENT,
) -> tuple[Margin, ...]:
    """Boundary margins at each ``k``: the k-set of section 7.1.

    Lenient by default, since a sweep over ``k in {5, 10, 20, 50}`` on a corpus
    smaller than 50 is a legitimate grid point rather than a misconfiguration.
    """
    return tuple(boundary_margin(sorted_scores, k, mode=mode) for k in ks)


@dataclass(frozen=True, slots=True)
class MarginSummary:
    """Aggregate statistics over many margins."""

    n: int
    n_defined: int
    n_undefined: int
    n_exact_tie: int
    percentiles: tuple[tuple[float, float], ...]

    @property
    def p_exact_tie(self) -> float:
        """Fraction of defined margins that are exactly zero.

        G3 calls this a headline statistic. On short-text corpora it is large:
        documents with no in-vocabulary tokens all score exactly 0 and form one
        enormous exact-tie block.
        """
        return self.n_exact_tie / self.n_defined if self.n_defined else math.nan


def summarise(
    margins: Sequence[Margin],
    quantiles: Sequence[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
) -> MarginSummary:
    """Percentile summary over defined margins, counting the undefined ones.

    Nearest-rank rather than interpolating: margin distributions have an atom at
    exactly zero, and interpolating across it would invent values no query
    produced.
    """
    defined = sorted(m.value for m in margins if m.defined)
    pct: list[tuple[float, float]] = []
    if defined:
        for q in quantiles:
            idx = min(len(defined) - 1, max(0, math.ceil(q * len(defined)) - 1))
            pct.append((q, defined[idx]))

    return MarginSummary(
        n=len(margins),
        n_defined=len(defined),
        n_undefined=sum(1 for m in margins if not m.defined),
        n_exact_tie=sum(1 for m in margins if m.is_exact_tie),
        percentiles=tuple(pct),
    )
