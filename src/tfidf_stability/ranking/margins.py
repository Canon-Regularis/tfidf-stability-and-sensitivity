"""Score-separation margins and flip radii (README sections 2.3.2 and 4.4).

``m_k = score(r_k) - score(r_{k+1})``, ``eps_k^flip = m_k / 2``, and ``m_min^top`` is the
minimum of ``score(r_j) - score(r_{j+1})`` over ``1 <= j < k``. Undefined margins are ``NaN``
plus a validity flag (``spec_addenda.md#g3``). Rationale: docs/engineering_notes.md#margins
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import islice, pairwise

from tfidf_stability.utils.validation import StrictMode, resolve_k

__all__ = [
    "Margin",
    "MarginKind",
    "MarginSummary",
    "adjacent_gaps",
    "boundary_margin",
    "margin_profile",
    "min_adjacent_margin_top",
    "summarise",
]


class MarginKind(str, Enum):
    """Which margin a :class:`Margin` holds (README section 2.3.2)."""

    BOUNDARY = "boundary"
    MIN_ADJACENT_TOP = "min_adjacent_top"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Margin:
    """One margin, with the validity flag G3 requires.

    ``kind`` is a :class:`MarginKind`; ``k_effective`` is ``k`` after lenient clamping. When
    ``defined`` is false, ``value`` is ``NaN`` and ``reason`` (else empty) says why;
    ``n_undefined`` counts it outside percentiles. Reporting: ``analysis/stability_profile.py``.
    """

    #: Coerced to a :class:`MarginKind`, and annotated with its base, ``str``, so a
    #: caller may pass the bare value.
    kind: str
    k: int
    k_effective: int
    value: float
    defined: bool
    reason: str = ""

    def __post_init__(self) -> None:
        """Normalise ``kind`` to its member, refusing an unknown value."""
        object.__setattr__(self, "kind", MarginKind(self.kind))

    @property
    def flip_radius(self) -> float:
        """``eps_k^flip = m_k / 2`` (section 2.3.2).

        Halving is exact for every margin of at least ``2**-1021``, so ``2 * flip_radius ==
        value`` there. Below it, the half of an odd multiple ``m`` of ``5e-324`` rounds to the
        nearest even multiple: down if ``m % 4 == 1``, up if ``m % 4 == 3``, zero for ``m == 1``.
        """
        return self.value / 2.0

    @property
    def is_exact_tie(self) -> bool:
        """A defined margin of exactly zero.

        Top-k membership is then decided by the tie-break alone (section 4.5), and
        ``P(m_k = 0)`` is a headline statistic.
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

    ``sorted_scores`` is non-increasing; ``k`` is the 1-indexed boundary rank. Returns a
    :class:`Margin`, undefined when ``k >= N`` since ``r_{k+1}`` does not exist. Raises
    ``KOutOfRangeError`` for ``k <= 0``, and for ``k > N`` in strict ``mode``; lenient clamps to N.
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
    # because `min` below needs a gap, and fewer than two scores give none.
    if n < 2:  # pragma: no cover - defensive
        return Margin("min_adjacent_top", k, k_eff, math.nan, False, "N == 1: no adjacent pair")

    gaps = adjacent_gaps(tuple(islice(sorted_scores, k_eff)))
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
        large exact-tie block.
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
