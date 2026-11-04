"""Stratification of ablation results by margin (README section 7.3).

Tests research question A2, that tie-break effects concentrate where ``m_k <= tau``.
The margin is tie-break independent, and :data:`EXACT_TIE_BAND` is a band of its own.
Rationale: docs/engineering_notes.md#margin-stratification
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tfidf_stability.analysis.tie_break_ablations import AblationResult, OperatorPair

__all__ = [
    "EXACT_TIE_BAND",
    "UNDEFINED_BAND",
    "Stratum",
    "margin_bands",
    "stratify_by_margin",
]

#: Margins of zero. Its own band rather than the smallest numeric one: at
#: ``m_k == 0`` the tie-break decides membership outright, a categorically
#: different situation from a small positive margin, and on real corpora it is
#: where most of the mass sits.
EXACT_TIE_BAND: Final[str] = "exact_tie"

#: Queries for which ``m_k`` does not exist (``k >= N``). Kept and counted.
UNDEFINED_BAND: Final[str] = "undefined"


def margin_bands(tau: float) -> tuple[tuple[str, float, float], ...]:
    """Bands for ``m_k`` relative to ``tau``, as ``(label, lo, hi)`` half-open.

    Centred on ``tau``, the transition section 7.3 predicts, with decade-wide
    bands either side so the shape of the transition shows up rather than only
    its existence.
    """
    return (
        ("(0, tau/100]", 0.0, tau / 100.0),
        ("(tau/100, tau/10]", tau / 100.0, tau / 10.0),
        ("(tau/10, tau]", tau / 10.0, tau),
        ("(tau, 10*tau]", tau, tau * 10.0),
        ("(10*tau, 100*tau]", tau * 10.0, tau * 100.0),
        ("(100*tau, inf)", tau * 100.0, math.inf),
    )


@dataclass(frozen=True, slots=True)
class Stratum:
    """Disagreement statistics for one margin band at one ``k``."""

    label: str
    k: int
    lo: float
    hi: float
    n: int
    n_disagree: int
    mean_fks: float
    mean_jaccard: float

    @property
    def disagreement_rate(self) -> float:
        """Fraction whose top-k set differs; ``NaN`` on an empty band.

        ``NaN`` rather than ``0.0``: an empty band means "no evidence", a
        different claim from "no disagreement", and plotting the two alike would
        invent a data point.
        """
        return self.n_disagree / self.n if self.n else math.nan


def _band_of(value: float, defined: bool, bands: Sequence[tuple[str, float, float]]) -> str:
    if not defined or math.isnan(value):
        return UNDEFINED_BAND
    if value == 0.0:
        return EXACT_TIE_BAND
    for label, lo, hi in bands:
        if lo < value <= hi:
            return label
    # The bands tile (0, inf) and zero is handled above, so only a negative
    # value reaches here, which non-increasing scores cannot produce. It joins
    # `undefined` rather than the top band: a negative margin is not evidence of
    # separation. Every value keeps a band, so the per-k totals reconcile.
    return UNDEFINED_BAND


def stratify_by_margin(
    results: Sequence[AblationResult],
    tau: float,
    *,
    baseline: str = "pi",
    variant: str = "pi_score",
    ks: Sequence[int] = (5, 10, 20, 50),
) -> list[Stratum]:
    """Group one operator comparison by margin band, per ``k``.

    ``results`` holds one ablation result per query; ``baseline`` and ``variant`` name
    the comparison. ``tau`` has no default, as section 7.1 makes every tie-break result
    conditional on it. Returns one :class:`Stratum` per (band, k), empty bands included.
    """
    # `not (tau >= 0.0)` rather than `tau < 0.0`: every comparison with NaN is false,
    # so the second form admits NaN. tie_ball_interval, tie_chains and tie_cliques use the
    # same form. A NaN tau gives every band NaN bounds and sends every positive margin
    # down _band_of's fallthrough into `undefined`.
    if not (tau >= 0.0):
        raise ValueError(f"tau must be non-negative, got {tau}")

    bands = margin_bands(tau)
    labels = [EXACT_TIE_BAND, *(b[0] for b in bands), UNDEFINED_BAND]
    bounds = {
        EXACT_TIE_BAND: (0.0, 0.0),
        UNDEFINED_BAND: (math.nan, math.nan),
        **{label: (lo, hi) for label, lo, hi in bands},
    }

    buckets: dict[tuple[str, int], list[OperatorPair]] = {
        (label, k): [] for label in labels for k in ks
    }
    # `pair.margin.k`, not `pair.k`: `pair.k` forwards the comparison's clamped
    # `k_eff = min(k, N)`, while the margin and its band use the requested `k`. A k above
    # N is a supported grid point; bucketing by `pair.k` would merge or drop such pairs.
    # Rationale: docs/engineering_notes.md#stratify-bucket-k
    for result in results:
        for pair in result.pairs:
            if pair.baseline != baseline or pair.variant != variant or pair.margin.k not in ks:
                continue
            label = _band_of(pair.margin.value, pair.margin.defined, bands)
            buckets[(label, pair.margin.k)].append(pair)

    strata: list[Stratum] = []
    for k in ks:
        for label in labels:
            group = buckets[(label, k)]
            finite_fks = [p.comparison.fks for p in group]
            finite_jac = [p.comparison.jaccard for p in group]
            lo, hi = bounds[label]
            strata.append(
                Stratum(
                    label=label,
                    k=k,
                    lo=lo,
                    hi=hi,
                    n=len(group),
                    n_disagree=sum(p.sets_differ for p in group),
                    # `math.fsum` matches `summarise_values`: these are
                    # published columns, so the reduction must not vary with the
                    # interpreter. CPython 3.12 and later compensate `sum` over
                    # floats; on 3.11 and earlier it is a naive fold.
                    mean_fks=math.fsum(finite_fks) / len(finite_fks) if finite_fks else math.nan,
                    mean_jaccard=(
                        math.fsum(finite_jac) / len(finite_jac) if finite_jac else math.nan
                    ),
                )
            )
    return strata
