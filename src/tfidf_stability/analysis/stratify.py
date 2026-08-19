"""Stratification of ablation results by margin (README section 7.3).

Section 7.3: "Results are stratified by the boundary margin ``m_k`` relative to
``tau``, since tie-break effects are expected to concentrate in the near-tie
regime ``m_k <= tau``." That expectation is research question A2, so the
stratification tests it: disagreement spread uniformly across bands would falsify
the claim that tie-breaking is a near-tie phenomenon.

Stratifying by ``m_k`` is legitimate because the margin is tie-break independent,
depending only on the sorted score multiset and so identical under all three
operators. Stratifying an operator comparison by an operator-dependent quantity
would be circular.

Empirical warning for the interpretation: on short-text corpora the ``m_k == 0``
band is no small tail. Documents with no in-vocabulary tokens all score zero and
form one large exact-tie block, so the near-tie regime is in practice
substantially an exact-tie regime. Hence :data:`EXACT_TIE_BAND` stays separate
from the merely-small band.
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
    return bands[-1][0]


def stratify_by_margin(
    results: Sequence[AblationResult],
    tau: float,
    *,
    baseline: str = "pi",
    variant: str = "pi_score",
    ks: Sequence[int] = (5, 10, 20, 50),
) -> list[Stratum]:
    """Group one operator comparison by margin band, per ``k``.

    Args:
        results: Ablation results, one per query.
        tau: The near-tie tolerance the bands are centred on. Explicit and
            required: nothing in this package supplies a default, because
            section 7.1 makes every tie-break result conditional on ``tau``.
        baseline, variant: Which operator comparison to stratify.
        ks: The k values to report.

    Returns:
        One :class:`Stratum` per (band, k), empty bands included so a plot's
        x-axis is complete rather than silently ragged.
    """
    # `not (tau >= 0.0)` rather than `tau < 0.0`: every comparison with NaN is
    # false, so the second form lets NaN through a guard whose own message says
    # non-negative. tie_groups.py:104,151,195 already spell it this way for
    # exactly this reason, and a NaN here would give every band NaN bounds and
    # drop every pair into the final band via _band_of's fallthrough.
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
    for result in results:
        for pair in result.pairs:
            if pair.baseline != baseline or pair.variant != variant or pair.k not in ks:
                continue
            label = _band_of(pair.margin.value, pair.margin.defined, bands)
            buckets[(label, pair.k)].append(pair)

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
                    mean_fks=sum(finite_fks) / len(finite_fks) if finite_fks else math.nan,
                    mean_jaccard=sum(finite_jac) / len(finite_jac) if finite_jac else math.nan,
                )
            )
    return strata
