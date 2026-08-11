"""Stratification of ablation results by margin (README section 7.3).

Section 7.3: "Results are stratified by the boundary margin ``m_k`` relative to
``tau``, since tie-break effects are expected to concentrate in the near-tie
regime ``m_k <= tau``."

That expectation is the substance of research question A2, so the stratification
is what tests it rather than merely presenting it. If disagreement were spread
uniformly across margin bands, the claim that tie-breaking is a *near-tie*
phenomenon would be false.

Stratifying by ``m_k`` is legitimate precisely because the margin is
**tie-break independent** -- it depends only on the sorted score multiset, so it
is identical under all three operators. Stratifying an operator comparison by a
quantity that itself depended on the operator would be circular.

One empirical warning, worth carrying into the interpretation. On short-text
corpora the ``m_k == 0`` band is not a small tail: documents with no
in-vocabulary tokens all score exactly zero, so they form one large exact-tie
block. The near-tie regime is, in practice, substantially an *exact*-tie regime,
and :data:`EXACT_TIE_BAND` is therefore kept separate from the merely-small band
rather than absorbed into it.
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

#: Margins of exactly zero. Kept as its own band rather than folded into the
#: smallest numeric band: ``m_k == 0`` means membership is decided *entirely* by
#: the tie-break, which is a categorically different situation from a small but
#: positive margin, and on real corpora it is where most of the mass sits.
EXACT_TIE_BAND: Final[str] = "exact_tie"

#: Queries for which ``m_k`` does not exist (``k >= N``). Counted, never dropped.
UNDEFINED_BAND: Final[str] = "undefined"


def margin_bands(tau: float) -> tuple[tuple[str, float, float], ...]:
    """Bands for ``m_k`` relative to ``tau``, as ``(label, lo, hi)`` half-open.

    Centred on ``tau`` because that is the transition section 7.3 predicts, with
    decade-wide bands either side so the shape of the transition is visible
    rather than just its existence.
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

        ``NaN`` rather than ``0.0``: an empty band means "no evidence", which is
        not the same claim as "no disagreement", and plotting the two the same
        way would invent a data point.
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
            required -- there is deliberately no default anywhere in this
            package, because section 7.1 makes every tie-break result
            conditional on the choice of ``tau``.
        baseline, variant: Which operator comparison to stratify.
        ks: The k values to report.

    Returns:
        One :class:`Stratum` per (band, k), including empty bands so that a
        plot's x-axis is complete rather than silently ragged.
    """
    if tau < 0.0:
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
