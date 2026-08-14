"""A1: how score-separation margins govern ranking stability (sections 7.2, 7.3).

Section 4.4 proves a sufficient condition: if every score moves by less than
``m_k / 2`` the top-k set is unchanged. This module measures what happens, which
differs from the bound in two directions at once.

**Worst case.** Stage 4's dyadic witness makes ``m_k / 2`` the flip radius: a
perturbation of size ``m_k / 2 + delta`` flips the pair, so the bound cannot be
improved.

**Average case.** Under random perturbation of size ``eps`` the flip rate stays
at zero well past ``eps = m_k / 2``, since the adversarial configuration (push
the rank-k score down and the rank-(k+1) score up, both by the full ``eps``) is a
measure-zero corner of the perturbation cube. Measured here: 0% at
``eps = m_k/2``, first non-zero around ``1.1 x``, roughly half around ``4 x``.

Neither implies the other. Reporting only the second understates the risk;
reporting only the first suggests rankings are far more fragile than they are.
:func:`transition_curve` measures the average case, :func:`certificate_audit` the
soundness and conservatism of the bound.

The certificate as a 2x2 table
------------------------------
``certified_stable`` is a proof when true and merely "not covered" when false, so
accuracy would reward a certificate that always said no. The table separates the
two error directions:

* **certified but changed** must be zero. A non-zero entry falsifies section 4.4
  and means a bug or a broken proof.
* **uncertified but unchanged** is the conservatism and is expected to be large.
  Reporting it stops "not certified" being read as "will break".

The A1/A2 boundary
------------------
Queries whose ``m_k`` is zero are excluded from the transition curve and reported
separately. At ``m_k = 0`` the boundary is an exact tie, no perturbation is
needed to change the outcome, and the tie-break decides: A2's regime, where
including it would let a tie-break effect read as a numerical-stability effect.
G3 requires the exclusion for margin distributions and the same logic applies
here.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tfidf_stability.perturbation.score_bounds import certified_radius
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.margins import boundary_margin
from tfidf_stability.ranking.ranker import rank_top_k

__all__ = [
    "CertificateAudit",
    "TransitionPoint",
    "certificate_audit",
    "transition_curve",
]

#: Multiples of the certified radius ``m_k / 2`` at which to sample.
DEFAULT_RATIOS: tuple[float, ...] = (0.25, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 5.0, 20.0)


@dataclass(frozen=True, slots=True)
class TransitionPoint:
    """Empirical flip rate at one multiple of the certified radius."""

    ratio: float
    n_flips: int
    n_trials: int

    @property
    def flip_rate(self) -> float:
        return self.n_flips / self.n_trials if self.n_trials else math.nan

    @property
    def within_certificate(self) -> bool:
        """Whether section 4.4 guarantees zero flips at this ratio.

        Strict: the theorem is ``eps < m_k / 2``, so ``ratio == 1.0`` sits on
        the boundary and is uncovered. Sampled anyway, since a flip at the
        boundary rather than beyond it is the interesting failure.
        """
        return self.ratio < 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "n_flips": self.n_flips,
            "n_trials": self.n_trials,
            "flip_rate": self.flip_rate,
            "within_certificate": self.within_certificate,
        }


@dataclass(frozen=True, slots=True)
class CertificateAudit:
    """The 2x2 table of section 4.4's certificate against observed outcomes."""

    certified_unchanged: int
    certified_changed: int
    uncertified_unchanged: int
    uncertified_changed: int
    n_undefined: int
    #: Queries excluded because ``m_k`` is zero (A2's regime). Counted rather
    #: than dropped, so the exclusion is visible in the published record.
    n_exact_tie: int = 0

    @property
    def is_sound(self) -> bool:
        """Whether the certificate ever failed. Must be ``True``.

        Read alongside :attr:`is_conclusive`: this says the certified cell holds
        no failures, and is vacuously true when it holds nothing.
        """
        return self.certified_changed == 0

    @property
    def n_certified(self) -> int:
        """Perturbations that actually fell inside the certified radius."""
        return self.certified_unchanged + self.certified_changed

    @property
    def is_conclusive(self) -> bool:
        """Whether the audit tested section 4.4 at all.

        ``is_sound`` is ``certified_changed == 0``, so an audit that drew no
        certified perturbation reports the theorem upheld having checked it zero
        times. An earlier version of the section 4.4 attack reported thousands
        of "certified perturbations" with none inside the radius, making its
        zero-violation result vacuous. Report soundness with this count beside
        it.
        """
        return self.n_certified > 0

    @property
    def conservatism(self) -> float:
        """Share of *uncertified* cases that were in fact unchanged."""
        total = self.uncertified_unchanged + self.uncertified_changed
        return self.uncertified_unchanged / total if total else math.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "certified_unchanged": self.certified_unchanged,
            "certified_changed": self.certified_changed,
            "uncertified_unchanged": self.uncertified_unchanged,
            "uncertified_changed": self.uncertified_changed,
            "n_undefined": self.n_undefined,
            "n_exact_tie": self.n_exact_tie,
            "n_certified": self.n_certified,
            "is_sound": self.is_sound,
            "is_conclusive": self.is_conclusive,
            "conservatism": self.conservatism,
        }


def _top_k_set(scores: Sequence[float], table: AttributeTable, k: int) -> frozenset[int]:
    return frozenset(rank_top_k(scores, table, k=k).order[:k])


def transition_curve(
    score_vectors: Sequence[Sequence[float]],
    table: AttributeTable,
    k: int,
    *,
    seed: int,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    trials: int = 40,
    tables: Sequence[AttributeTable] | None = None,
) -> tuple[tuple[TransitionPoint, ...], int, int]:
    """Measure the empirical top-k flip rate against ``eps / (m_k / 2)``.

    Each trial perturbs every score independently and uniformly in
    ``[-eps, +eps]``: the random case. The adversarial case is ``m_k / 2`` by the
    Stage 4 witness and needs no sampling.

    Args:
        score_vectors: One score vector per query, in document-index order.
        table: The attribute table supplying the tie-break, when every query
            ranks over the same documents.
        k: The rank to test.
        seed: Seeds a local :class:`random.Random`; the global generator is never
            touched, so this is reproducible regardless of what else has run.
        ratios: Multiples of ``m_k / 2`` to sample.
        trials: Perturbations per (query, ratio) cell.
        tables: Per-query attribute tables, overriding ``table``. Required under
            section 7.1's protocol, where each query excludes its own profile
            items and so ranks over a different candidate set (G19). A restricted
            score vector against the full-corpus table would let the tie-break
            consider documents the query could never retrieve.

    Returns:
        ``(points, n_used, n_excluded)``, where ``n_excluded`` counts queries
        dropped because ``m_k`` was undefined or zero: the A2 regime, which must
        not be averaged into an A1 curve.
    """
    rng = random.Random(seed)
    flips = dict.fromkeys(ratios, 0)
    counts = dict.fromkeys(ratios, 0)
    n_used = n_excluded = 0

    for index, scores in enumerate(score_vectors):
        active = tables[index] if tables is not None else table
        if k >= len(scores):
            # Candidate sets vary per query under section 7.1, so k can exceed
            # one. Counted rather than clamped: a clamped k measures a different
            # quantity.
            n_excluded += 1
            continue
        margin = boundary_margin(sorted(scores, reverse=True), k)
        if not margin.defined or margin.value == 0.0:
            n_excluded += 1
            continue
        n_used += 1
        base = _top_k_set(scores, active, k)
        radius = margin.flip_radius
        for ratio in ratios:
            eps = radius * ratio
            for _ in range(trials):
                perturbed = [s + rng.uniform(-eps, eps) for s in scores]
                counts[ratio] += 1
                if _top_k_set(perturbed, active, k) != base:
                    flips[ratio] += 1

    points = tuple(TransitionPoint(r, flips[r], counts[r]) for r in ratios)
    return points, n_used, n_excluded


def certificate_audit(
    score_vectors: Sequence[Sequence[float]],
    table: AttributeTable,
    k: int,
    *,
    seed: int,
    trials: int = 20,
    max_ratio: float = 8.0,
    tables: Sequence[AttributeTable] | None = None,
) -> CertificateAudit:
    """Audit section 4.4's certificate against observed top-k changes.

    Perturbations straddle the certified radius so both cells of each row are
    populated; drawing only tiny ones would make the certificate look trivially
    sound.

    Args:
        score_vectors: One score vector per query.
        table: The attribute table supplying the tie-break.
        k: The rank to certify.
        seed: Seeds a local generator.
        trials: Perturbations per query.
        max_ratio: Largest multiple of the certified radius to draw.
        tables: Per-query attribute tables; see :func:`transition_curve`.

    Returns:
        The :class:`CertificateAudit`. Check :attr:`~CertificateAudit.is_sound`
        first: a false value falsifies section 4.4 and nothing else in the
        result is meaningful.
    """
    rng = random.Random(seed)
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    n_undefined = 0
    n_exact_tie = 0

    for index, scores in enumerate(score_vectors):
        active = tables[index] if tables is not None else table
        if k >= len(scores):
            n_undefined += 1
            continue
        sorted_scores = sorted(scores, reverse=True)
        cert = certified_radius(sorted_scores, k)
        if not cert.defined or math.isnan(cert.set_radius):
            n_undefined += 1
            continue
        # Exact ties are A2's regime. The audit used to include them: at m_k = 0
        # the radius is 0.0, so eps is 0.0, `perturbed` equals `scores` element
        # for element, `realised < 0.0` is false and "unchanged" is trivially
        # true. Every such trial landed in (uncertified, unchanged) and inflated
        # the published conservatism with cases where nothing was perturbed.
        if cert.set_radius == 0.0:
            n_exact_tie += 1
            continue
        base = _top_k_set(scores, active, k)
        for _ in range(trials):
            eps = cert.set_radius * rng.uniform(0.0, max_ratio)
            perturbed = [s + rng.uniform(-eps, eps) for s in scores]
            # Measure the realised delta rather than the drawn eps: `fl(s + d)`
            # rounds, so movement can exceed |d| by half an ulp, and the theorem
            # is about the movement that happened.
            realised = max(
                (abs(p - s) for p, s in zip(perturbed, scores, strict=True)), default=0.0
            )
            certified = realised < cert.set_radius
            unchanged = _top_k_set(perturbed, active, k) == base
            cells[(certified, unchanged)] += 1

    return CertificateAudit(
        certified_unchanged=cells[(True, True)],
        certified_changed=cells[(True, False)],
        uncertified_unchanged=cells[(False, True)],
        uncertified_changed=cells[(False, False)],
        n_undefined=n_undefined,
        n_exact_tie=n_exact_tie,
    )
