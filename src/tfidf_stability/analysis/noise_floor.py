"""Derive the band ``[tau_floor, g_min)`` of admissible ``tau`` (section 7.1).

``tau_floor = 2 * eta``, ``eta`` the worst score error against :data:`Reduction.EXACT`; ``g_min``
the least positive adjacent gap. :func:`verify_band_invariance` checks the :func:`tau_band` result.
``tau`` is not a ``configs/default.yaml`` key. Rationale: docs/engineering_notes.md#tau-band
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tfidf_stability.ranking.margins import adjacent_gaps
from tfidf_stability.ranking.tie_groups import tie_chains
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.numerics import Reduction, bits_of, ulps_between
from tfidf_stability.vectorisation.sparse import SparseVector
from tfidf_stability.vectorisation.tfidf import TfidfModel

__all__ = [
    "NoiseFloor",
    "TauBand",
    "measure_noise_floor",
    "tau_band",
    "verify_band_invariance",
]

#: Policies compared against :data:`Reduction.EXACT`.
_INSTRUMENTS = (Reduction.NAIVE, Reduction.NEUMAIER, Reduction.PAIRWISE)


@dataclass(frozen=True, slots=True)
class PolicyError:
    """How far one reduction policy strays from correctly-rounded arithmetic."""

    policy: str
    n_compared: int
    n_differing: int
    max_abs: float
    #: Largest unsigned ulp distance. `ulps_between(policy, exact)` is negative where
    #: the policy value lies above the exact one; this field holds its magnitude.
    #: Reported only: `NoiseFloor.eta` and `tau_floor` use `max_abs`.
    max_ulps: float

    @property
    def share_differing(self) -> float:
        return self.n_differing / self.n_compared if self.n_compared else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "n_compared": self.n_compared,
            "n_differing": self.n_differing,
            "share_differing": self.share_differing,
            "max_abs": self.max_abs,
            # Hex so the bound survives the reader's decimal parser intact.
            "max_abs_hex": float.hex(self.max_abs),
            "max_ulps": self.max_ulps,
        }


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """The measured arithmetic floor, and the ``tau`` it implies."""

    per_policy: tuple[PolicyError, ...]
    n_queries: int
    n_documents: int

    @property
    def eta(self) -> float:
        """Worst per-score disagreement with correctly-rounded arithmetic."""
        return max((p.max_abs for p in self.per_policy), default=0.0)

    @property
    def tau_floor(self) -> float:
        """``2 * eta``: the smallest ``tau`` that cannot be arithmetic noise.

        A margin is a difference of two scores, each off by at most ``eta``.
        Rationale: docs/engineering_notes.md#tau-band
        """
        return 2.0 * self.eta

    def as_dict(self) -> dict[str, Any]:
        return {
            "eta": self.eta,
            "eta_hex": float.hex(self.eta),
            "tau_floor": self.tau_floor,
            "tau_floor_hex": float.hex(self.tau_floor),
            "n_queries": self.n_queries,
            "n_documents": self.n_documents,
            "per_policy": [p.as_dict() for p in self.per_policy],
        }


@dataclass(frozen=True, slots=True)
class TauBand:
    """The admissible interval for ``tau``, and whether it is non-empty."""

    tau_floor: float
    g_min: float
    n_gaps_in_band: int
    n_exact_ties: int
    n_positive_gaps: int

    @property
    def is_valid(self) -> bool:
        """Whether the paper's two-sided constraint has any solution at all."""
        return self.tau_floor < self.g_min

    @property
    def is_invariant(self) -> bool:
        """Whether every ``tau`` in the band gives identical tie structure.

        True when no observed gap falls inside the band; piecewise constancy in ``tau``
        then establishes it by argument rather than by sampling.
        Rationale: docs/engineering_notes.md#tau-band
        """
        return self.is_valid and self.n_gaps_in_band == 0

    @property
    def decades(self) -> float:
        """Width of the band in orders of magnitude."""
        if not self.is_valid or self.tau_floor <= 0.0:
            return math.nan
        return math.log10(self.g_min / self.tau_floor)

    def display_tau(self) -> float:
        """A ``tau`` to quote in prose: the geometric midpoint of the band.

        A presentation value, not a derived constant: :attr:`is_invariant` makes it immaterial.
        Infinite ``g_min`` returns ``tau_floor``; zero ``tau_floor`` returns ``g_min / 2``, as 0
        is the G3 exact-tie baseline. Rationale: docs/engineering_notes.md#display-tau
        """
        if not self.is_valid:
            return math.nan
        if math.isinf(self.g_min):
            return self.tau_floor
        if self.tau_floor <= 0.0:
            return self.g_min / 2.0
        return math.sqrt(self.tau_floor * self.g_min)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau_floor": self.tau_floor,
            "tau_floor_hex": float.hex(self.tau_floor),
            "g_min": self.g_min,
            "g_min_hex": float.hex(self.g_min),
            "is_valid": self.is_valid,
            "is_invariant": self.is_invariant,
            "decades": self.decades,
            "display_tau": self.display_tau(),
            "n_gaps_in_band": self.n_gaps_in_band,
            "n_exact_ties": self.n_exact_ties,
            "n_positive_gaps": self.n_positive_gaps,
        }


def measure_noise_floor(
    model: TfidfModel,
    queries: Sequence[SparseVector],
) -> NoiseFloor:
    """Measure each reduction policy's error and return the :class:`NoiseFloor`.

    Row norms and dot product in :func:`cosine_against_corpus` both run under each policy, the
    norm's longer sum carrying most error; weights have no sum and a correctly rounded log (G13).
    ``queries`` must already be in ``model``'s space. Rationale: docs/engineering_notes.md#norms
    """
    documents = [model.document(i) for i in range(model.n_documents)]
    exact_norms = model.matrix.row_norms(Reduction.EXACT)

    per_policy: list[PolicyError] = []
    for policy in _INSTRUMENTS:
        policy_norms = model.matrix.row_norms(policy)
        n_compared = n_differing = 0
        max_abs = 0.0
        max_ulps = 0.0
        for query in queries:
            exact = cosine_against_corpus(query, documents, exact_norms, Reduction.EXACT)
            got = cosine_against_corpus(query, documents, policy_norms, policy)
            for a, b in zip(got, exact, strict=True):
                n_compared += 1
                if bits_of(a) != bits_of(b):
                    n_differing += 1
                    max_abs = max(max_abs, abs(a - b))
                    # abs(): `ulps_between` is signed and `max_ulps` starts at
                    # zero, so a negative distance would not register in the max.
                    max_ulps = max(max_ulps, abs(ulps_between(a, b)))
        per_policy.append(PolicyError(str(policy), n_compared, n_differing, max_abs, max_ulps))

    return NoiseFloor(
        per_policy=tuple(per_policy),
        n_queries=len(queries),
        n_documents=model.n_documents,
    )


def tau_band(floor: NoiseFloor, sorted_score_vectors: Sequence[Sequence[float]]) -> TauBand:
    """Sandwich ``tau`` between the arithmetic floor and the score lattice.

    ``floor`` is the measured floor; ``sorted_score_vectors`` holds one non-increasing vector
    per query. Returns a :class:`TauBand`; check :attr:`~TauBand.is_valid` first. An empty band
    is a result, not an error: noise reaches the decision boundary; A1/A2 regimes are not separable.
    """
    tau_floor = floor.tau_floor
    positive: list[float] = []
    n_exact = 0

    for scores in sorted_score_vectors:
        for gap in adjacent_gaps(scores):
            if gap == 0.0:
                n_exact += 1
            else:
                positive.append(gap)

    g_min = min(positive, default=math.inf)
    # A gap inside the band would break piecewise constancy, and none can land
    # there: every positive gap is >= g_min and the band is open at g_min. So
    # this checks the definition rather than measuring anything. Counted rather
    # than asserted, so a change to the definition surfaces as a number.
    n_in_band = sum(1 for gap in positive if tau_floor <= gap < g_min)

    return TauBand(
        tau_floor=tau_floor,
        g_min=g_min,
        n_gaps_in_band=n_in_band,
        n_exact_ties=n_exact,
        n_positive_gaps=len(positive),
    )


def verify_band_invariance(
    band: TauBand,
    sorted_score_vectors: Sequence[Sequence[float]],
    probes: int = 8,
) -> bool:
    """Recompute the tie structure across the band and check it never moves.

    Probes ``probes`` log-spaced taus over ``band`` on ``sorted_score_vectors``; ``False`` for an
    invalid band, else ``ValueError`` if ``probes < 1``. Piecewise constancy proves invariance
    with no gap in the band; this exercises the code and catches a wrong-strictness comparison.
    """
    if not band.is_valid:
        return False

    # Log spacing needs a positive finite band. `tau_floor` is 0 when every policy is
    # correctly rounded, and `log10(0)` raises; `g_min` is inf when no strictly-positive
    # gap exists. A degenerate endpoint is substituted so the reachable part is probed.
    upper = band.g_min if math.isfinite(band.g_min) else max(band.tau_floor, 1.0) * 1e6
    lower = band.tau_floor if band.tau_floor > 0.0 else min(upper, 1.0) * 1e-300

    if probes < 1:
        raise ValueError(f"probes must be at least 1, got {probes}")

    lo = math.log10(lower)
    hi = math.log10(upper)
    # A single probe cannot be spaced across the band, and `i / (probes - 1)` would
    # divide by zero. One probe starts at the lower endpoint; the clamp below moves it
    # to the top of the band whenever `g_min` is finite.
    if probes == 1:
        taus = [lower]
    else:
        taus = [10.0 ** (lo + (hi - lo) * i / (probes - 1)) for i in range(probes)]
    # The top end must stay strictly inside the band. The clamp runs before the
    # exact-tie baseline is appended, so `taus[-1]` is still the sweep's last
    # point; clamping the appended baseline would drop tau = 0.
    if math.isfinite(band.g_min):
        taus[-1] = math.nextafter(band.g_min, 0.0)
    # tau = 0 is admissible whenever the floor is 0, and it is the exact-tie
    # baseline.
    if band.tau_floor <= 0.0:
        taus.append(0.0)

    reference: list[tuple[tuple[int, ...], ...]] | None = None
    for tau in taus:
        shape = [tuple(tuple(g) for g in tie_chains(s, tau)) for s in sorted_score_vectors]
        if reference is None:
            reference = shape
        elif shape != reference:
            return False
    return True
