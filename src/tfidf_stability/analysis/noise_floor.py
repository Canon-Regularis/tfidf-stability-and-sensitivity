"""Deriving ``tau`` instead of choosing it (section 7.1).

The paper's complete guidance on ``tau`` is a two-sided *qualitative* constraint:
it should "exceed floating-point noise while remaining small relative to typical
score separations". That defines an interval, not a value, and the paper gives no
procedure for resolving it -- which is why ``configs/default.yaml`` deliberately
has no ``tau`` key and why every function here takes it explicitly.

This module measures both endpoints and then shows the choice inside them does
not matter.

The lower endpoint: arithmetic
------------------------------
``tau``'s operational job is to decide whether ``|s_i - s_j| <= tau`` is evidence
of a real difference or an artefact of arithmetic. The quantity to bound is
therefore the error in a **margin**, not in a score. If each score carries error
at most ``eta``, the margin ``s_i - s_j`` carries at most ``2 * eta``, and both
signs are attainable. So::

    tau_floor = 2 * eta

The factor 2 is exact, and it is the same 2 as in section 4.4's
``eps_k^flip = m_k / 2`` -- not a safety fudge.

``eta`` is measured, not bounded a priori: the four reduction policies are run
over the same corpus with :data:`Reduction.EXACT` (Shewchuk/``math.fsum``) as
correctly-rounded ground truth, and ``eta`` is the worst per-score disagreement.

The weights themselves are policy-independent -- ``w = tf * idf`` is one
division, one correctly-rounded logarithm (G13) and one multiply, with no
summation anywhere -- so all disagreement lives in the two reductions downstream:
the row norms and the dot product. One fit therefore suffices, and only those two
need to vary. **Both must.** ``TfidfModel.norms`` is precomputed under the
model's own reduction, so varying only the policy passed to the scorer would hold
the norms fixed and understate ``eta`` by roughly threefold -- a query dot product
runs over a handful of shared terms, whereas a norm sums the whole document
vector, and the longer summation is where the error accumulates.

The upper endpoint: structure
-----------------------------
The smallest strictly-positive adjacent gap actually observed. Below it, no pair
of *distinctly* scored documents is within ``tau`` of each other.

Why the band can be proved rather than sampled
----------------------------------------------
Every ``tau``-dependent object in this repository is **piecewise constant in
tau**, with breakpoints only at observed gap values:

* ``tie_chains`` cuts wherever an adjacent gap exceeds ``tau``, so it changes only
  when ``tau`` crosses a gap;
* ``tie_cliques`` admits an interval when its diameter is ``<= tau``, and every
  diameter is a sum of gaps;
* ``tie_ball(j, tau)`` is delimited by ``|s_i - s_j| <= tau``, again a gap sum.

So if the open band ``[tau_floor, g_min)`` contains **no** observed gap, every
``tau`` in it yields bit-identical tie structure -- not "approximately the same",
identical, and by an argument rather than by a sweep. That is what
:func:`tau_band` returns and what :func:`verify_band_invariance` checks by
recomputation.

This also discharges the circularity objection. ``tau_floor`` comes from
arithmetic and ``g_min`` from the score lattice; neither is a function of
``tau``. Nothing here is derived from a quantity that ``tau`` then determines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tfidf_stability.ranking.margins import adjacent_gaps
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
            # The hex form is the auditable one: a decimal literal round-trips
            # through the reader's parser, and this value is a bound.
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
        """``2 * eta`` -- the smallest ``tau`` that cannot be arithmetic noise.

        See the module docstring for why the factor is exactly 2.
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

        True exactly when no observed gap falls inside the band -- which is the
        piecewise-constancy argument from the module docstring, and is stronger
        than "we tried several values and they agreed".
        """
        return self.is_valid and self.n_gaps_in_band == 0

    @property
    def decades(self) -> float:
        """Width of the band in orders of magnitude."""
        if not self.is_valid or self.tau_floor <= 0.0:
            return math.nan
        return math.log10(self.g_min / self.tau_floor)

    def display_tau(self) -> float:
        """A ``tau`` to quote in prose, as a **presentation choice**.

        The geometric midpoint of the band. This is not a derived constant and
        must never be treated as one: :attr:`is_invariant` is the claim, and it
        says the specific value is immaterial. It exists so a figure caption can
        name a number without implying the number was fitted.

        Two degenerate bands need care, and both are reachable:

        * ``g_min`` is infinite when the corpus has no strictly-positive gap at
          all -- every score identical. The geometric midpoint would then be
          ``inf``, which is not a usable tolerance: at ``tau = inf`` every tie
          ball swallows the corpus. Since no gap exists to cross, every
          admissible tau gives the same structure, so the smallest one is
          returned.
        * ``tau_floor`` is zero when every reduction policy was exactly
          correctly-rounded, which does happen -- Neumaier was measured at zero
          error on this corpus. The geometric midpoint collapses to 0, which is
          legal (G3 admits ``tau = 0``) but is the exact-tie baseline rather
          than a midpoint, so half the upper endpoint is returned instead.
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
    """Measure how far each reduction policy strays from exact arithmetic.

    **The norms are recomputed under each policy**, which is easy to get wrong
    and matters more than the dot product. ``TfidfModel.norms`` is precomputed
    under the model's own reduction, so passing a different policy to
    :func:`cosine_against_corpus` alone would vary only the dot product and hold
    the norms fixed. That understates the floor badly here: a query dot product
    runs over the *intersection* of query and document terms (typically a
    handful), whereas a norm sums over the whole document vector (tens of
    terms). The longer summation is where the error actually accumulates.

    The weight matrix itself is genuinely policy-independent -- ``w = tf * idf``
    is one division, one correctly-rounded logarithm (G13) and one multiply, with
    no summation anywhere -- so one fit suffices and only the two reductions
    downstream of it need to vary.

    Args:
        model: A fitted model, supplying the weight matrix.
        queries: Query vectors already embedded into ``model``'s space.

    Returns:
        The :class:`NoiseFloor`.
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
                    max_ulps = max(max_ulps, ulps_between(a, b))
        per_policy.append(PolicyError(str(policy), n_compared, n_differing, max_abs, max_ulps))

    return NoiseFloor(
        per_policy=tuple(per_policy),
        n_queries=len(queries),
        n_documents=model.n_documents,
    )


def tau_band(floor: NoiseFloor, sorted_score_vectors: Sequence[Sequence[float]]) -> TauBand:
    """Sandwich ``tau`` between the arithmetic floor and the score lattice.

    Args:
        floor: The measured arithmetic floor.
        sorted_score_vectors: One non-increasing score vector per query.

    Returns:
        The :class:`TauBand`. Check :attr:`~TauBand.is_valid` before using it:
        an empty band is a *finding*, not an error -- it would mean arithmetic
        noise reaches the decision boundary, and that A1's and A2's regimes are
        not separable on this corpus.
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
    # A gap inside the band is what would break piecewise-constancy. By
    # construction none can be -- every positive gap is >= g_min, and the band
    # is open at g_min -- so this is a self-check on the definition rather than
    # a measurement. It is counted, not asserted, so that a future change to how
    # the band is defined shows up as a number rather than a crash.
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

    The piecewise-constancy argument already *proves* invariance when no gap
    lies inside the band. This recomputes it anyway, at logarithmically spaced
    probes, because an argument about the code and the code itself are different
    things -- and this is the one that would catch an implementation that
    compared against ``tau`` with the wrong strictness.

    Returns:
        ``True`` if every probe gives identical chain structure.
    """
    if not band.is_valid:
        return False

    from tfidf_stability.ranking.tie_groups import tie_chains

    # Logarithmic spacing needs a positive, finite band. Both endpoints can be
    # degenerate: `tau_floor` is 0 when every policy was exactly
    # correctly-rounded, and `log10(0)` raises; `g_min` is infinite when no
    # strictly-positive gap exists. Probe the reachable part in those cases
    # rather than crashing or sampling nothing.
    upper = band.g_min if math.isfinite(band.g_min) else max(band.tau_floor, 1.0) * 1e6
    lower = band.tau_floor if band.tau_floor > 0.0 else min(upper, 1.0) * 1e-300

    lo = math.log10(lower)
    hi = math.log10(upper)
    taus = [10.0 ** (lo + (hi - lo) * i / (probes - 1)) for i in range(probes)]
    # tau = 0 is admissible whenever the floor is 0, and it is the exact-tie
    # baseline -- the single most important point to check.
    if band.tau_floor <= 0.0:
        taus.append(0.0)
    # The top end must stay strictly inside the band.
    if math.isfinite(band.g_min):
        taus[probes - 1] = math.nextafter(band.g_min, 0.0)

    reference: list[tuple[tuple[int, ...], ...]] | None = None
    for tau in taus:
        shape = [tuple(tuple(g) for g in tie_chains(s, tau)) for s in sorted_score_vectors]
        if reference is None:
            reference = shape
        elif shape != reference:
            return False
    return True
