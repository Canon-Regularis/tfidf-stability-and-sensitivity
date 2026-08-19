"""Deriving ``tau`` instead of choosing it (section 7.1).

The paper constrains ``tau`` qualitatively from both sides: it should "exceed
floating-point noise while remaining small relative to typical score
separations". That is an interval with no procedure for picking a point inside
it, hence no ``tau`` key in ``configs/default.yaml`` and an explicit argument on
every function here. This module measures both endpoints and shows the choice
between them does not matter.

Lower endpoint: arithmetic
--------------------------
``tau`` decides whether ``|s_i - s_j| <= tau`` is evidence of a real difference
or an artefact of arithmetic, so the quantity to bound is the error in a margin.
Score error at most ``eta`` gives margin error at most ``2 * eta``, with both
signs attainable::

    tau_floor = 2 * eta

Same factor 2 as in section 4.4's ``eps_k^flip = m_k / 2``.

``eta`` is measured rather than bounded a priori: the four reduction policies run
over one corpus with :data:`Reduction.EXACT` (Shewchuk/``math.fsum``) as
correctly-rounded ground truth, and ``eta`` is the worst per-score disagreement.
Weights are policy-independent (``w = tf * idf`` is one division, one
correctly-rounded logarithm (G13) and one multiply, no summation), so all
disagreement lives in the two reductions downstream: row norms and dot product.
Both must vary. ``TfidfModel.norms`` is precomputed under the model's own
reduction, so varying only the scorer's policy holds the norms fixed and
understates ``eta`` by roughly threefold: a query dot product runs over a handful
of shared terms, a norm sums the whole document vector, and the longer summation
is where the error accumulates.

Upper endpoint: structure
-------------------------
``g_min``, the smallest strictly-positive adjacent gap observed. Below it no pair
of distinctly scored documents is within ``tau``.

Why the band needs no sweep
---------------------------
Every ``tau``-dependent object in this repository is piecewise constant in
``tau``, with breakpoints only at observed gap values:

* ``tie_chains`` cuts wherever an adjacent gap exceeds ``tau``, so it moves only
  when ``tau`` crosses a gap;
* ``tie_cliques`` admits an interval when its diameter is ``<= tau``, and every
  diameter is a sum of gaps;
* ``tie_ball(j, tau)`` is delimited by ``|s_i - s_j| <= tau``, again a gap sum.

So if the open band ``[tau_floor, g_min)`` contains no observed gap, every
``tau`` in it yields bit-identical tie structure, by argument rather than by
sweep. :func:`tau_band` returns the band; :func:`verify_band_invariance`
recomputes across it.

No circularity: ``tau_floor`` comes from arithmetic and ``g_min`` from the score
lattice, and neither is a function of ``tau``.
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

        Module docstring has the factor of 2.
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

        True when no observed gap falls inside the band; piecewise constancy
        (module docstring) then settles it by argument, without sampling values
        and finding they agreed.
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

        A presentation choice, so a figure caption can name a number without
        implying it was fitted. It must never be treated as a derived constant;
        :attr:`is_invariant` is the claim, and it says the value is immaterial.

        Two degenerate bands, both reachable:

        * ``g_min`` infinite, when no strictly-positive gap exists and every
          score is identical. The midpoint would be ``inf``, at which every tie
          ball swallows the corpus. No gap exists to cross, so every admissible
          tau gives the same structure and the smallest is returned.
        * ``tau_floor`` zero, when every reduction policy was correctly-rounded;
          Neumaier measured zero error on this corpus. The midpoint collapses to
          0, legal under G3 but the exact-tie baseline, so half the upper
          endpoint is returned.
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

    The norms are recomputed under each policy, and that matters more than the
    dot product. Passing a different policy to :func:`cosine_against_corpus`
    alone varies the dot product and leaves ``TfidfModel.norms`` at the model's
    own reduction, understating the floor: a dot product runs over the
    intersection of query and document terms (a handful), a norm over the whole
    document vector (tens), and the longer summation is where the error
    accumulates.

    The weight matrix is policy-independent (``w = tf * idf`` is one division,
    one correctly-rounded logarithm (G13) and one multiply, no summation), so one
    fit suffices and only the two reductions downstream vary.

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
        an empty band is a finding rather than an error, meaning arithmetic noise
        reaches the decision boundary and A1's and A2's regimes are not separable
        on this corpus.
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

    Piecewise constancy already proves invariance when no gap lies inside the
    band. This probes it anyway at logarithmically spaced taus: the argument is
    about the specification and this exercises the code, so it would catch a
    ``tau`` comparison made with the wrong strictness.

    Returns:
        ``True`` if every probe gives identical chain structure.
    """
    if not band.is_valid:
        return False

    from tfidf_stability.ranking.tie_groups import tie_chains

    # Log spacing needs a positive finite band, and either endpoint can be
    # degenerate: `tau_floor` is 0 when every policy was correctly-rounded and
    # `log10(0)` raises; `g_min` is inf when no strictly-positive gap exists.
    # Substitute and probe the reachable part instead of crashing or sampling
    # nothing.
    upper = band.g_min if math.isfinite(band.g_min) else max(band.tau_floor, 1.0) * 1e6
    lower = band.tau_floor if band.tau_floor > 0.0 else min(upper, 1.0) * 1e-300

    if probes < 1:
        raise ValueError(f"probes must be at least 1, got {probes}")

    lo = math.log10(lower)
    hi = math.log10(upper)
    # A single probe cannot be spaced across the band, and `i / (probes - 1)`
    # divided by zero rather than saying so. One probe is the lower endpoint.
    if probes == 1:
        taus = [lower]
    else:
        taus = [10.0 ** (lo + (hi - lo) * i / (probes - 1)) for i in range(probes)]
    # tau = 0 is admissible whenever the floor is 0, and it is the exact-tie
    # baseline.
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
