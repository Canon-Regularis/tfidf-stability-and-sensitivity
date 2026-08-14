"""Score and ranking stability (README sections 4.3 and 4.4).

Section 4.3 bounds how far a similarity score can move when the vectors move;
section 4.4 turns a score bound into a *ranking* guarantee via the margin.

    section 4.3   |cos(u',v') - cos(u,v)| <= C (||u'-u|| + ||v'-v||)
    section 4.4   |ds_i| <= eps  and  eps < m_k / 2   =>  top-k set is invariant
                  |ds_i| <= eps  and  eps < m_min^top / 2  =>  top-k order too

The explicit constant ``C = 1/L`` lives in
:mod:`~tfidf_stability.similarity.geometry` (``spec_addenda.md#g4``); this module
covers what follows for the ranking.

Two things here go beyond the paper.

The radius is necessary as well as sufficient. Section 4.4 states ``eps < m_k/2``
as a sufficient condition and never addresses necessity; :func:`flip_witness`
constructs the perturbation that breaks it at ``m_k/2 + delta``, built from
dyadic rationals wherever the margin permits so the construction carries no
rounding of its own.

A certificate is a bound rather than a prediction. :func:`certified_radius`
reports the largest perturbation a ranking provably tolerates and claims nothing
about typical perturbations. Section 7.2 uses these as "empirical certificates of
stability"; reading them as expected behaviour inverts the meaning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tfidf_stability.ranking.margins import boundary_margin, min_adjacent_margin_top
from tfidf_stability.utils.validation import StrictMode

__all__ = [
    "StabilityCertificate",
    "certified_radius",
    "flip_witness",
    "is_order_stable",
    "is_top_k_stable",
]


@dataclass(frozen=True, slots=True)
class StabilityCertificate:
    """What a ranking provably tolerates at one ``k`` (section 4.4).

    Attributes:
        k: The boundary rank.
        set_radius: Largest ``eps`` for which the top-k set is guaranteed
            invariant, ``m_k / 2``. ``NaN`` when ``m_k`` is undefined.
        order_radius: Largest ``eps`` for which the top-k ordering is guaranteed
            invariant, ``m_min^top / 2``.
        exact_tie: ``m_k == 0``, so the certificate is ``0``: no radius at all.
            Membership already depends entirely on the tie-break, so a change of
            operator flips it with ``ds = 0``.
        defined: Whether a certificate exists at all.
    """

    k: int
    set_radius: float
    order_radius: float
    exact_tie: bool
    defined: bool

    @property
    def order_radius_is_binding(self) -> bool:
        """Whether the order radius is the tighter of the two here.

        Neither radius dominates the other in general: ``m_min^top`` minimises
        over the gaps strictly inside the top-k (ranks 1->2 through (k-1)->k)
        while ``m_k`` is the gap at the boundary (rank k->k+1), and those gap
        sets are disjoint. A tight cluster at the top with a wide boundary gives
        ``order_radius < set_radius``; a well-spread top with a near-tied
        boundary gives the reverse.

        So a certificate quoted without saying which invariant it certifies is
        ambiguous. Both are carried, and :attr:`joint_radius` covers callers who
        want both guarantees.
        """
        if not self.defined or math.isnan(self.order_radius):
            return False
        return self.order_radius < self.set_radius

    @property
    def joint_radius(self) -> float:
        """Largest ``eps`` guaranteeing the top-k set and its ordering.

        The minimum of the two radii, since section 4.4's conditions constrain
        disjoint sets of gaps and both must hold.
        """
        if not self.defined:
            return math.nan
        if math.isnan(self.order_radius):
            return self.set_radius
        return min(self.set_radius, self.order_radius)


def certified_radius(
    sorted_scores: Sequence[float],
    k: int,
    *,
    mode: StrictMode = StrictMode.LENIENT,
) -> StabilityCertificate:
    """The section 4.4 certificates at rank ``k``.

    Both radii come straight from the margins, which depend only on the sorted
    score multiset, so a certificate is a property of the scores and is the same
    under pi, pi_score and pi_alt.
    """
    boundary = boundary_margin(sorted_scores, k, mode=mode)
    order = min_adjacent_margin_top(sorted_scores, k, mode=mode)
    return StabilityCertificate(
        k=k,
        set_radius=boundary.flip_radius,
        order_radius=order.flip_radius,
        exact_tie=boundary.is_exact_tie,
        defined=boundary.defined,
    )


def is_top_k_stable(sorted_scores: Sequence[float], k: int, eps: float) -> bool:
    """Whether section 4.4 guarantees top-k set invariance under ``|ds| <= eps``.

    A sufficient condition, so ``False`` means "not guaranteed" and never "will
    change". The strict inequality is the paper's: at ``eps == m_k / 2`` the two
    boundary scores can be driven to equality, and membership passes to the
    tie-break.
    """
    cert = certified_radius(sorted_scores, k)
    return bool(cert.defined and not math.isnan(cert.set_radius) and eps < cert.set_radius)


def is_order_stable(sorted_scores: Sequence[float], k: int, eps: float) -> bool:
    """Whether section 4.4 guarantees the top-k ordering is preserved."""
    cert = certified_radius(sorted_scores, k)
    return bool(cert.defined and not math.isnan(cert.order_radius) and eps < cert.order_radius)


def flip_witness(
    scores: Sequence[float],
    order: Sequence[int],
    k: int,
    *,
    delta: float | None = None,
) -> tuple[list[float], float] | None:
    """Construct a perturbation that provably flips the top-k boundary.

    Establishes that section 4.4's radius is necessary as well as sufficient,
    which the paper does not address. The construction moves the two documents
    straddling the boundary past each other and nothing else:

        s'[r_k]     = s[r_k]     - eps
        s'[r_{k+1}] = s[r_{k+1}] + eps        with eps = m_k / 2 + delta

    so ``|ds_i| <= eps`` for every document while ``s'[r_{k+1}] > s'[r_k]``.

    Args:
        scores: The unperturbed score vector, indexed by document.
        order: A ranking's document order, best first.
        k: The boundary rank (1-indexed).
        delta: How far past the radius to go. Defaults to a value scaled to the
            margin, floored at a few ulp so the excess survives rounding.

    Returns:
        ``(perturbed_scores, eps)``, or ``None`` when no witness exists: ``k``
        out of range, or ``m_k == 0`` with no radius to exceed.

    Note:
        Where the margin is dyadic the construction is exact, since ``m/2`` only
        shifts an exponent and the additions land on representable values.
        Otherwise the caller checks the realised perturbation against ``eps``
        rather than assuming it: ``fl(s + eps)`` can differ from ``s + eps`` by
        up to half an ulp, which is what makes a naive version of this test
        flaky.
    """
    n = len(scores)
    if not 1 <= k < n:
        return None

    a, b = order[k - 1], order[k]
    margin = scores[a] - scores[b]
    if margin <= 0.0:
        return None  # already tied: no radius to exceed

    if delta is None:
        delta = max(margin / 2.0 * 1e-6, 4.0 * math.ulp(max(abs(scores[a]), 1.0)))
    eps = margin / 2.0 + delta

    perturbed = list(scores)
    perturbed[a] = scores[a] - eps
    perturbed[b] = scores[b] + eps

    # The witness is only a witness if it actually reverses the pair.
    if not perturbed[b] > perturbed[a]:
        return None
    return perturbed, eps
