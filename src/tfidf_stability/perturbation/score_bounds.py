"""Score and ranking stability bounds (README sections 4.3 and 4.4).

Section 4.3: ``|cos(u',v') - cos(u,v)| <= C (||u'-u|| + ||v'-v||)`` with ``C = 1/L`` from
:mod:`~tfidf_stability.similarity.geometry` (``spec_addenda.md#g4``). Section 4.4: if
``|ds_i| <= eps < m_k/2`` the top-k set is invariant; if ``eps < m_min^top/2``, its order too.
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
    """What a ranking provably tolerates at boundary rank ``k`` (section 4.4).

    ``set_radius`` is ``m_k / 2`` (``NaN`` when ``m_k`` is undefined), ``order_radius`` is
    ``m_min^top / 2``. ``exact_tie`` means ``m_k == 0``: the radius is 0, and a change of tie-break
    operator flips membership with ``ds = 0``. ``defined`` says whether a certificate exists.
    """

    k: int
    set_radius: float
    order_radius: float
    exact_tie: bool
    defined: bool

    @property
    def order_radius_is_binding(self) -> bool:
        """Whether the order radius is the tighter of the two here.

        Neither dominates: ``m_min^top`` minimises over the gaps inside the top-k (ranks
        1->2 through (k-1)->k) and ``m_k`` is the disjoint boundary gap (k->k+1). A tight
        top cluster over a wide boundary makes this true; :attr:`joint_radius` covers both.
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

    The radii come from the margins, which depend only on the sorted score multiset, so the
    certificate is the same under pi, pi_score and pi_alt. It is a worst-case bound, not a
    prediction. Rationale: docs/engineering_notes.md#certificate-is-a-bound
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

    Built on :func:`certified_radius`. The condition is sufficient, so ``False`` means "not
    guaranteed", never "will change". At ``eps == m_k / 2`` the boundary scores can meet and
    membership passes to the tie-break; :func:`flip_witness` exceeds that radius.
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

    Moves ``s[r_k]`` down and ``s[r_{k+1}]`` up by ``eps = m_k/2 + delta`` (``k`` 1-indexed,
    ``order`` best first). Returns ``(perturbed_scores, eps)``; ``None`` if ``k`` is out of range,
    ``m_k <= 0`` or the pair does not reverse. Rationale: docs/engineering_notes.md#flip-witness
    """
    # Bounded by whichever sequence runs out first. `order` may be a truncated Ranking,
    # since `rank_top_k` returns `n_selected = min(k + 1, n)`, and the boundary needs
    # `order[k - 1]` and `order[k]`. A `k` past the end of `order` returns None, as the
    # signature documents, rather than raising IndexError.
    n = min(len(scores), len(order))
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
