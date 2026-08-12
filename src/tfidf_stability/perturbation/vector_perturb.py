"""TF-IDF vector perturbation: section 4.2's three-term decomposition.

Section 4.2 writes ``w = tf (*) idf`` and expands

    w' - w = (dtf) (*) idf + tf (*) (didf) + (dtf) (*) (didf)

then applies ``||a (*) b||_2 <= ||a||_2 ||b||_inf`` termwise to get

    ||w' - w||_2 <= ||dtf||_2 ||idf||_inf
                  + ||tf||_2  ||didf||_inf
                  + ||dtf||_2 ||didf||_inf

separating a **local** document edit, a **global** corpus change, and their
**interaction**.

This module evaluates all three terms and, separately, the quantity they bound,
so a caller can assert the inequality and report how tight it was. Which of the
three dominates is also reported: section 4.2 claims the interaction term
provides "a natural mechanism for perturbation amplification" but offers no
evidence, and the decomposition is what turns that into something measurable.

Everything is computed on the **union vocabulary** (``spec_addenda.md#g5``),
because a corpus perturbation moves the vocabulary and the bound as stated
presupposes a fixed index set. The exact Pythagorean split is reported
alongside, so the part of the movement that is genuine coordinate change can be
separated from the part that is vocabulary churn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tfidf_stability.perturbation.idf_perturb import Alignment, align_models, l2
from tfidf_stability.utils.numerics import Reduction, reduce_sum, sqrt
from tfidf_stability.vectorisation.tfidf import TfidfModel

__all__ = ["ThreeTermBound", "VectorPerturbation", "analyse_vector_shift", "three_term_terms"]


@dataclass(frozen=True, slots=True)
class ThreeTermBound:
    """Section 4.2's bound, with its three constituents kept apart."""

    #: ``||dtf||_2 * ||idf||_inf`` -- the local document edit.
    local: float
    #: ``||tf||_2 * ||didf||_inf`` -- the global corpus change.
    glob: float
    #: ``||dtf||_2 * ||didf||_inf`` -- their interaction.
    interaction: float
    #: What the bound is bounding.
    observed: float

    @property
    def total(self) -> float:
        return self.local + self.glob + self.interaction

    @property
    def holds(self) -> bool:
        """Whether the inequality is respected.

        The slack covers rounding incurred while *evaluating* the bound, not
        slack in the mathematics: every term is a product of two norms, each of
        which is itself a rounded square root.
        """
        return self.observed <= self.total * (1.0 + 1e-12) + 1e-15

    @property
    def tightness(self) -> float:
        """``observed / total`` -- how close the bound comes to being attained."""
        return self.observed / self.total if self.total > 0.0 else 0.0

    @property
    def dominant_term(self) -> str:
        """Which of the three contributes most.

        Section 4.2 asserts the interaction term matters without showing when.
        This is the quantity that answers it.
        """
        return max(
            (("local", self.local), ("global", self.glob), ("interaction", self.interaction)),
            key=lambda pair: pair[1],
        )[0]


def three_term_terms(
    tf: Sequence[float],
    delta_tf: Sequence[float],
    idf_linf: float,
    delta_idf_linf: float,
    observed: float,
    *,
    # Keyword-only: the reduction chooses which summation order the two l2 calls
    # below use, so it is the one argument here that changes the returned bits.
    # A bare `Reduction.EXACT` trailing five floats would hide that.
    policy: Reduction = Reduction.NAIVE,
) -> ThreeTermBound:
    """Assemble section 4.2's bound from its parts."""
    n_dtf = l2(delta_tf, policy)
    n_tf = l2(tf, policy)
    return ThreeTermBound(
        local=n_dtf * idf_linf,
        glob=n_tf * delta_idf_linf,
        interaction=n_dtf * delta_idf_linf,
        observed=observed,
    )


@dataclass(frozen=True, slots=True)
class VectorPerturbation:
    """One document's movement between two models, fully decomposed."""

    doc_id: str
    bound: ThreeTermBound
    alignment: Alignment
    #: ``||(w' - w) restricted to V n V'||`` -- genuine coordinate change.
    shared_shift: float
    #: ``||w' restricted to V' \\ V||`` -- mass on tokens that did not exist before.
    gained_mass: float
    #: ``||w restricted to V \\ V'||`` -- mass on tokens that ceased to exist.
    lost_mass: float

    @property
    def pythagoras_holds(self) -> bool:
        """The three parts are supported on disjoint coordinate sets, so

            ||w' - w||^2 = shared^2 + gained^2 + lost^2

        holds *exactly* in the reals. In binary64 it holds to rounding, and
        checking it is a cheap guard against an alignment bug -- a misaligned
        index would break this identity long before it broke any inequality.
        """
        lhs = self.bound.observed**2
        rhs = self.shared_shift**2 + self.gained_mass**2 + self.lost_mass**2
        return abs(lhs - rhs) <= 1e-9 * max(1.0, lhs)

    @property
    def churn_fraction(self) -> float:
        """Share of the squared movement attributable to vocabulary churn.

        Zero when the vocabulary was stable. Large values mean the section 4.2
        bound is being driven by tokens that exist on only one side, which is
        the looseness G5 identifies.
        """
        total = self.bound.observed**2
        if total <= 0.0:
            return 0.0
        return (self.gained_mass**2 + self.lost_mass**2) / total


def analyse_vector_shift(
    before: TfidfModel,
    after: TfidfModel,
    doc_id: str,
    policy: Reduction = Reduction.NAIVE,
) -> VectorPerturbation:
    """Decompose one document's TF-IDF movement between two models.

    Args:
        before, after: Models fitted on the unperturbed and perturbed corpora.
        doc_id: A document present in *both* -- the section 4.2 bound compares a
            document with itself, so a document that was added or removed has no
            ``w' - w`` to bound.
        policy: Reduction policy for every norm.

    Raises:
        KeyError: If the document is absent from either model.
    """
    if doc_id not in before.doc_ids or doc_id not in after.doc_ids:
        raise KeyError(
            f"document {doc_id!r} must be present in both models; section 4.2 bounds "
            f"the movement of a document, not its creation or destruction"
        )

    i_before = before.doc_ids.index(doc_id)
    i_after = after.doc_ids.index(doc_id)
    alignment = align_models(before, after)

    w = alignment.embed_before(before, i_before)
    w_prime = alignment.embed_after(after, i_after)
    diff = [b - a for a, b in zip(w, w_prime, strict=True)]
    observed = l2(diff, policy)

    # tf is recoverable exactly: w = tf * idf, and idf is never zero on a token
    # that is in the vocabulary (idf >= 1 always -- see spec_addenda G4).
    def tf_of(weights: Sequence[float], idf: Sequence[float]) -> list[float]:
        return [
            (weight / scale) if scale != 0.0 else 0.0
            for weight, scale in zip(weights, idf, strict=True)
        ]

    tf = tf_of(w, alignment.idf_before)
    tf_prime = tf_of(w_prime, alignment.idf_after)
    delta_tf = [b - a for a, b in zip(tf, tf_prime, strict=True)]
    delta_idf = alignment.delta_idf()

    bound = three_term_terms(
        tf=tf,
        delta_tf=delta_tf,
        idf_linf=max((abs(v) for v in alignment.idf_before), default=0.0),
        delta_idf_linf=max((abs(v) for v in delta_idf), default=0.0),
        observed=observed,
        policy=policy,
    )

    def restricted_norm(indices: Sequence[int], values: Sequence[float]) -> float:
        return sqrt(reduce_sum([values[i] * values[i] for i in indices], policy))

    return VectorPerturbation(
        doc_id=doc_id,
        bound=bound,
        alignment=alignment,
        shared_shift=restricted_norm(alignment.shared, diff),
        gained_mass=restricted_norm(alignment.gained, w_prime),
        lost_mass=restricted_norm(alignment.lost, w),
    )
