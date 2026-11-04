"""IDF perturbation and union-vocabulary alignment (section 4.1, G5).

Section 4.1: ``delta_idf(t) = log((1 + N') / (1 + df'(t))) - log((1 + N) / (1 + df(t)))``.
G5 (``spec_addenda.md#g5``) defines it on ``V u V'``, zero outside each vocabulary, so
``delta_idf(t) = idf'(t)`` on ``V' \\ V``; :class:`Alignment` gives the Pythagorean split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tfidf_stability.utils.numerics import Reduction, reduce_sum, sqrt
from tfidf_stability.vectorisation.tfidf import TfidfModel

__all__ = ["Alignment", "IdfPerturbation", "align_models", "analyse_idf_shift"]


@dataclass(frozen=True, slots=True)
class Alignment:
    """Two models expressed over their union vocabulary.

    Arrays are indexed by position in :attr:`tokens`, the union in ascending
    UTF-8 byte order, which is the vocabulary's own ordering rule.
    Rationale: docs/engineering_notes.md#idf-union-vocabulary
    """

    tokens: tuple[str, ...]
    #: Token index sets, for the Pythagorean split.
    in_before: tuple[bool, ...]
    in_after: tuple[bool, ...]
    idf_before: tuple[float, ...]
    idf_after: tuple[float, ...]

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def shared(self) -> tuple[int, ...]:
        """Indices of ``V n V'``, where a coordinate genuinely changed."""
        return tuple(i for i in range(self.n_tokens) if self.in_before[i] and self.in_after[i])

    @property
    def gained(self) -> tuple[int, ...]:
        """Indices of ``V' \\ V``: tokens the perturbation created."""
        return tuple(i for i in range(self.n_tokens) if self.in_after[i] and not self.in_before[i])

    @property
    def lost(self) -> tuple[int, ...]:
        """Indices of ``V \\ V'``: tokens the perturbation destroyed."""
        return tuple(i for i in range(self.n_tokens) if self.in_before[i] and not self.in_after[i])

    @property
    def vocabulary_changed(self) -> bool:
        return bool(self.gained or self.lost)

    def delta_idf(self) -> tuple[float, ...]:
        """``idf' - idf`` on the union vocabulary."""
        return tuple(self.idf_after[i] - self.idf_before[i] for i in range(self.n_tokens))

    def embed_before(self, model: TfidfModel, doc_index: int) -> list[float]:
        """A document's TF-IDF vector from the *before* model, on the union."""
        return self._embed(model, doc_index, self.tokens)

    def embed_after(self, model: TfidfModel, doc_index: int) -> list[float]:
        """A document's TF-IDF vector from the *after* model, on the union."""
        return self._embed(model, doc_index, self.tokens)

    @staticmethod
    def _embed(model: TfidfModel, doc_index: int, tokens: Sequence[str]) -> list[float]:
        row = model.matrix.row(doc_index)
        by_token = {
            model.vocabulary.token_of(t): v for t, v in zip(row.indices, row.values, strict=True)
        }
        return [by_token.get(tok, 0.0) for tok in tokens]


def align_models(before: TfidfModel, after: TfidfModel) -> Alignment:
    """Put two models on their union vocabulary (G5).

    The union is sorted in ascending UTF-8 byte order, matching
    :mod:`~tfidf_stability.vectorisation.vocabulary`, so the alignment is a pure
    function of the two token sets.
    """
    tokens = tuple(
        sorted(
            set(before.vocabulary.tokens) | set(after.vocabulary.tokens),
            key=lambda t: t.encode("utf-8"),
        )
    )
    in_before = tuple(t in before.vocabulary for t in tokens)
    in_after = tuple(t in after.vocabulary for t in tokens)

    def idf_of(model: TfidfModel, token: str) -> float:
        term_id = model.vocabulary.id_of(token)
        return 0.0 if term_id is None else model.idf[term_id]

    return Alignment(
        tokens=tokens,
        in_before=in_before,
        in_after=in_after,
        idf_before=tuple(idf_of(before, t) for t in tokens),
        idf_after=tuple(idf_of(after, t) for t in tokens),
    )


@dataclass(frozen=True, slots=True)
class IdfPerturbation:
    """Section 4.1's analysis of how the IDF vector moved."""

    alignment: Alignment
    n_before: int
    n_after: int
    #: ``||delta_idf||_inf`` over the union: what section 4.2's bound uses.
    linf: float
    #: The same, restricted to shared tokens. Smaller whenever the vocabulary
    #: churned; the gap between the two is the looseness G5 warns of.
    linf_shared: float
    #: The token that moved most, and by how much.
    worst_token: str
    worst_delta: float

    @property
    def looseness(self) -> float:
        """``linf / linf_shared``: how much vocabulary churn inflates the bound.

        ``1.0`` means the vocabulary was stable and the section 4.2 bound is as
        tight as it gets. Large values mean the bound is driven by tokens that
        did not exist on one side, and should be read with that in mind.
        """
        if self.linf_shared == 0.0:
            return float("inf") if self.linf > 0.0 else 1.0
        return self.linf / self.linf_shared


def analyse_idf_shift(before: TfidfModel, after: TfidfModel) -> IdfPerturbation:
    """Measure the IDF movement between two models (section 4.1).

    Section 4.1 notes that low-``df`` tokens stay sensitive even under
    smoothing, so the worst-moving token is usually a rare one: ``idf`` is
    logarithmic in ``df`` and steepest where ``df`` is small.
    """
    alignment = align_models(before, after)
    deltas = alignment.delta_idf()

    shared = alignment.shared
    linf = max((abs(d) for d in deltas), default=0.0)
    linf_shared = max((abs(deltas[i]) for i in shared), default=0.0)

    worst_index = max(range(len(deltas)), key=lambda i: abs(deltas[i])) if deltas else 0
    return IdfPerturbation(
        alignment=alignment,
        n_before=before.vocabulary.n_documents,
        n_after=after.vocabulary.n_documents,
        linf=linf,
        linf_shared=linf_shared,
        worst_token=alignment.tokens[worst_index] if alignment.tokens else "",
        worst_delta=deltas[worst_index] if deltas else 0.0,
    )


def l2(values: Sequence[float], policy: Reduction = Reduction.NAIVE) -> float:
    """Euclidean norm of a dense sequence, under an explicit reduction policy."""
    return sqrt(reduce_sum([v * v for v in values], policy))
