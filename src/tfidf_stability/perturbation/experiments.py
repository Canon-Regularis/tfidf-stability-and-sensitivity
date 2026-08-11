"""Perturbation experiments: apply an edit, refit, measure everything (section 4).

Ties the pieces together. Given a corpus, a query and an edit, this reports the
whole chain section 4 describes -- corpus change, IDF shift, vector movement,
score movement, ranking consequence -- for one perturbation, in one object.

Section 4's framing is that a perturbation propagates:

    corpus edit -> df -> idf -> w -> cos -> ranking

and that each stage has a bound relating its output movement to its input
movement. :class:`PerturbationReport` records the measured movement at every
stage alongside the bound that governs it, so the chain can be inspected rather
than only its endpoints.

The pipeline is deliberately *not* incremental. Both models are fitted from
scratch, because a partial update that shared state with the baseline would be
the thing most likely to hide a real perturbation effect, and correctness
matters more here than speed -- these runs are a handful per experiment, not per
query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tfidf_stability.perturbation.corpus_edits import Corpus, EditRecord
from tfidf_stability.perturbation.idf_perturb import IdfPerturbation, analyse_idf_shift
from tfidf_stability.perturbation.score_bounds import StabilityCertificate, certified_radius
from tfidf_stability.perturbation.vector_perturb import VectorPerturbation, analyse_vector_shift
from tfidf_stability.ranking.ranker import sorted_scores_desc
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

__all__ = ["PerturbationReport", "run_perturbation"]


@dataclass(frozen=True, slots=True)
class PerturbationReport:
    """One perturbation, measured at every stage of section 4's chain."""

    edit: EditRecord
    idf_shift: IdfPerturbation
    #: Per-document vector movement, for documents present on both sides.
    vector_shifts: tuple[VectorPerturbation, ...]
    #: Score movement for the documents that survived the edit, keyed by id.
    score_before: dict[str, float]
    score_after: dict[str, float]
    certificates_before: tuple[StabilityCertificate, ...]

    @property
    def max_score_shift(self) -> float:
        """``max |ds_i|`` over surviving documents -- the ``eps`` of section 4.4."""
        shared = set(self.score_before) & set(self.score_after)
        return max((abs(self.score_after[d] - self.score_before[d]) for d in shared), default=0.0)

    @property
    def all_bounds_hold(self) -> bool:
        """Whether every section 4.2 bound was respected."""
        return all(v.bound.holds for v in self.vector_shifts)

    def certified_stable(self, k: int) -> bool | None:
        """Whether section 4.4 *guarantees* the top-k set survived this edit.

        Returns ``None`` when no certificate exists at that ``k``. Note the
        asymmetry: ``True`` is a proof, ``False`` merely means the certificate
        does not cover this perturbation -- the ranking may well be unchanged
        anyway. Section 7.2 uses these as certificates, not predictions.
        """
        for cert in self.certificates_before:
            if cert.k == k:
                if not cert.defined:
                    return None
                return self.max_score_shift < cert.set_radius
        return None

    @property
    def dominant_terms(self) -> dict[str, int]:
        """How often each of section 4.2's three terms dominated.

        Section 4.2 claims the interaction term is a mechanism for
        amplification; this is the count that lets the claim be checked against
        data rather than assumed.
        """
        counts = {"local": 0, "global": 0, "interaction": 0}
        for shift in self.vector_shifts:
            if shift.bound.total > 0.0:
                counts[shift.bound.dominant_term] += 1
        return counts


def run_perturbation(
    corpus: Corpus,
    perturbed: Corpus,
    edit: EditRecord,
    query_features: Sequence[str],
    *,
    ks: Sequence[int] = (5, 10, 20, 50),
) -> PerturbationReport:
    """Fit both corpora, score one query against each, and measure the chain.

    Args:
        corpus: The unperturbed corpus.
        perturbed: The result of applying ``edit`` to it.
        edit: What changed, as returned by the functions in
            :mod:`~tfidf_stability.perturbation.corpus_edits`.
        query_features: The query's preprocessed feature stream. Embedded
            separately into each model, since section 3 requires a query to use
            the same vocabulary and IDF mapping as the corpus it is scored
            against -- so the *query vector itself* moves under a perturbation,
            which is why section 4.3 bounds movement in both arguments.
        ks: Ranks at which to certify stability.

    Returns:
        The :class:`PerturbationReport`.
    """
    before = TfidfVectoriser().fit(list(corpus[1]), list(corpus[0]))
    after = TfidfVectoriser().fit(list(perturbed[1]), list(perturbed[0]))

    shared_ids = sorted(set(before.doc_ids) & set(after.doc_ids))
    return PerturbationReport(
        edit=edit,
        idf_shift=analyse_idf_shift(before, after),
        vector_shifts=tuple(analyse_vector_shift(before, after, doc_id) for doc_id in shared_ids),
        score_before=_score(before, query_features),
        score_after=_score(after, query_features),
        certificates_before=tuple(
            certified_radius(sorted_scores_desc(list(_score(before, query_features).values())), k)
            for k in ks
        ),
    )


def _score(model: TfidfModel, query_features: Sequence[str]) -> dict[str, float]:
    """Score a query against every document, keyed by identifier.

    Keyed by identifier rather than index because an add or remove shifts every
    index after it, and comparing scores by position across a perturbation would
    silently compare different documents.
    """
    query = TfidfVectoriser.transform_query(query_features, model)
    docs = [model.document(i) for i in range(model.n_documents)]
    scores = cosine_against_corpus(query, docs, model.norms)
    return dict(zip(model.doc_ids, scores, strict=True))
