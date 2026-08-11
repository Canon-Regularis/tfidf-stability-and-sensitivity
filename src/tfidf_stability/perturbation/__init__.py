"""Perturbation analysis and the bounds of README section 4."""

from tfidf_stability.perturbation.corpus_edits import (
    Corpus,
    EditKind,
    EditRecord,
    add_document,
    duplicate_document,
    edit_document,
    remove_document,
)
from tfidf_stability.perturbation.experiments import PerturbationReport, run_perturbation
from tfidf_stability.perturbation.idf_perturb import (
    Alignment,
    IdfPerturbation,
    align_models,
    analyse_idf_shift,
)
from tfidf_stability.perturbation.score_bounds import (
    StabilityCertificate,
    certified_radius,
    flip_witness,
    is_order_stable,
    is_top_k_stable,
)
from tfidf_stability.perturbation.vector_perturb import (
    ThreeTermBound,
    VectorPerturbation,
    analyse_vector_shift,
)

__all__ = [
    "Alignment",
    "Corpus",
    "EditKind",
    "EditRecord",
    "IdfPerturbation",
    "PerturbationReport",
    "StabilityCertificate",
    "ThreeTermBound",
    "VectorPerturbation",
    "add_document",
    "align_models",
    "analyse_idf_shift",
    "analyse_vector_shift",
    "certified_radius",
    "duplicate_document",
    "edit_document",
    "flip_witness",
    "is_order_stable",
    "is_top_k_stable",
    "remove_document",
    "run_perturbation",
]
