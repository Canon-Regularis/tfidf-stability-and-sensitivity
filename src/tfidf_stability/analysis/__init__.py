"""Section 7 analyses: query grid, noise floor, A1 and A2."""

from tfidf_stability.analysis.noise_floor import (
    NoiseFloor,
    TauBand,
    measure_noise_floor,
    tau_band,
    verify_band_invariance,
)
from tfidf_stability.analysis.query_grid import (
    EvaluatedQuery,
    QueryGrid,
    build_query_grid,
    evaluate,
)
from tfidf_stability.analysis.stability_profile import (
    CertificateAudit,
    TransitionPoint,
    certificate_audit,
    transition_curve,
)
from tfidf_stability.analysis.stratify import (
    EXACT_TIE_BAND,
    UNDEFINED_BAND,
    Stratum,
    margin_bands,
    stratify_by_margin,
)
from tfidf_stability.analysis.summarise import (
    Distribution,
    ExperimentResult,
    percentile,
    summarise_values,
)
from tfidf_stability.analysis.tie_break_ablations import (
    AblationResult,
    OperatorPair,
    ablate_queries,
    ablate_query,
    disagreement_rate,
)

__all__ = [
    "EXACT_TIE_BAND",
    "UNDEFINED_BAND",
    "AblationResult",
    "CertificateAudit",
    "Distribution",
    "EvaluatedQuery",
    "ExperimentResult",
    "NoiseFloor",
    "OperatorPair",
    "QueryGrid",
    "Stratum",
    "TauBand",
    "TransitionPoint",
    "ablate_queries",
    "ablate_query",
    "build_query_grid",
    "certificate_audit",
    "disagreement_rate",
    "evaluate",
    "margin_bands",
    "measure_noise_floor",
    "percentile",
    "stratify_by_margin",
    "summarise_values",
    "tau_band",
    "transition_curve",
    "verify_band_invariance",
]
