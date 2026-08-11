"""Stability profiling (A1) and tie-break ablations (A2) (section 7)."""

from tfidf_stability.analysis.stratify import (
    EXACT_TIE_BAND,
    UNDEFINED_BAND,
    Stratum,
    margin_bands,
    stratify_by_margin,
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
    "OperatorPair",
    "Stratum",
    "ablate_queries",
    "ablate_query",
    "disagreement_rate",
    "margin_bands",
    "stratify_by_margin",
]
