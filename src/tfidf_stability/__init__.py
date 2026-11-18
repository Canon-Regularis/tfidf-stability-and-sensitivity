"""Numerical stability and perturbation behaviour in TF-IDF similarity systems.

The pure-Python reference (README sections 2-4) is normative; the optional compiled backend
in :mod:`tfidf_stability._native` is held to bit-for-bit agreement with it, so there is no
registry and no numpy evaluator. Rationale: docs/engineering_notes.md#no-backend-registry
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
