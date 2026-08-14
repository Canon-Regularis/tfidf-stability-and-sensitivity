"""Numerical stability and perturbation behaviour in TF-IDF similarity systems.

The pure-Python reference here is normative: a literal transcription of README
sections 2-4. The compiled backend in :mod:`tfidf_stability._native` is an
optional accelerator, held to bit-for-bit agreement with it by the test suite.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
