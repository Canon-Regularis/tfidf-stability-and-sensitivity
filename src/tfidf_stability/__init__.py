"""Numerical stability and perturbation behaviour in TF-IDF similarity systems.

The pure-Python `reference` implementation in this package is **normative**: it
is a literal transcription of README sections 2-4 and defines what correct means.
The compiled backend in :mod:`tfidf_stability._native` is an optional accelerator
whose agreement with the reference is enforced bit-for-bit by the test suite.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
