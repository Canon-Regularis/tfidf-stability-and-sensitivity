"""Numerical stability and perturbation behaviour in TF-IDF similarity systems.

The pure-Python reference here is normative: a literal transcription of README
sections 2-4. The compiled backend in :mod:`tfidf_stability._native` is an
optional accelerator, held to bit-for-bit agreement with it by the test suite.

Those are the two backends, and there is no registry. A registry implies
interchangeable implementations chosen at run time; here one side defines the
answer and the other is asserted to agree to the last bit, so selection is one
availability check (``native_available()`` plus an ABI guard). A numpy
evaluator would have to reproduce the reference's summation order, and numpy's
reductions are pairwise with an unspecified block size, so adding one would be
a research decision rather than an implementation.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
