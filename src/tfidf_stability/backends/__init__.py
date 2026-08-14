"""Signpost: backend selection lives in :mod:`tfidf_stability._native`.

This package is empty. Its original docstring described "the reference, native
and numpy evaluators", promising a registry and a numpy evaluator; neither
exists, and neither is called for in ``README.md`` or ``docs/``.

There are two backends: the pure-Python reference under ``vectorisation/``,
``similarity/`` and ``ranking/``, which fixes correctness, and the C++20 core
loaded through :mod:`tfidf_stability._native`, required to be bit-identical to
it. A registry implies interchangeable implementations chosen at run time; here
one side defines the answer and the other is asserted to agree to the last bit,
so selection is one availability check (``native_available()`` plus an ABI
guard).

A numpy evaluator would have to reproduce the reference's summation order.
numpy's reductions are pairwise with an unspecified block size, so it could not;
adding one would be a research decision.
"""
