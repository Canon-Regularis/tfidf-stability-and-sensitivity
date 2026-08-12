"""Where backend selection actually lives -- which is not here.

This package is a signpost, deliberately empty. Its original docstring described
"the reference, native and numpy evaluators", which promised two things that do
not exist: a registry, and a numpy evaluator. Neither is called for anywhere in
``README.md`` or ``docs/``, and a docstring that advertises unimplemented
machinery is worse than no docstring, so it is corrected here rather than left
to mislead.

What exists instead, and why it is not a registry
-------------------------------------------------
There are exactly **two** backends, and the relationship between them is not
pluggable:

* the pure-Python reference under ``vectorisation/``, ``similarity/`` and
  ``ranking/``, which is **normative** -- it defines correctness;
* the C++20 core, loaded through :mod:`tfidf_stability._native`, which is
  required to be **bit-identical** to it.

A registry implies interchangeable implementations chosen at run time. These two
are not interchangeable in that sense: one is the definition and the other is an
optimisation asserted to agree with it to the last bit. Selection is therefore a
single availability check in :mod:`tfidf_stability._native` -- ``native_available()``
plus an ABI guard -- not a lookup table.

A third, numpy-based evaluator would have to clear the same bar, and every
reduction it performed would have to reproduce the reference's summation order
exactly. numpy's reductions are pairwise with an unspecified block size, so it
could not. That is why one does not exist, and why adding one would be a
research decision rather than an implementation detail.
"""
