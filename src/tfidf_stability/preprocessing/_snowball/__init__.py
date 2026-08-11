"""Vendored Snowball English stemmer (snowballstemmer 3.1.1, BSD-3-Clause).

Why this is vendored rather than hand-written or pip-installed
--------------------------------------------------------------
The Snowball project generates every language binding -- Python, C, Java, Rust --
from a single ``.sbl`` source. Vendoring the *generated* code means the Python
reference backend and the C++ native backend run implementations that agree **by
construction**, not by our diligence in keeping two hand ports in sync. That
matters here more than usual: the architecture of this repository rests on the
two backends being bit-identical, and a preprocessing divergence would surface as
a confusing downstream numerical difference.

A hand-written Porter2 was implemented first and reached 99.845% agreement with
the official test vectors (66 mismatches in 42 649 words), concentrated in the
R1-prefix table, apostrophe stripping, the ``ogist`` rule and the step-1b double
condition. Rather than chase bit-perfection twice -- once in Python and again in
C++ -- the authoritative generated implementation is used. Stemming is a fixed
preprocessing detail, not the object of study; the mathematics under
investigation begins at document frequency.

Provenance
----------
Files, versions and SHA-256 digests are recorded in ``MANIFEST.sha256`` and
verified by ``tests/test_preprocessing_determinism.py``, which also replays the
official 42 649-word ``voc.txt``/``output.txt`` vector pair vendored under
``tests/fixtures/snowball/``.

Upstream: https://snowballstem.org  --  licence: BSD-3-Clause.
"""

from tfidf_stability.preprocessing._snowball.english_stemmer import EnglishStemmer

__all__ = ["EnglishStemmer"]
