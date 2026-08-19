"""Shared scalar type aliases.

These record the width the native side uses wherever a value crosses the
boundary. Python does not check them at run time.

They do **not** mirror ``cpp/include/tfidf/core/types.hpp`` one-for-one, and an
earlier version of this docstring said they did. The correspondence is:

===============  ==========================  ====================================
Python           C++                         note
===============  ==========================  ====================================
``TermId``       ``TermId``                  same concept, ``int32`` there
``DocIndex``     ``DocId``                   **the names differ**: a row index
``DocId``        no counterpart              the external identifier, a ``str``
``Rank``         no counterpart              1-indexed, Python-side reporting
``Offset``       ``Offset``                  ``int64`` there
``Real``         ``Real``                    ``double``
``Score``        ``Score``                   ``double``
===============  ==========================  ====================================

The ``DocId`` row is the trap. C++ ``DocId`` is a positional row index, which is
what this module calls ``DocIndex``; Python ``DocId`` is the stable external
string that survives a corpus reordering and terminates every sort key. The same
name denotes a different concept on each side, so a signature read across the
boundary means the opposite of what it appears to.

``tests/test_public_api_surface.py`` parses the header and asserts this table, so
the correspondence cannot drift again without a test failing.
"""

from __future__ import annotations

from typing import TypeAlias

__all__ = [
    "DocId",
    "DocIndex",
    "Offset",
    "Rank",
    "Real",
    "Score",
    "TermId",
]

#: Vocabulary identifier. ``int32`` in C++; assigned by UTF-8 byte order at
#: vocabulary freeze time, so it is a pure function of the token set.
TermId: TypeAlias = int

#: A document's row index in the corpus matrix. ``int32`` in C++.
DocIndex: TypeAlias = int

#: A document's stable external identifier, and the final tie-break key.
#: :data:`DocIndex` is positional and moves when the corpus is reordered.
DocId: TypeAlias = str

#: A 1-indexed position in a ranking, matching the paper's ``r_1 ... r_n``. The
#: arrays beneath the public API are 0-indexed; every function taking a rank
#: says which convention it means.
Rank: TypeAlias = int

#: Offset into a sparse structure's flat arrays. ``int64`` in C++, because total
#: non-zeros can exceed 2^31 even when document and term counts cannot.
Offset: TypeAlias = int

#: Every value this project computes is binary64.
Real: TypeAlias = float

#: A similarity score. Distinguished from :data:`Real` for readability only.
Score: TypeAlias = float
