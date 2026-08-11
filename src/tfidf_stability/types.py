"""Shared scalar type aliases.

Mirrors ``cpp/include/tfidf/core/types.hpp`` one-for-one, so the claim that the
Python package and the C++ tree have the same shape holds at the level of names
as well as directories. The aliases are documentation rather than enforcement --
Python will not check them at run time -- but they make the *width* the native
side uses visible at every Python call site, which is what matters when a value
is about to cross the boundary.
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

#: A document's stable external identifier -- the final tie-break key. Distinct
#: from :data:`DocIndex`, which is positional and changes if the corpus is
#: reordered.
DocId: TypeAlias = str

#: A 1-indexed position in a ranking, matching the paper's ``r_1 ... r_n``.
#: Ranks are 1-indexed in the public API and 0-indexed in the arrays beneath it;
#: every function that takes one says which it means.
Rank: TypeAlias = int

#: Offset into a sparse structure's flat arrays. ``int64`` in C++, because total
#: non-zeros can exceed 2^31 even when document and term counts cannot.
Offset: TypeAlias = int

#: Every value this project computes is binary64.
Real: TypeAlias = float

#: A similarity score. Distinguished from :data:`Real` for readability only.
Score: TypeAlias = float
