"""Shared scalar type aliases.

Mirrors ``cpp/include/tfidf/core/types.hpp`` one-for-one. Python does not check
these at run time; they record the width the native side uses wherever a value
crosses the boundary.
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
