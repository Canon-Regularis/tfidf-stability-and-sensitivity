"""BM25 weighting, one rounding after the IDF's (``spec_addenda.md#g30``).

``w(t, d) = idf(t) * f / (f + k1 * (1 - b + b * dl / avgdl))`` with Lucene's IDF and
``docCount``. Scores are plain dot products: norms are 1, so the cosine kernels serve.
Rationale: docs/engineering_notes.md#bm25-exact-weight
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar

from tfidf_stability.utils.numerics import (
    Reduction,
    correctly_rounded_log_ratio,
    platform_log_ratio,
)
from tfidf_stability.vectorisation.idf import LogImpl
from tfidf_stability.vectorisation.sparse import CsrMatrix, SparseVector
from tfidf_stability.vectorisation.tf import in_vocabulary_counts, in_vocabulary_length
from tfidf_stability.vectorisation.vocabulary import (
    Vocabulary,
    VocabularyConfig,
    build_vocabulary,
)

__all__ = [
    "Bm25Model",
    "Bm25Vectoriser",
    "LengthNorm",
    "bm25_idf",
    "bm25_idf_one",
    "bm25_weight",
    "byte4_to_int",
    "encoded_length",
    "int_to_byte4",
]

#: Codes below this are lengths stored as themselves (Lucene ``SmallFloat``).
_NUM_FREE = 24
_MAX_LENGTH = 2**31 - 1


class LengthNorm(str, Enum):
    """How a document's length enters its weight.

    Rationale: docs/engineering_notes.md#bm25-lucene-byte-length
    """

    #: The exact in-vocabulary feature count.
    EXACT = "exact"
    #: Lucene's one-byte encoding, exact to 39 and coarser above.
    LUCENE_BYTE = "lucene_byte"

    def __str__(self) -> str:
        return self.value


def _long_to_int4(i: int) -> int:
    """Four significant bits and a shift, as Lucene's ``SmallFloat.longToInt4``."""
    n_bits = i.bit_length()
    if n_bits < 4:
        return i
    shift = n_bits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(code: int) -> int:
    bits, shift = code & 0x07, (code >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


def int_to_byte4(length: int) -> int:
    """Lucene's ``SmallFloat.intToByte4``: a length to a one-byte code."""
    if not 0 <= length <= _MAX_LENGTH:
        raise ValueError(f"length must lie in [0, 2**31 - 1], got {length}")
    return length if length < _NUM_FREE else _NUM_FREE + _long_to_int4(length - _NUM_FREE)


def byte4_to_int(code: int) -> int:
    """Lucene's ``SmallFloat.byte4ToInt``: the smallest length a code stands for."""
    if not 0 <= code <= 255:
        raise ValueError(f"code must lie in [0, 255], got {code}")
    return code if code < _NUM_FREE else _NUM_FREE + _int4_to_long(code - _NUM_FREE)


def encoded_length(length: int, norm: LengthNorm) -> int:
    """The length a weight sees: ``length`` itself, or its one-byte round trip."""
    if LengthNorm(norm) is LengthNorm.LUCENE_BYTE:
        return byte4_to_int(int_to_byte4(length))
    return length


def bm25_idf_one(df: int, doc_count: int, impl: LogImpl = LogImpl.CORRECTLY_ROUNDED) -> float:
    """``ln(1 + (D - df + 0.5) / (df + 0.5))`` for ``0 <= df <= D``; always positive.

    Evaluated as ``ln((2D + 2) / (2df + 1))``, the same value with integer operands.
    """
    if not 0 <= df <= doc_count:
        raise ValueError(f"need 0 <= df <= doc_count, got df={df}, doc_count={doc_count}")
    if LogImpl(impl) is LogImpl.PLATFORM:
        return platform_log_ratio(2 * doc_count + 2, 2 * df + 1)
    return correctly_rounded_log_ratio(2 * doc_count + 2, 2 * df + 1)


def bm25_idf(
    df: Sequence[int], doc_count: int, impl: LogImpl = LogImpl.CORRECTLY_ROUNDED
) -> tuple[float, ...]:
    """:func:`bm25_idf_one` per term, each distinct df evaluated once."""
    by_df: dict[int, float] = {}
    for d in df:
        if d not in by_df:
            by_df[d] = bm25_idf_one(d, doc_count, impl)
    return tuple(by_df[d] for d in df)


def bm25_weight(
    idf: float,
    count: int,
    length: int,
    *,
    k1: Fraction,
    b: Fraction,
    total_length: int,
    doc_count: int,
) -> float:
    """``idf * f / (f + k1 * (1 - b + b * dl / avgdl))``, rounded once from exact.

    ``avgdl = total_length / doc_count``. Every operand is an integer ratio, so the
    weight is one integer division, which Python rounds correctly.
    """
    # The length may fall below the count: a byte-encoded length rounds down.
    if count < 1 or length < 0 or total_length < 1 or doc_count < 1:
        raise ValueError(
            f"need count >= 1, length >= 0 and positive totals, got count={count}, "
            f"length={length}, total_length={total_length}, doc_count={doc_count}"
        )
    p1, q1 = k1.numerator, k1.denominator
    p2, q2 = b.numerator, b.denominator
    n, d = idf.as_integer_ratio()
    # sat = A / B: f / (f + k1 (1 - b + b dl / avgdl)) scaled by q1 q2 T.
    numerator = count * q1 * q2 * total_length
    denominator = numerator + p1 * ((q2 - p2) * total_length + p2 * doc_count * length)
    return (n * numerator) / (d * denominator)


def _exact_decimal(value: str, name: str) -> Fraction:
    """Parse a decimal string exactly; a float is refused, since ``0.9 != 9/10``."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a decimal string such as '0.9', got {value!r}")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} is not a decimal number: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite, got {value!r}")
    return Fraction(parsed)


@dataclass(frozen=True, slots=True)
class Bm25Model:
    """A fitted BM25 model: weights, raw counts, exact lengths and its parameters.

    ``matrix`` holds the weights and ``counts`` the in-vocabulary counts behind them;
    :attr:`norms` is all 1, so a score is a plain dot product.
    """

    #: The query norm the scoring kernels divide by: 1, so a score is a dot product.
    QUERY_NORM: ClassVar[float] = 1.0

    vocabulary: Vocabulary
    idf: tuple[float, ...]
    matrix: CsrMatrix
    counts: tuple[int, ...]
    lengths: tuple[int, ...]
    doc_ids: tuple[str, ...]
    k1: Fraction
    b: Fraction
    length_norm: LengthNorm
    log_impl: LogImpl
    reduction: Reduction

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_norm", LengthNorm(self.length_norm))
        object.__setattr__(self, "log_impl", LogImpl(self.log_impl))
        object.__setattr__(self, "reduction", Reduction(self.reduction))

    @property
    def n_documents(self) -> int:
        return self.matrix.n_rows

    @property
    def n_features(self) -> int:
        return len(self.vocabulary)

    @property
    def doc_count(self) -> int:
        """``D``: documents with an in-vocabulary feature, Lucene's ``docCount``."""
        return sum(1 for length in self.lengths if length > 0)

    @property
    def total_length(self) -> int:
        return sum(self.lengths)

    @property
    def avgdl(self) -> Fraction:
        return Fraction(self.total_length, self.doc_count)

    @property
    def norms(self) -> tuple[float, ...]:
        return (1.0,) * self.n_documents

    def document(self, i: int) -> SparseVector:
        """The BM25 weight vector of document ``i``."""
        return self.matrix.row(i)

    def intermediates(self, i: int) -> dict[str, Any]:
        """Every quantity behind document ``i``'s weights, keyed by token."""
        row = self.matrix.row(i)
        lo = self.matrix.indptr[i]
        return {
            "doc_id": self.doc_ids[i],
            "in_vocabulary_length": self.lengths[i],
            "encoded_length": encoded_length(self.lengths[i], self.length_norm),
            "terms": [
                {
                    "token": self.vocabulary.token_of(t),
                    "term_id": t,
                    "df": self.vocabulary.df[t],
                    "idf": self.idf[t],
                    "count": self.counts[lo + j],
                    "weight": w,
                }
                for j, (t, w) in enumerate(zip(row.indices, row.values, strict=True))
            ],
        }

    def digest(self) -> str:
        """SHA-256 over the parameters, IDF, weights and lengths, byte-exactly.

        ``k1`` and ``b`` enter as exact ratios, so ``"0.9"`` and ``"0.90"`` agree.
        ``reduction`` enters no weight and is not hashed.
        """
        h = hashlib.sha256(b"bm25\x00")
        h.update(self.vocabulary.digest().encode())
        h.update(f"{self.log_impl}|{self.k1}|{self.b}|{self.length_norm}\n".encode())
        for v in self.idf:
            h.update(struct.pack("<d", v))
        for v in self.matrix.values:
            h.update(struct.pack("<d", v))
        h.update(repr(self.lengths).encode())
        h.update(repr(self.matrix.indptr).encode())
        h.update(repr(self.matrix.indices).encode())
        return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Bm25Vectoriser:
    """Fits a :class:`Bm25Model`; ``k1`` and ``b`` are decimal strings, read exactly.

    The defaults are Anserini's BEIR settings. ``reduction`` is the summation policy of
    the scoring kernels; it enters no weight.
    """

    vocabulary_config: VocabularyConfig = field(default_factory=VocabularyConfig)
    k1: str = "0.9"
    b: str = "0.4"
    length_norm: LengthNorm = LengthNorm.EXACT
    log_impl: LogImpl = LogImpl.CORRECTLY_ROUNDED
    reduction: Reduction = Reduction.NAIVE

    def __post_init__(self) -> None:
        k1, b = _exact_decimal(self.k1, "k1"), _exact_decimal(self.b, "b")
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {self.k1!r}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must lie in [0, 1], got {self.b!r}")
        object.__setattr__(self, "length_norm", LengthNorm(self.length_norm))
        object.__setattr__(self, "log_impl", LogImpl(self.log_impl))
        object.__setattr__(self, "reduction", Reduction(self.reduction))

    def fit(
        self, documents: Sequence[Sequence[str]], doc_ids: Sequence[str] | None = None
    ) -> Bm25Model:
        """Build the vocabulary, IDF and weight matrix from feature streams."""
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]
        elif len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids has length {len(doc_ids)} but there are {len(documents)} documents"
            )
        vocab = build_vocabulary(documents, self.vocabulary_config)
        per_doc = [in_vocabulary_counts(features, vocab) for features in documents]
        lengths = [in_vocabulary_length(counts) for counts in per_doc]
        doc_count = sum(1 for length in lengths if length > 0)
        total = sum(lengths)
        idf = bm25_idf(vocab.df, doc_count, self.log_impl)
        k1, b = _exact_decimal(self.k1, "k1"), _exact_decimal(self.b, "b")

        rows: list[SparseVector] = []
        counts: list[int] = []
        for doc_counts, length in zip(per_doc, lengths, strict=True):
            dl = encoded_length(length, self.length_norm)
            terms = sorted(doc_counts)
            counts.extend(doc_counts[t] for t in terms)
            weights = tuple(
                bm25_weight(
                    idf[t], doc_counts[t], dl, k1=k1, b=b, total_length=total, doc_count=doc_count
                )
                for t in terms
            )
            rows.append(SparseVector(indices=tuple(terms), values=weights, dim=len(vocab)))

        return Bm25Model(
            vocabulary=vocab,
            idf=idf,
            matrix=CsrMatrix.from_rows(rows, len(vocab)),
            counts=tuple(counts),
            lengths=tuple(lengths),
            doc_ids=tuple(doc_ids),
            k1=k1,
            b=b,
            length_norm=self.length_norm,
            log_impl=self.log_impl,
            reduction=self.reduction,
        )

    @staticmethod
    def transform_query(features: Sequence[str], model: Bm25Model) -> SparseVector:
        """Each in-vocabulary query term at its count, so a repeat adds its weight again."""
        counts = in_vocabulary_counts(features, model.vocabulary)
        return SparseVector.from_mapping(
            {t: float(c) for t, c in counts.items()}, len(model.vocabulary)
        )
