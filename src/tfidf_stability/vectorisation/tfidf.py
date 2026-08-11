"""The TF-IDF vectoriser (README section 2.2): ``w_i(t) = tf_i(t) * idf(t)``.

This is the object the rest of the pipeline is built on. It holds the frozen
vocabulary, the IDF vector, the document matrix in CSR form, and the precomputed
row norms, and it exposes all of them -- section 1.2 requires that intermediate
quantities stay inspectable rather than being abstracted away.

Queries are embedded with :meth:`TfidfVectoriser.transform_query`, which reuses
**the same vocabulary and the same IDF mapping** as the corpus, as section 3
requires. IDF is never recomputed for a query and the vocabulary is never
extended; both are asserted rather than assumed (``spec_addenda.md#g12``).
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tfidf_stability.utils.numerics import Reduction
from tfidf_stability.vectorisation.idf import IdfVector, LogImpl, smoothed_idf
from tfidf_stability.vectorisation.sparse import CsrMatrix, SparseVector
from tfidf_stability.vectorisation.tf import term_frequencies
from tfidf_stability.vectorisation.vocabulary import (
    Vocabulary,
    VocabularyConfig,
    build_vocabulary,
)

__all__ = ["TfidfModel", "TfidfVectoriser"]


@dataclass(frozen=True, slots=True)
class TfidfModel:
    """A fitted TF-IDF model, with every intermediate retained.

    Attributes:
        vocabulary: The frozen vocabulary; identifiers are its byte-sorted order.
        idf: IDF per term identifier.
        matrix: Document-term TF-IDF matrix in CSR form, one row per document.
        norms: ``||w_i||_2`` per document, precomputed under ``reduction``.
        lengths: ``L_i``, the exact in-vocabulary token count per document.
        doc_ids: Stable document identifiers, parallel to the matrix rows.
        reduction: The policy every sum in this model was computed under.
    """

    vocabulary: Vocabulary
    idf: IdfVector
    matrix: CsrMatrix
    norms: tuple[float, ...]
    lengths: tuple[int, ...]
    doc_ids: tuple[str, ...]
    reduction: Reduction

    @property
    def n_documents(self) -> int:
        return self.matrix.n_rows

    @property
    def n_features(self) -> int:
        return len(self.vocabulary)

    @property
    def zero_norm_documents(self) -> tuple[int, ...]:
        """Indices of documents that embed to the zero vector.

        Never a silent condition. Section 2.3 gives these a similarity of 0
        against every query, so they cluster at the bottom of every ranking and
        form a large block of *exact* ties -- which is precisely the regime the
        tie-break analysis of section 4.5 is about. Their count is reported in
        every experiment.
        """
        return tuple(i for i, n in enumerate(self.norms) if n == 0.0)

    def document(self, i: int) -> SparseVector:
        """The TF-IDF vector of document ``i``."""
        return self.matrix.row(i)

    def intermediates(self, i: int) -> dict[str, Any]:
        """Every intermediate quantity for one document, keyed by token.

        Exists to satisfy README section 1.2: "retaining access to intermediate
        quantities that are typically abstracted away in higher-level libraries".
        """
        row = self.matrix.row(i)
        return {
            "doc_id": self.doc_ids[i],
            "in_vocabulary_length": self.lengths[i],
            "norm": self.norms[i],
            "terms": [
                {
                    "token": self.vocabulary.token_of(t),
                    "term_id": t,
                    "df": self.vocabulary.df[t],
                    "idf": self.idf[t],
                    # w = tf * idf, so tf is recoverable exactly by division.
                    "tf": w / self.idf[t] if self.idf[t] != 0.0 else 0.0,
                    "weight": w,
                }
                for t, w in zip(row.indices, row.values, strict=True)
            ],
        }

    def digest(self) -> str:
        """SHA-256 over the vocabulary, IDF and every weight, byte-exactly.

        Uses the raw binary64 bit patterns rather than a decimal rendering, so
        the digest changes if any value changes by a single ulp. This is the
        backbone of the reproducibility snapshot test.
        """
        h = hashlib.sha256()
        h.update(self.vocabulary.digest().encode())
        h.update(str(self.idf.log_impl).encode())
        for v in self.idf.values:
            h.update(struct.pack("<d", v))
        for v in self.matrix.values:
            h.update(struct.pack("<d", v))
        for v in self.norms:
            h.update(struct.pack("<d", v))
        h.update(repr(self.matrix.indptr).encode())
        h.update(repr(self.matrix.indices).encode())
        return h.hexdigest()


@dataclass
class TfidfVectoriser:
    """Fits a :class:`TfidfModel` and embeds queries into the same space.

    Args:
        vocabulary_config: ``min_df`` / ``max_df`` / ``max_features`` options.
        log_impl: Which logarithm to use for IDF. The default is the
            correctly-rounded one; see ``spec_addenda.md#g13``.
        reduction: The summation policy for norms. The default is the plain
            left-to-right fold that section 2.3 specifies.
    """

    vocabulary_config: VocabularyConfig = field(default_factory=VocabularyConfig)
    log_impl: LogImpl = LogImpl.CORRECTLY_ROUNDED
    reduction: Reduction = Reduction.NAIVE

    def fit(
        self,
        documents: Sequence[Sequence[str]],
        doc_ids: Sequence[str] | None = None,
    ) -> TfidfModel:
        """Build the vocabulary, IDF and document matrix from feature streams.

        Args:
            documents: One preprocessed feature sequence per document.
            doc_ids: Stable identifiers, defaulting to ``"0"``, ``"1"``, ...
                These are *not* the ranking tie-break identifiers; those live in
                the attribute table.

        Returns:
            The fitted :class:`TfidfModel`.
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]
        elif len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids has length {len(doc_ids)} but there are {len(documents)} documents"
            )

        vocab = build_vocabulary(documents, self.vocabulary_config)
        idf = smoothed_idf(vocab.df, vocab.n_documents, self.log_impl)

        rows: list[SparseVector] = []
        lengths: list[int] = []
        for features in documents:
            tf, length = term_frequencies(features, vocab)
            lengths.append(length)
            # w = tf * idf, one multiply per stored entry. Indices stay ascending,
            # so the row is canonical without re-sorting.
            rows.append(
                SparseVector(
                    indices=tf.indices,
                    values=tuple(v * idf[t] for t, v in zip(tf.indices, tf.values, strict=True)),
                    dim=len(vocab),
                )
            )

        matrix = CsrMatrix.from_rows(rows, len(vocab))
        return TfidfModel(
            vocabulary=vocab,
            idf=idf,
            matrix=matrix,
            norms=matrix.row_norms(self.reduction),
            lengths=tuple(lengths),
            doc_ids=tuple(doc_ids),
            reduction=self.reduction,
        )

    @staticmethod
    def transform_query(features: Sequence[str], model: TfidfModel) -> SparseVector:
        """Embed a query using the model's own vocabulary and IDF.

        Section 3: query vectors are "embedded using the same vocabulary V and
        IDF mapping as the corpus documents, ensuring that all similarity
        computations take place in a common vector space". No IDF is recomputed
        and the vocabulary is not extended -- out-of-vocabulary query terms are
        simply dropped, exactly as for documents.
        """
        tf, _ = term_frequencies(features, model.vocabulary)
        return SparseVector(
            indices=tf.indices,
            values=tuple(v * model.idf[t] for t, v in zip(tf.indices, tf.values, strict=True)),
            dim=len(model.vocabulary),
        )
