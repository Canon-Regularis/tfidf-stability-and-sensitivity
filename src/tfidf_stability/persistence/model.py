"""The serialisable model schema.

:mod:`~tfidf_stability.persistence.save_load` owns the byte layout; this module
holds what a saved model must contain and what it means. Split so the layout can
gain a field without touching the schema, and the schema can be printed without
reading struct format strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "BLOCK_SEPARATOR",
    "HEADER_FIELDS",
    "MODEL_FIELDS",
    "HeaderField",
    "ModelField",
    "describe_header",
    "describe_schema",
]


@dataclass(frozen=True, slots=True)
class ModelField:
    """One array in a saved model."""

    name: str
    dtype: str
    length: str
    purpose: str


@dataclass(frozen=True, slots=True)
class HeaderField:
    """One scalar in a saved model's header."""

    name: str
    dtype: str
    purpose: str


#: The header scalars, in file order. ``flags`` bit 0 carries
#: :class:`~tfidf_stability.vectorisation.idf.LogImpl`, G13's cross-platform
#: logarithm switch. ``reduction`` carries the accumulation policy of every sum.
#: Both change the stored weights. The array schema records neither.
HEADER_FIELDS: Final[tuple[HeaderField, ...]] = (
    HeaderField("magic", "8 bytes", "the container's literal magic string"),
    HeaderField("format_version", "uint32", "container layout version"),
    HeaderField("n_docs", "uint32", "document count"),
    HeaderField("n_terms", "uint32", "vocabulary size"),
    HeaderField("nnz", "uint64", "stored non-zeros"),
    HeaderField("flags", "uint32", "bit 0: idf used the correctly-rounded logarithm"),
    HeaderField("reduction", "uint32", "0 naive, 1 neumaier, 2 pairwise, 3 exact"),
    HeaderField("token_bytes", "uint64", "length of the encoded token block"),
    HeaderField("doc_id_bytes", "uint64", "length of the encoded document-id block"),
    HeaderField("reserved_a", "uint32", "must be zero; readers reject anything else"),
    HeaderField("reserved_b", "uint32", "must be zero; readers reject anything else"),
)

#: Always present between the token block and the document-id block. A reader
#: that assumes the two blocks abut misparses every container.
BLOCK_SEPARATOR: Final[bytes] = b"\n"

#: What a `.tfsx` container carries, in file order.
#:
#: `df` and `cf` are recomputable from the corpus but stored anyway: a saved
#: model has to be usable without it, and the vocabulary digest every manifest
#: records is taken over them.
MODEL_FIELDS: Final[tuple[ModelField, ...]] = (
    ModelField("indptr", "int64", "n_docs + 1", "CSR row boundaries"),
    ModelField("indices", "int32", "nnz", "term ids, ascending within a row"),
    ModelField("values", "float64", "nnz", "TF-IDF weights"),
    ModelField("idf", "float64", "n_terms", "smoothed IDF (section 2.1)"),
    ModelField("norms", "float64", "n_docs", "precomputed L2 norms"),
    ModelField("lengths", "int64", "n_docs", "in-vocabulary token counts"),
    ModelField("df", "int64", "n_terms", "document frequency"),
    ModelField("cf", "int64", "n_terms", "collection frequency"),
    ModelField("tokens", "utf-8", "n_terms", "vocabulary, byte-sorted"),
    ModelField("doc_ids", "utf-8", "n_docs", "document identifiers"),
)


def describe_schema() -> list[dict[str, str]]:
    """The array schema as plain data, for the manifest and for documentation.

    Arrays only. :func:`describe_header` carries the header scalars, and a
    reader of a container needs both.
    """
    return [
        {"name": f.name, "dtype": f.dtype, "length": f.length, "purpose": f.purpose}
        for f in MODEL_FIELDS
    ]


def describe_header() -> list[dict[str, str]]:
    """The header scalars as plain data, in file order."""
    return [{"name": f.name, "dtype": f.dtype, "purpose": f.purpose} for f in HEADER_FIELDS]
