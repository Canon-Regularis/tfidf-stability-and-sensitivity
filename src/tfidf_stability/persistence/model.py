"""The serialisable model schema.

:mod:`~tfidf_stability.persistence.save_load` owns the byte layout; this module
holds what a saved model must contain and what it means. Split so the layout can
gain a field without touching the schema, and the schema can be printed without
reading struct format strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["MODEL_FIELDS", "ModelField", "describe_schema"]


@dataclass(frozen=True, slots=True)
class ModelField:
    """One array in a saved model."""

    name: str
    dtype: str
    length: str
    purpose: str


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
    """The schema as plain data, for the manifest and for documentation."""
    return [
        {"name": f.name, "dtype": f.dtype, "length": f.length, "purpose": f.purpose}
        for f in MODEL_FIELDS
    ]
