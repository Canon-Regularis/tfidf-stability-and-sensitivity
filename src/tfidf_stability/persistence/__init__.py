"""Byte-deterministic model serialisation and run manifests."""

from tfidf_stability.persistence.manifest import RunManifest, environment_block
from tfidf_stability.persistence.model import (
    BLOCK_SEPARATOR,
    HEADER_FIELDS,
    MODEL_FIELDS,
    HeaderField,
    ModelField,
    describe_header,
    describe_schema,
)
from tfidf_stability.persistence.save_load import (
    FORMAT_VERSION,
    MAGIC,
    load_model,
    model_bytes,
    save_model,
)

__all__ = [
    "BLOCK_SEPARATOR",
    "FORMAT_VERSION",
    "HEADER_FIELDS",
    "MAGIC",
    "MODEL_FIELDS",
    "HeaderField",
    "ModelField",
    "RunManifest",
    "describe_header",
    "describe_schema",
    "environment_block",
    "load_model",
    "model_bytes",
    "save_model",
]
