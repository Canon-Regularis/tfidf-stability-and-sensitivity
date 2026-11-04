"""Byte-deterministic model serialisation: the ``.tfsx`` container.

A fixed-width little-endian header, raw little-endian arrays and UTF-8 string blocks: one
model always gives the same bytes, and a one-ulp change changes them. ``save_model`` writes
an optional JSON sidecar beside it. Rationale: docs/engineering_notes.md#tfsx-container-format
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tfidf_stability.utils.hashing import hash_bytes
from tfidf_stability.utils.io import atomic_write_bytes, write_json
from tfidf_stability.utils.numerics import Reduction
from tfidf_stability.utils.validation import (
    DataIntegrityError,
    TfidfStabilityError,
    check_finite,
    check_non_negative,
    check_unique_ids,
)
from tfidf_stability.vectorisation.idf import IdfVector, LogImpl
from tfidf_stability.vectorisation.sparse import CsrMatrix
from tfidf_stability.vectorisation.tfidf import TfidfModel
from tfidf_stability.vectorisation.vocabulary import Vocabulary

__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "load_model",
    "model_bytes",
    "model_from_bytes",
    "save_model",
]

MAGIC: Final[bytes] = b"TFIDFSTB"
FORMAT_VERSION: Final[int] = 1

_HEADER = struct.Struct("<8sIIIQIIQQ2I")
_FLAG_CORRECTLY_ROUNDED_LOG: Final[int] = 1 << 0
#: Every flag bit this version defines. Bits outside the mask are rejected
#: rather than ignored: ignoring them would let two different files decode to
#: one model, and the reproducibility snapshot rests on the container being a
#: bijection between models and byte strings.
_FLAG_MASK: Final[int] = _FLAG_CORRECTLY_ROUNDED_LOG

#: Wire values for the header's reduction word, written out rather than derived
#: from the enum. Deriving them with `list(Reduction).index(...)` would tie the
#: byte on disk to the member order in `numerics.py`, so inserting or reordering
#: a member would reinterpret existing containers. The enum may be reordered.
_REDUCTION_CODES: Final[dict[Reduction, int]] = {
    Reduction.NAIVE: 0,
    Reduction.NEUMAIER: 1,
    Reduction.PAIRWISE: 2,
    Reduction.EXACT: 3,
}
_REDUCTION_BY_CODE: Final[dict[int, Reduction]] = {v: k for k, v in _REDUCTION_CODES.items()}
_LINE_SEP: Final[bytes] = b"\n"


class TfsxFormatError(DataIntegrityError, ValueError):
    """The file is not a readable ``.tfsx`` container.

    ``DataIntegrityError`` puts container corruption in the package hierarchy,
    so ``except TfidfStabilityError`` around a load catches it. ``ValueError``
    keeps the signature callers already handle.
    """


def _pack(fmt: str, values: Sequence[int] | Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}{fmt}", *values)


def _unpack_ints(fmt: str, data: bytes, offset: int, count: int) -> tuple[tuple[int, ...], int]:
    s = struct.Struct(f"<{count}{fmt}")
    return tuple(int(v) for v in s.unpack_from(data, offset)), offset + s.size


def _unpack_floats(data: bytes, offset: int, count: int) -> tuple[tuple[float, ...], int]:
    s = struct.Struct(f"<{count}d")
    return tuple(float(v) for v in s.unpack_from(data, offset)), offset + s.size


def _encode_strings(items: list[str], what: str) -> bytes:
    """Join strings with LF, raising if any item contains a newline.

    Normalisation strips control characters, so no token holds an LF and the
    encoding is unambiguous.
    """
    for item in items:
        if "\n" in item:
            raise TfsxFormatError(f"{what} contains a newline: {item!r}")
    return _LINE_SEP.join(item.encode("utf-8") for item in items)


def _check_csr(
    indptr: Sequence[int],
    indices: Sequence[int],
    values: Sequence[float],
    *,
    n_rows: int,
    n_cols: int,
) -> None:
    """Raise ``TfsxFormatError`` unless the arrays form a canonical CSR matrix.

    Canonical is what ``CsrMatrix.from_rows`` produces and the scorer, native core and ``row()``
    assume unchecked: ``n_rows + 1`` offsets non-decreasing from 0 to ``nnz``, and in-range
    columns strictly increasing per row, so no dot product double-counts a term.
    """
    if len(indptr) != n_rows + 1:
        raise TfsxFormatError(
            f"indptr has {len(indptr)} entries; expected n_docs + 1 = {n_rows + 1}"
        )
    if len(values) != len(indices):
        raise TfsxFormatError(f"{len(values)} weights for {len(indices)} column indices")
    if indptr[0] != 0:
        raise TfsxFormatError(f"indptr must start at 0, got {indptr[0]}")
    if indptr[-1] != len(indices):
        raise TfsxFormatError(f"indptr ends at {indptr[-1]} but there are {len(indices)} indices")
    for row in range(n_rows):
        lo, hi = indptr[row], indptr[row + 1]
        if hi < lo:
            raise TfsxFormatError(f"indptr decreases at row {row}: {lo} then {hi}")
        segment = indices[lo:hi]
        for position, column in enumerate(segment):
            if not 0 <= column < n_cols:
                raise TfsxFormatError(
                    f"row {row} references column {column}, outside 0..{n_cols - 1}"
                )
            if position and column <= segment[position - 1]:
                raise TfsxFormatError(
                    f"row {row} column indices are not strictly increasing: "
                    f"{segment[position - 1]} then {column}"
                )


def model_bytes(model: TfidfModel) -> bytes:
    """Serialise a model to the ``.tfsx`` byte string.

    The same model always produces the same bytes, with nothing from the
    environment in them.
    """
    vocab = model.vocabulary
    tokens = _encode_strings(list(vocab.tokens), "token")
    doc_ids = _encode_strings(list(model.doc_ids), "document id")

    flags = _FLAG_CORRECTLY_ROUNDED_LOG if model.idf.log_impl is LogImpl.CORRECTLY_ROUNDED else 0
    reduction = _REDUCTION_CODES[Reduction(model.reduction)]

    header = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        model.n_documents,
        model.n_features,
        model.matrix.nnz,
        flags,
        reduction,
        len(tokens),
        len(doc_ids),
        0,
        0,
    )
    return b"".join(
        (
            header,
            _pack("q", model.matrix.indptr),
            _pack("i", model.matrix.indices),
            _pack("d", model.matrix.values),
            _pack("d", model.idf.values),
            _pack("d", model.norms),
            _pack("q", model.lengths),
            _pack("q", vocab.df),
            _pack("q", vocab.cf),
            tokens,
            _LINE_SEP,
            doc_ids,
        )
    )


def save_model(model: TfidfModel, path: Path | str, *, sidecar: bool = True) -> dict[str, str]:
    """Write a model to ``path`` atomically, optionally with a JSON sidecar.

    Returns the provenance dict (container digest and model digest) for
    embedding in a run manifest.
    """
    target = Path(path)
    # For a target already ending in `.json`, `target.with_suffix(".json")` is
    # the target itself. Without this refusal the sidecar overwrites the
    # container, and the manifest's container_sha256 names bytes absent from
    # disk. The check runs before the first write, so a refusal writes nothing.
    if sidecar and target.suffix.lower() == ".json":
        raise TfsxFormatError(
            f"the container path must not end in .json, got {target.name!r}: the sidecar "
            f"is written to the same name and would overwrite the container"
        )
    payload = model_bytes(model)
    atomic_write_bytes(target, payload)

    provenance = {
        "path": target.name,
        "container_sha256": hash_bytes(payload),
        "model_digest": model.digest(),
        "vocabulary_digest": model.vocabulary.digest(),
    }
    if sidecar:
        write_json(
            target.with_suffix(".json"),
            {
                **provenance,
                "format_version": FORMAT_VERSION,
                "n_documents": model.n_documents,
                "n_features": model.n_features,
                "nnz": model.matrix.nnz,
                "reduction": str(model.reduction),
                "log_impl": str(model.idf.log_impl),
                "n_zero_norm_documents": len(model.zero_norm_documents),
            },
        )
    return provenance


@dataclass(frozen=True, slots=True)
class _Header:
    n_docs: int
    n_terms: int
    nnz: int
    flags: int
    reduction: int
    token_bytes: int
    doc_id_bytes: int


def _read_header(data: bytes) -> _Header:
    if len(data) < _HEADER.size:
        raise TfsxFormatError("file is shorter than the header")
    (
        magic,
        version,
        n_docs,
        n_terms,
        nnz,
        flags,
        reduction,
        token_bytes,
        doc_id_bytes,
        reserved_a,
        reserved_b,
    ) = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise TfsxFormatError(f"bad magic {magic!r}; expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        raise TfsxFormatError(
            f"format version {version} is not supported (this build reads {FORMAT_VERSION})"
        )
    if flags & ~_FLAG_MASK:
        raise TfsxFormatError(f"unknown flag bits set: {flags:#010x} (known mask {_FLAG_MASK:#x})")
    if reserved_a or reserved_b:
        raise TfsxFormatError(f"reserved header words must be zero, got {reserved_a}, {reserved_b}")
    if reduction not in _REDUCTION_BY_CODE:
        raise TfsxFormatError(
            f"reduction policy {reduction} is out of range (0..{max(_REDUCTION_BY_CODE)})"
        )
    return _Header(n_docs, n_terms, nnz, flags, reduction, token_bytes, doc_id_bytes)


def _decode_block(block: bytes, count: int, what: str) -> tuple[str, ...]:
    """Decode one LF-separated string block and check it holds ``count`` entries.

    Splitting an empty block yields one empty string, so ``count == 0`` is
    settled before the split. A non-empty block under ``count == 0`` carries
    bytes the format cannot reproduce; rejected rather than dropped.
    """
    try:
        text = block.decode("utf-8")
    except UnicodeDecodeError as e:
        raise TfsxFormatError(f"the {what} block is not valid UTF-8: {e}") from e
    if count == 0:
        if block:
            raise TfsxFormatError(f"{what} count is 0 but the block holds {len(block)} bytes")
        return ()
    items = tuple(text.split("\n"))
    if len(items) != count:
        raise TfsxFormatError(f"expected {count} {what} entries, decoded {len(items)}")
    return items


def load_model(path: Path | str) -> TfidfModel:
    """Read a model back from a ``.tfsx`` container. See :func:`model_from_bytes`."""
    return model_from_bytes(Path(path).read_bytes())


def model_from_bytes(data: bytes) -> TfidfModel:
    """Parse a ``.tfsx`` container, or raise.

    Reads untrusted input; fuzzed by ``tests/test_fuzz_parsers.py``. Lengths are checked against
    the file size before any slice, and accepted bytes re-serialise identically. Rejection raises
    ``TfidfStabilityError``, never ``struct.error``, ``IndexError`` or ``UnicodeDecodeError``.
    """
    # The structural checks live in the parser rather than with the first caller
    # to trip over them. Without them, a CSR matrix whose `indices` point
    # outside the vocabulary raises `IndexError` somewhere unrelated.
    head = _read_header(data)
    offset = _HEADER.size

    expected = (
        offset
        + 8 * (head.n_docs + 1)
        + 4 * head.nnz
        + 8 * head.nnz
        + 8 * head.n_terms
        + 8 * head.n_docs
        + 8 * head.n_docs
        + 8 * head.n_terms
        + 8 * head.n_terms
        + head.token_bytes
        + len(_LINE_SEP)
        + head.doc_id_bytes
    )
    # Equality, since the document-id block is last: a lower bound would accept a
    # truncated file and yield a model whose final identifier had lost its
    # trailing bytes, which passes every count check. Equality also rejects
    # trailing garbage.
    if len(data) != expected:
        raise TfsxFormatError(
            f"file is {len(data)} bytes but the header describes exactly {expected}"
        )

    indptr, offset = _unpack_ints("q", data, offset, head.n_docs + 1)
    indices, offset = _unpack_ints("i", data, offset, head.nnz)
    values, offset = _unpack_floats(data, offset, head.nnz)
    idf_values, offset = _unpack_floats(data, offset, head.n_terms)
    norms, offset = _unpack_floats(data, offset, head.n_docs)
    lengths, offset = _unpack_ints("q", data, offset, head.n_docs)
    df, offset = _unpack_ints("q", data, offset, head.n_terms)
    cf, offset = _unpack_ints("q", data, offset, head.n_terms)

    token_block = data[offset : offset + head.token_bytes]
    offset += head.token_bytes

    # The header fixes both block lengths, so this byte carries no information
    # and accepting any value for it breaks the model-to-bytes bijection.
    # Without the check, all 255 other values decode to a byte-identical model.
    separator = data[offset : offset + len(_LINE_SEP)]
    if separator != _LINE_SEP:
        raise TfsxFormatError(
            f"expected {_LINE_SEP!r} between the token and document-id blocks, got {separator!r}"
        )
    offset += len(_LINE_SEP)
    doc_id_block = data[offset : offset + head.doc_id_bytes]

    # Through the guarded helper rather than `bytes.decode`: one flipped byte
    # makes a block invalid UTF-8, and a bare UnicodeDecodeError cannot be
    # handled alongside the other container failures. A lone 0x80 in the token
    # block is one such byte.
    tokens = _decode_block(token_block, head.n_terms, "token")
    doc_ids = _decode_block(doc_id_block, head.n_docs, "document id")

    # Semantic validation: the checks above fix the container's shape; these confirm that
    # its contents could come from a real fit. CsrMatrix is frozen with no __post_init__,
    # so nothing downstream re-checks the three CSR arrays.
    # Rationale: docs/engineering_notes.md#tfsx-semantic-validation
    _check_csr(indptr, indices, values, n_rows=head.n_docs, n_cols=head.n_terms)

    vocabulary = Vocabulary(
        tokens=tokens,
        df=df,
        cf=cf,
        n_documents=head.n_docs,
        n_discarded=0,  # not part of the identity; recomputable only by refitting
        _index={t: i for i, t in enumerate(tokens)},
    )

    try:
        check_unique_ids(doc_ids)
        # `Vocabulary` is frozen with no `__post_init__`, so the parser runs the `is_sorted`
        # guard `build_vocabulary` runs. Strictly ascending UTF-8 byte order implies
        # uniqueness and also rejects a permuted block, which breaks the order
        # `Vocabulary.digest` relies on. Rationale: docs/engineering_notes.md#tfsx-token-order
        if not vocabulary.is_sorted():
            raise TfidfStabilityError("vocabulary identifiers are not in UTF-8 byte order")
        check_finite(idf_values, "idf")
        check_finite(values, "weights")
        check_finite(norms, "norms")
        # README section 2.1: idf(t) = log((1 + N) / (1 + df(t))) + 1 with df <= N, so
        # idf >= 1; a negative entry comes only from a corrupt file, and scoring and ranking
        # do not check sign. Non-negativity rather than `>= 1` bakes in no smoothing
        # convention. Rationale: docs/engineering_notes.md#tfsx-idf-sign
        check_non_negative(idf_values, "idf")
        check_non_negative(values, "weights")
        check_non_negative(norms, "norms")
        # A real fit gives `lengths[i] >= 0` (zero for an all-stopword document),
        # `1 <= df[t] <= N` (a term enters the vocabulary only by appearing) and
        # `cf[t] >= df[t]` (a term occurs at least once in each document holding it).
        # Rationale: docs/engineering_notes.md#tfsx-integer-arrays
        check_non_negative(lengths, "lengths")
        for doc, length in enumerate(lengths):
            # Each distinct term in row `doc` occurs at least once, so the length is at
            # least the row's non-zero count. A lower bound only: a term occurring once
            # and one occurring three times are one non-zero each, so an inflated length
            # cannot be detected from the container.
            row_nnz = indptr[doc + 1] - indptr[doc]
            if length < row_nnz:
                raise TfidfStabilityError(
                    f"lengths[{doc}] = {length} is below the {row_nnz} distinct terms "
                    f"row {doc} holds; each occurs at least once"
                )
        for term, (d, c) in enumerate(zip(df, cf, strict=True)):
            if not 1 <= d <= head.n_docs:
                raise TfidfStabilityError(
                    f"df[{term}] = {d} is outside 1..{head.n_docs}; a term in the "
                    f"vocabulary appears in at least one document and at most all"
                )
            if c < d:
                raise TfidfStabilityError(
                    f"cf[{term}] = {c} is below df[{term}] = {d}; a term occurs at "
                    f"least once in each document that contains it"
                )
    except TfidfStabilityError as exc:
        raise TfsxFormatError(
            f"container is structurally valid but its contents are not: {exc}"
        ) from exc
    return TfidfModel(
        vocabulary=vocabulary,
        idf=IdfVector(
            values=idf_values,
            n_documents=head.n_docs,
            log_impl=(
                LogImpl.CORRECTLY_ROUNDED
                if head.flags & _FLAG_CORRECTLY_ROUNDED_LOG
                else LogImpl.PLATFORM
            ),
        ),
        matrix=CsrMatrix(
            indptr=indptr,
            indices=indices,
            values=values,
            n_rows=head.n_docs,
            n_cols=head.n_terms,
        ),
        norms=norms,
        lengths=lengths,
        doc_ids=doc_ids,
        reduction=_REDUCTION_BY_CODE[head.reduction],
    )
