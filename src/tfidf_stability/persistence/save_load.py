"""Byte-deterministic model serialisation: the ``.tfsx`` container.

Why a custom format
-------------------
The obvious choices break the reproducibility snapshot:

* ``numpy.savez`` writes a zip, and a zip embeds a per-member modification
  timestamp, so two saves of the same model differ in bytes.
* ``pickle`` is version-sensitive, insecure to load, and its output tracks
  Python's object layout rather than the data.
* JSON cannot carry a ``double`` without losing bits or bloating the file with
  17-digit decimals whose parse-back is a separate risk.

``.tfsx`` is a plain uncompressed container: fixed-width header, raw
little-endian arrays, UTF-8 token block. Saving the same model twice gives
identical bytes and a one-ulp change anywhere changes the file, which is what
``test_reproducibility_snapshot.py`` needs.

Layout
------
All integers little-endian. The header is fixed width so it can be read without
parsing::

    magic          8   b"TFIDFSTB"
    format_version u32
    n_docs         u32
    n_terms        u32
    nnz            u64
    flags          u32   bit 0: idf computed with a correctly-rounded logarithm
    reduction      u32   the Reduction policy every sum was taken under
    token_bytes    u64   length of the token block
    doc_id_bytes   u64   length of the document-id block
    reserved       u32 x 2

    indptr    i64 x (n_docs + 1)
    indices   i32 x nnz
    values    f64 x nnz
    idf       f64 x n_terms
    norms     f64 x n_docs
    lengths   i64 x n_docs
    df        i64 x n_terms
    cf        i64 x n_terms
    tokens    utf-8, LF-separated, n_terms entries
    doc_ids   utf-8, LF-separated, n_docs entries

A JSON sidecar carries the same metadata in readable form, written alongside and
never inside, so the container stays byte-stable whatever the sidecar gains.
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
_LINE_SEP: Final[bytes] = b"\n"


class TfsxFormatError(DataIntegrityError, ValueError):
    """The file is not a readable ``.tfsx`` container.

    Both bases carry weight. ``DataIntegrityError`` puts container corruption in
    the package hierarchy, so ``except TfidfStabilityError`` around a load
    catches it and the fuzz harness can state its property as "nothing but a
    package exception ever escapes". ``ValueError`` is kept for callers written
    against the original signature.
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
    """Join with LF. Normalisation strips control characters, so no token holds an
    LF and the encoding is unambiguous."""
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
    """Reject index arrays that do not describe a canonical CSR matrix.

    Canonical means what ``build`` produces and what the reference scorer, the
    native core and ``row()`` assume without re-checking: one more offset than
    rows, starting at zero, non-decreasing, ending at ``nnz``, and column indices
    in range and strictly increasing within a row. Strict increase makes a row's
    support a set; a duplicate column double-counts a term in every dot product.
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
    reduction = list(Reduction).index(model.reduction)

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
    if not 0 <= reduction < len(Reduction):
        raise TfsxFormatError(
            f"reduction policy {reduction} is out of range (0..{len(Reduction) - 1})"
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

    Every length comes from the header and is checked against the file size
    before any slice is taken, so a truncated or corrupt file raises rather than
    yielding a model built from adjacent bytes. This is the only parser here that
    reads untrusted input, and the fuzz target of ``tests/test_fuzz_parsers.py``,
    which holds it to two properties:

    * anything it accepts re-serialises to the identical bytes, so the container
      is a bijection and no two files decode to one model;
    * anything it rejects raises a
      :class:`~tfidf_stability.utils.validation.TfidfStabilityError`, never a
      ``struct.error``, ``IndexError`` or ``UnicodeDecodeError`` from the
      internals.

    The second property is why the structural checks below live here rather than
    with the first caller to trip over them. A CSR matrix whose ``indices`` point
    outside the vocabulary raises ``IndexError`` somewhere unrelated, hours
    later.
    """
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
    # Before the check existed, all 255 other values decoded to a byte-identical
    # model.
    separator = data[offset : offset + len(_LINE_SEP)]
    if separator != _LINE_SEP:
        raise TfsxFormatError(
            f"expected {_LINE_SEP!r} between the token and document-id blocks, got {separator!r}"
        )
    offset += len(_LINE_SEP)
    doc_id_block = data[offset : offset + head.doc_id_bytes]

    # Through the guarded helper rather than `bytes.decode`: one flipped byte
    # makes a block invalid UTF-8, and a bare UnicodeDecodeError cannot be
    # handled alongside the other container failures. Fuzzing found it with a
    # single 0x80 in the token block.
    tokens = _decode_block(token_block, head.n_terms, "token")
    doc_ids = _decode_block(doc_id_block, head.n_docs, "document id")

    # Semantic validation: the checks above confirm the container is shaped
    # right, these that its contents could have come from a real fit. A file can
    # satisfy every count and still carry NaN weights (undefined behaviour for
    # the sort) or repeated ids (which destroy the ranking operator's strict
    # total order), both of which produce a plausible-looking ranking rather than
    # an error. The three CSR arrays are read independently with only their
    # lengths fixed by the header: of 2715 mutated containers this parser
    # accepted, 425 carried an indptr that was non-monotonic or did not start at
    # zero. Nothing downstream re-checks (CsrMatrix is frozen with no
    # __post_init__), so such a model was returned intact and misbehaved far from
    # its cause.
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
        # The token block gets the same treatment as the document-id block one
        # line up, and for the same reason. It did not: `check_unique_ids` was
        # applied to `doc_ids` alone, and `Vocabulary` is frozen with no
        # `__post_init__`, so constructing it here bypassed the `is_sorted`
        # guard that `build_vocabulary` runs -- the one vocabulary.py:287 calls
        # "the one check standing between a mis-sorted vocabulary and silently
        # wrong weights".
        #
        # Measured on a 301-byte container over the vocabulary ('aa','ab','cc'):
        # flipping byte 288 from 'b' to 'a' gives ('aa','aa','cc'), which this
        # parser accepted. `_index` then maps 'aa' to column 1, so the query
        # ['aa'] scored the document containing "aa" twice at 0.0 and a document
        # containing no "aa" at all at 0.707. Flipping the equivalent byte in
        # the document-id block was already rejected, which is the asymmetry
        # this closes -- and `load_model` is, by its own docstring, the only
        # parser here that reads untrusted input.
        #
        # Ascent rather than mere uniqueness: it subsumes uniqueness and also
        # catches a permuted block, which passes a uniqueness test while making
        # the binary searches in tf.py and the merge in align_models wrong.
        if not vocabulary.is_sorted():
            raise TfidfStabilityError("vocabulary identifiers are not in UTF-8 byte order")
        check_finite(idf_values, "idf")
        check_finite(values, "weights")
        check_finite(norms, "norms")
        check_non_negative(values, "weights")
        check_non_negative(norms, "norms")
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
        reduction=list(Reduction)[head.reduction],
    )
