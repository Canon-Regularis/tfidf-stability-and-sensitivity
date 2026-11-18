"""Model serialisation: round-trip fidelity and byte determinism.

A loaded model must be bit-identical to the saved one, and two saves of one
model must give identical bytes; the container is hand-rolled because a zip
embeds a per-member timestamp. Rationale: docs/engineering_notes.md#tfsx-container
"""

from __future__ import annotations

import dataclasses
import struct

import pytest

from tfidf_stability.persistence.model import (
    BLOCK_SEPARATOR,
    HEADER_FIELDS,
    MODEL_FIELDS,
    describe_header,
    describe_schema,
)
from tfidf_stability.persistence.save_load import (
    _FLAG_MASK,
    _HEADER,
    _REDUCTION_BY_CODE,
    _REDUCTION_CODES,
    FORMAT_VERSION,
    MAGIC,
    _check_csr,
    _decode_block,
    load_model,
    model_bytes,
    model_from_bytes,
    save_model,
)
from tfidf_stability.utils.hashing import hash_bytes, hash_floats
from tfidf_stability.utils.numerics import Reduction, same_bits
from tfidf_stability.vectorisation.idf import LogImpl
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

TfsxFormatError = pytest.importorskip("tfidf_stability.persistence.save_load").TfsxFormatError

#: Byte offset of the flags word, derived from the header layout rather than
#: written down, so a field added ahead of it moves this with it.
_FLAGS_OFFSET = struct.calcsize("<8sIIIQ")


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
def test_round_trip_is_bit_identical(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Identical rather than close: downstream comparisons are bitwise."""
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path)
    restored = load_model(path)

    assert restored.doc_ids == mini_model.doc_ids
    assert restored.vocabulary.tokens == mini_model.vocabulary.tokens
    assert restored.vocabulary.df == mini_model.vocabulary.df
    assert restored.vocabulary.cf == mini_model.vocabulary.cf
    assert restored.matrix.indptr == mini_model.matrix.indptr
    assert restored.matrix.indices == mini_model.matrix.indices
    assert restored.lengths == mini_model.lengths
    assert restored.reduction is mini_model.reduction
    assert restored.idf.log_impl is mini_model.idf.log_impl

    for a, b in zip(mini_model.matrix.values, restored.matrix.values, strict=True):
        assert same_bits(a, b)
    for a, b in zip(mini_model.idf.values, restored.idf.values, strict=True):
        assert same_bits(a, b)
    for a, b in zip(mini_model.norms, restored.norms, strict=True):
        assert same_bits(a, b)


def test_the_model_digest_survives_a_round_trip(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The digest, which covers every weight bitwise, survives a round trip."""
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path)
    assert load_model(path).digest() == mini_model.digest()
    assert load_model(path).vocabulary.digest() == mini_model.vocabulary.digest()


@pytest.mark.parametrize("policy", list(Reduction))
def test_the_reduction_policy_round_trips(mini_features, policy, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The reduction policy round-trips as part of the model's identity.

    The same corpus under a different policy has different norms.
    """
    model = TfidfVectoriser(reduction=policy).fit(list(mini_features))
    path = tmp_path / f"{policy.value}.tfsx"
    save_model(model, path)
    assert load_model(path).reduction is policy


@pytest.mark.parametrize("impl", list(LogImpl))
def test_the_log_implementation_round_trips(mini_features, impl, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The log implementation round-trips as part of the provenance.

    G13: the two logarithms differ in about 15% of IDF entries.
    """
    model = TfidfVectoriser(log_impl=impl).fit(list(mini_features))
    path = tmp_path / f"{impl.value}.tfsx"
    save_model(model, path)
    assert load_model(path).idf.log_impl is impl


def test_a_zero_norm_document_round_trips(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A zero-norm document round-trips as an empty CSR row.

    The mini corpus has one (all-stopword d5); an empty row is the case an
    off-by-one in the row boundaries corrupts.
    """
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path)
    restored = load_model(path)
    assert restored.zero_norm_documents == mini_model.zero_norm_documents
    assert restored.matrix.row(4).nnz == 0


# ---------------------------------------------------------------------------
# Byte determinism
# ---------------------------------------------------------------------------
def test_saving_twice_gives_identical_bytes(mini_model) -> None:  # type: ignore[no-untyped-def]
    """Two saves of one model give identical bytes.

    A zip stores a modification time per member, so ``numpy.savez`` cannot
    provide this.
    """
    assert model_bytes(mini_model) == model_bytes(mini_model)


def test_the_container_carries_no_ambient_state(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Saving to two different paths must give the same bytes.

    Catches a path, a hostname or a timestamp leaking into the container.
    """
    a, b = tmp_path / "one.tfsx", tmp_path / "nested" / "two.tfsx"
    save_model(mini_model, a, sidecar=False)
    save_model(mini_model, b, sidecar=False)
    assert a.read_bytes() == b.read_bytes()


def test_a_one_ulp_change_changes_the_file(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The container must be sensitive at the ulp, the scale under study."""
    import dataclasses

    original = model_bytes(mini_model)
    nudged_norms = list(mini_model.norms)
    nudged_norms[0] = struct.unpack("<d", struct.pack("<d", nudged_norms[0]))[0]
    nudged_norms[0] = nudged_norms[0] + __import__("math").ulp(nudged_norms[0])
    nudged = dataclasses.replace(mini_model, norms=tuple(nudged_norms))

    assert model_bytes(nudged) != original
    assert hash_floats(nudged.norms) != hash_floats(mini_model.norms)


def test_the_sidecar_does_not_affect_the_container(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with_sidecar = tmp_path / "a.tfsx"
    without = tmp_path / "b.tfsx"
    save_model(mini_model, with_sidecar, sidecar=True)
    save_model(mini_model, without, sidecar=False)
    assert with_sidecar.read_bytes() == without.read_bytes()
    assert with_sidecar.with_suffix(".json").exists()
    assert not without.with_suffix(".json").exists()


def test_save_reports_provenance(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "model.tfsx"
    provenance = save_model(mini_model, path)
    assert provenance["container_sha256"] == hash_bytes(path.read_bytes())
    assert provenance["model_digest"] == mini_model.digest()


# ---------------------------------------------------------------------------
# Rejecting corrupt input
# ---------------------------------------------------------------------------
def test_a_truncated_file_is_rejected(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A truncated file raises rather than yielding a model.

    This parser is the only reader of untrusted bytes. Every length comes from
    the header and is checked against the file size before any slice is taken.
    """
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path, sidecar=False)
    payload = path.read_bytes()
    for cut in (0, 8, 16, len(payload) // 2, len(payload) - 1):
        path.write_bytes(payload[:cut])
        with pytest.raises((TfsxFormatError, struct.error, UnicodeDecodeError)):
            load_model(path)


def test_bad_magic_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.tfsx"
    path.write_bytes(b"NOTATFSX" + bytes(64))
    with pytest.raises(TfsxFormatError, match="bad magic"):
        load_model(path)


def test_an_unsupported_format_version_is_rejected(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A future format version fails loudly rather than being misread."""
    path = tmp_path / "future.tfsx"
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, len(MAGIC), FORMAT_VERSION + 99)
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="format version"):
        load_model(path)


def test_a_newline_in_a_token_is_refused(mini_features) -> None:  # type: ignore[no-untyped-def]
    """A token containing LF is refused: the token block is LF-separated.

    Normalisation strips control characters, so real text cannot reach this;
    the format refuses it without relying on that.
    """
    model = TfidfVectoriser().fit([["ok"], ["also\nbad"]])
    with pytest.raises(TfsxFormatError, match="newline"):
        model_bytes(model)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------
def test_writes_are_atomic_and_leave_no_temporary_files(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_model(mini_model, tmp_path / "model.tfsx")
    assert not list(tmp_path.glob("*.tmp"))


def test_overwriting_replaces_cleanly(mini_model, mini_features, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path, sidecar=False)
    other = TfidfVectoriser(reduction=Reduction.EXACT).fit(list(mini_features))
    save_model(other, path, sidecar=False)
    assert load_model(path).reduction is Reduction.EXACT


def test_an_unknown_flag_bit_is_refused_rather_than_ignored(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Only one flag is defined: it selects the correctly-rounded logarithm.

    A future writer that sets a second bit describes a model this build cannot
    reconstruct; masking the bit off would load a plausible model computed
    under the wrong assumption.
    """
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, _FLAGS_OFFSET, _FLAG_MASK | (1 << 17))
    path = tmp_path / "flagged.tfsx"
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="unknown flag bits"):
        load_model(path)


# ---------------------------------------------------------------------------
# The two guards the container format cannot reach
# ---------------------------------------------------------------------------


# `_check_csr`'s length arms and `_decode_block`'s empty-block arms cannot be
# provoked by a corrupt file: the header fixes every count, and a file whose
# length disagrees with it is rejected first. `row()`, the reference scorer and
# the native core rely on them without re-checking, so the tests call them.
def test_an_indptr_of_the_wrong_length_is_rejected() -> None:
    """A short indptr is rejected.

    One offset per row plus a final one; a short indptr makes the last row's
    extent depend on whatever the array ends with.
    """
    with pytest.raises(TfsxFormatError, match=r"indptr has 2 entries; expected n_docs \+ 1 = 3"):
        _check_csr([0, 1], [0], [1.0], n_rows=2, n_cols=4)


def test_a_weight_for_every_column_index_is_required() -> None:
    """Every column index needs a weight.

    The two arrays are read independently with only their lengths fixed, so a
    mismatch pairs a weight with the wrong column in every dot product.
    """
    with pytest.raises(TfsxFormatError, match="1 weights for 2 column indices"):
        _check_csr([0, 2], [0, 1], [1.0], n_rows=1, n_cols=4)


def test_a_canonical_matrix_passes_every_arm() -> None:
    """A canonical matrix passes every arm of ``_check_csr``.

    This is the positive control for the rejection cases above.
    """
    _check_csr([0, 2, 3], [0, 2, 1], [1.0, 2.0, 3.0], n_rows=2, n_cols=4)


def test_an_empty_block_decodes_to_no_entries_rather_than_one_empty_string() -> None:
    """An empty block decodes to no entries.

    Splitting an empty block yields one empty string, so a zero count is
    settled before the split; otherwise an empty model gains a phantom token.
    """
    assert _decode_block(b"", 0, "token") == ()


def test_a_non_empty_block_under_a_zero_count_is_rejected_rather_than_dropped() -> None:
    """A non-empty block under a zero count is rejected.

    Those bytes cannot be reproduced on the way back out, so discarding them
    would break the model-to-bytes bijection.
    """
    with pytest.raises(TfsxFormatError, match="count is 0 but the block holds"):
        _decode_block(b"orphan", 0, "token")


def test_a_block_holding_the_wrong_number_of_entries_is_rejected() -> None:
    """A block with the wrong number of entries is rejected.

    The header states the count and the block carries the bytes; the two are
    written independently, so a file can pass every length check and still
    decode to the wrong number of tokens.
    """
    with pytest.raises(TfsxFormatError, match="expected 3 token entries, decoded 2"):
        _decode_block(b"a\nb", 3, "token")


# ---------------------------------------------------------------------------
# The header, field by field
# ---------------------------------------------------------------------------

# A .tfsx file is read by builds other than the one that wrote it, so the byte
# layout and the exact rejection boundaries are the contract: `column < n_cols`
# rather than `<=`, both reserved words checked, and the correctly-rounded flag
# at a fixed bit.
_RESERVED_A, _RESERVED_B = 52, 56
_REDUCTION_OFFSET = 32


def test_the_correctly_rounded_flag_is_bit_zero(mini_model) -> None:  # type: ignore[no-untyped-def]
    """The correctly-rounded flag is bit zero of the flags word.

    The bit position is on-disk format. A build that moved it would write files
    this one reads as "platform logarithm": a model that differs from the
    normative one in about 15% of its idf entries, loaded without complaint.
    """
    payload = model_bytes(mini_model)
    (flags,) = struct.unpack_from("<I", payload, _FLAGS_OFFSET)
    assert flags == 1
    assert _FLAG_MASK == 1


def test_a_file_of_exactly_header_length_is_measured_not_dismissed(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A file of exactly header length fails on the missing body.

    The guard is `len(data) < header_size`. At exactly the header size the
    header is complete, so the failure names the missing body rather than the
    header.
    """
    path = tmp_path / "headeronly.tfsx"
    path.write_bytes(model_bytes(mini_model)[:60])
    with pytest.raises(TfsxFormatError, match="describes exactly"):
        load_model(path)

    path.write_bytes(model_bytes(mini_model)[:59])
    with pytest.raises(TfsxFormatError, match="shorter than the header"):
        load_model(path)


@pytest.mark.parametrize(("offset", "which"), [(_RESERVED_A, "first"), (_RESERVED_B, "second")])
def test_either_reserved_word_being_set_is_enough_to_refuse_the_file(
    mini_model,  # type: ignore[no-untyped-def]
    tmp_path,  # type: ignore[no-untyped-def]
    offset: int,
    which: str,
) -> None:
    """Both reserved words are checked, not just one.

    A reserved word carries meaning a future version defines, so a file that
    sets either one describes something this build cannot honour.
    """
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, offset, 0xABCD)
    path = tmp_path / f"reserved_{which}.tfsx"
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="reserved header words must be zero"):
        load_model(path)


def test_the_reduction_policy_is_rejected_one_past_the_last_one(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The reduction code is rejected one past the last valid value.

    `0 <= reduction < len(Reduction)`; the count itself is the value an
    off-by-one in a future writer produces.
    """
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, _REDUCTION_OFFSET, len(Reduction))
    path = tmp_path / "reduction.tfsx"
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="out of range"):
        load_model(path)

    # The last valid value is accepted.
    struct.pack_into("<I", payload, _REDUCTION_OFFSET, len(Reduction) - 1)
    path.write_bytes(bytes(payload))
    assert load_model(path).matrix.n_rows == mini_model.matrix.n_rows


def test_a_column_index_equal_to_the_vocabulary_size_is_out_of_range() -> None:
    """A column index equal to `n_cols` is out of range.

    Columns are 0-based, so `n_cols` is the first invalid one; it is the value
    an off-by-one produces, and it would index one past the idf array.
    """
    with pytest.raises(TfsxFormatError, match=r"outside 0\.\.3"):
        _check_csr([0, 1], [4], [1.0], n_rows=1, n_cols=4)
    # One below is fine.
    _check_csr([0, 1], [3], [1.0], n_rows=1, n_cols=4)


def test_a_row_that_repeats_a_column_is_not_strictly_increasing() -> None:
    """A row that repeats a column is rejected as not strictly increasing.

    Strict increase makes a row's support a set; a duplicate double-counts that
    term in every dot product. Equal neighbours are the case `<` in place of
    `<=` lets through.
    """
    with pytest.raises(TfsxFormatError, match="not strictly increasing: 1 then 1"):
        _check_csr([0, 2], [1, 1], [1.0, 2.0], n_rows=1, n_cols=4)


def test_the_sidecar_is_written_unless_it_is_declined(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The sidecar is written by default and omitted only when declined.

    The default carries the digests a manifest quotes, so dropping it leaves a
    model whose provenance has to be recomputed to be checked.
    """
    with_sidecar = tmp_path / "a.tfsx"
    save_model(mini_model, with_sidecar)
    assert with_sidecar.with_suffix(".json").is_file()

    without = tmp_path / "b.tfsx"
    save_model(mini_model, without, sidecar=False)
    assert not without.with_suffix(".json").exists()


def test_a_loaded_model_reports_no_discarded_terms(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A loaded model reports zero discarded terms.

    `n_discarded` counts what `max_features` dropped at fit time. It is not in
    the file because it is not part of the model's identity, so a load reports
    zero rather than a count that would travel into a manifest.
    """
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path)
    assert load_model(path).vocabulary.n_discarded == 0


def test_the_format_version_this_build_writes_is_one(mini_model) -> None:  # type: ignore[no-untyped-def]
    """The format version this build writes is one.

    It is stamped into every file and compared on every read. Asserted both as
    the constant and as the bytes written, since only the second is what
    another build sees.
    """
    assert FORMAT_VERSION == 1
    (written,) = struct.unpack_from("<I", model_bytes(mini_model), len(MAGIC))
    assert written == FORMAT_VERSION


def test_a_rejected_indptr_start_and_end_name_the_value_they_found() -> None:
    """A rejected indptr start or end names the value found.

    A format error is read by someone holding a file they cannot open; one
    that quotes the wrong number points to the wrong place.
    """
    with pytest.raises(TfsxFormatError, match="indptr must start at 0, got 1"):
        _check_csr([1, 2], [0, 1], [1.0, 2.0], n_rows=1, n_cols=4)
    with pytest.raises(TfsxFormatError, match="indptr ends at 9 but there are 2 indices"):
        _check_csr([0, 9], [0, 1], [1.0, 2.0], n_rows=1, n_cols=4)


def test_an_out_of_range_reduction_names_the_range_it_checked(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An out-of-range reduction names the range checked.

    `0..len(Reduction) - 1` is computed from the enum rather than written down,
    so the message stays right as policies are added.
    """
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, _REDUCTION_OFFSET, 99)
    path = tmp_path / "reduction99.tfsx"
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match=r"reduction policy 99 is out of range \(0\.\.3\)"):
        load_model(path)


def test_a_descending_pair_is_reported_in_the_order_it_was_found() -> None:
    """A descending pair is reported in the order found.

    The duplicate case cannot tell the two operands apart: "1 then 1" reads the
    same whichever index is quoted. A descending pair can, and a reversed
    diagnostic points at the wrong end of the file.
    """
    with pytest.raises(TfsxFormatError, match="not strictly increasing: 3 then 1"):
        _check_csr([0, 2], [3, 1], [1.0, 2.0], n_rows=1, n_cols=4)


# ---------------------------------------------------------------------------
# Erroneous: the token block gets the same scrutiny as the document-id block
# ---------------------------------------------------------------------------


# `load_model` checks `doc_ids` for uniqueness and `tokens` for strict UTF-8
# byte order. `Vocabulary` is frozen with no `__post_init__`, so a vocabulary
# built directly skips the `is_sorted` check that `build_vocabulary` runs.
# Rationale: docs/engineering_notes.md#tfsx-field-validation
def _three_token_model() -> object:
    """A model whose vocabulary is short enough to locate inside the container.

    Local by house convention. Two-character tokens make the token block
    ``aa|ab|cc`` (LF-separated), so a single byte identifies a single token
    unambiguously.
    """
    return TfidfVectoriser().fit(
        [["aa", "aa", "cc"], ["ab", "cc"], ["aa", "ab"]], ["d0", "d1", "d2"]
    )


def _flip_to_duplicate_a_token(payload: bytes) -> tuple[bytes, int]:
    """Turn the token ``ab`` into a second ``aa``, returning the byte changed.

    Located by searching rather than by a hard-coded offset: the offset moves
    with any header change, and a stale one would silently corrupt some other
    field and test nothing.
    """
    block = b"aa\nab\ncc"
    at = payload.index(block) + len(b"aa\na")
    mutated = bytearray(payload)
    mutated[at] = ord("a")
    return bytes(mutated), at


def test_a_vocabulary_that_repeats_a_token_is_refused_rather_than_loaded() -> None:
    """A vocabulary that repeats a token is refused rather than loaded.

    Flipping the ``b`` of ``ab`` gives the token block ``aa aa cc``. ``_index``
    is built by enumeration, so the later ``aa`` would take column 1, and the
    mutated container re-serialises to itself, so a round trip cannot detect it.
    """
    payload = model_bytes(_three_token_model())
    mutated, at = _flip_to_duplicate_a_token(payload)

    assert mutated != payload, f"the flip at byte {at} changed nothing"
    assert len(mutated) == len(payload), "one byte overwritten, not inserted"
    with pytest.raises(TfsxFormatError, match="not in UTF-8 byte order"):
        model_from_bytes(mutated)


def test_a_vocabulary_merely_out_of_order_is_refused_too_not_only_a_repeated_one() -> None:
    """A vocabulary merely out of order is refused, not only a repeated one.

    A permuted block passes any uniqueness test but breaks the ascending byte
    order that ``Vocabulary.digest`` and the determinism guarantee rest on, so
    the guard checks order rather than ``check_unique_ids``.
    """
    payload = model_bytes(_three_token_model())
    mutated = payload.replace(b"aa\nab\ncc", b"ab\naa\ncc", 1)

    assert mutated != payload, "the permutation changed nothing"
    assert len(mutated) == len(payload), "a permutation, not a resize"
    with pytest.raises(TfsxFormatError, match="not in UTF-8 byte order"):
        model_from_bytes(mutated)


def test_the_document_id_block_was_already_guarded_which_is_the_asymmetry_closed() -> None:
    """A repeated document id is rejected, as a repeated token is.

    The equivalent flip one block along turns ``d1`` into a second ``d0``; both
    identifier blocks are checked.
    """
    payload = model_bytes(_three_token_model())
    mutated = payload.replace(b"d0\nd1\nd2", b"d0\nd0\nd2", 1)

    assert mutated != payload, "the flip changed nothing"
    assert len(mutated) == len(payload), "one byte overwritten, not inserted"
    with pytest.raises(TfsxFormatError, match="appears at positions 0 and 1"):
        model_from_bytes(mutated)


def test_an_untouched_container_still_loads_so_the_guard_admits_valid_files() -> None:
    """An untouched container loads, so the order check admits valid files."""
    model = _three_token_model()
    payload = model_bytes(model)

    restored = model_from_bytes(payload)

    assert restored.vocabulary.tokens == model.vocabulary.tokens
    assert restored.vocabulary.is_sorted()
    assert model_bytes(restored) == payload, "and the round trip is still bit-identical"


def _flip_idf(payload: bytes, model: object, to: float) -> bytes:
    """Overwrite the first idf entry, located by its bytes rather than an offset.

    A hard-coded offset moves with any header change and would silently corrupt
    some other field, testing nothing.
    """
    at = payload.index(struct.pack("<d", model.idf.values[0]))  # type: ignore[attr-defined]
    mutated = bytearray(payload)
    mutated[at : at + 8] = struct.pack("<d", to)
    return bytes(mutated)


def test_a_negative_idf_is_refused_rather_than_loaded() -> None:
    """A negative idf is refused rather than loaded.

    Section 2.2 fixes ``idf(t) = log((1 + N) / (1 + df(t))) + 1`` with
    ``df <= N``, so a fitted ``idf`` is at least 1. A negative entry negates the
    scores of queries on its term, and ``ranker.py`` checks only finiteness.
    """
    model = _three_token_model()
    payload = model_bytes(model)

    mutated = _flip_idf(payload, model, -5.0)
    assert len(mutated) == len(payload), "eight bytes overwritten, not inserted"
    with pytest.raises(TfsxFormatError, match=r"idf.*negative"):
        model_from_bytes(mutated)


def test_the_weights_and_norms_were_already_guarded_which_is_the_asymmetry_closed() -> None:
    """A negative weight is refused, as a negative idf is.

    ``check_non_negative`` applies to the weight, norm and idf arrays alike.
    """
    model = _three_token_model()
    payload = model_bytes(model)
    first_weight = model.matrix.values[0]

    at = payload.index(struct.pack("<d", first_weight))
    mutated = bytearray(payload)
    mutated[at : at + 8] = struct.pack("<d", -1.0)
    with pytest.raises(TfsxFormatError, match=r"weights.*negative"):
        model_from_bytes(bytes(mutated))


def test_an_idf_of_exactly_one_is_admitted_because_a_full_df_term_produces_it() -> None:
    """An idf of exactly one is admitted, since a term in every document has it.

    A term with ``df == N`` has ratio exactly 1 and ``idf`` exactly ``1.0``.
    No term is in every document here, so 1.0 is written into ``idf[0]``:
    the parser accepts the whole legitimate range, not only fitted values.
    """
    model = _three_token_model()
    payload = model_bytes(model)

    restored = model_from_bytes(_flip_idf(payload, model, 1.0))

    assert restored.idf.values[0] == 1.0
    assert model_from_bytes(model_bytes(restored)) is not None, "and it round-trips"


# ---------------------------------------------------------------------------
# The integer arrays get the same scrutiny as the float ones
# ---------------------------------------------------------------------------


# `model_bytes` writes ten fields and the parser checks each one: the CSR
# triple, `idf` and `norms`, both string blocks, and `lengths`, `df` and `cf`,
# the last three against bounds every fit satisfies.
def _int_offsets(model: object) -> tuple[int, int, int]:
    """Byte offsets of `lengths`, `df` and `cf`, derived from the layout.

    A byte search does not work here: several of these arrays hold the same
    small integers. Computed offsets also move with a field added ahead of them.
    """
    n_docs = model.n_documents  # type: ignore[attr-defined]
    n_terms = len(model.vocabulary.tokens)  # type: ignore[attr-defined]
    nnz = len(model.matrix.values)  # type: ignore[attr-defined]
    lengths = (
        _HEADER.size
        + 8 * (n_docs + 1)  # indptr, q
        + 4 * nnz  # indices, i
        + 8 * nnz  # values, d
        + 8 * n_terms  # idf, d
        + 8 * n_docs  # norms, d
    )
    return lengths, lengths + 8 * n_docs, lengths + 8 * n_docs + 8 * n_terms


def _poke(payload: bytes, at: int, value: int) -> bytes:
    mutated = bytearray(payload)
    mutated[at : at + 8] = struct.pack("<q", value)
    return bytes(mutated)


def test_a_document_frequency_outside_one_to_n_is_refused() -> None:
    """`1 <= df[t] <= N`, because a term reaches the vocabulary by appearing.

    Zero would place the term in the vocabulary and in no document; above `N`
    would place it in more documents than exist.
    """
    model = _three_token_model()
    payload = model_bytes(model)
    _, df_at, _ = _int_offsets(model)

    for impossible in (-1, 0, model.n_documents + 1, 999):
        with pytest.raises(TfsxFormatError, match=r"df\[0\]"):
            model_from_bytes(_poke(payload, df_at, impossible))


def test_the_document_frequency_at_its_upper_bound_is_accepted() -> None:
    r"""The upper edge of `1 <= df[t] <= N`.

    The rejection test above pokes only values outside the bound, so a guard
    narrowed to `df[t] < N` still passes it. A term occurring in every document
    is ordinary, so the accepting side needs its own case.
    """
    model = _three_token_model()
    payload = model_bytes(model)
    _, df_at, _ = _int_offsets(model)

    restored = model_from_bytes(_poke(payload, df_at, model.n_documents))

    assert restored.vocabulary.df[0] == model.n_documents, (
        "a term recorded in every document is legitimate and must round-trip"
    )


def test_a_fitted_corpus_whose_term_is_in_every_document_round_trips() -> None:
    """The same boundary reached by fitting rather than by editing bytes.

    `df == N` is a value the vectoriser emits for an ordinary corpus, from any
    common word that survives stopword removal. The round trip is checked on the
    recorded counts.
    """
    model = TfidfVectoriser().fit([["aa", "bb"], ["aa", "cc"], ["aa", "dd"]], ["d0", "d1", "d2"])
    universal = model.vocabulary.tokens.index("aa")

    assert model.n_documents == 3, "the premise: three documents"
    assert model.vocabulary.df[universal] == 3, (
        "the premise: aa is in every one of them, so df reaches its upper bound"
    )

    restored = model_from_bytes(model_bytes(model))

    assert restored.vocabulary.df == model.vocabulary.df
    assert restored.vocabulary.tokens == model.vocabulary.tokens


def test_a_collection_frequency_below_its_document_frequency_is_refused() -> None:
    """`cf[t] >= df[t]`: a term occurs at least once in each document holding it.

    Checked against `df` rather than against zero, because the interesting
    corruption is the one that stays positive and still cannot have come from a
    fit.
    """
    model = _three_token_model()
    payload = model_bytes(model)
    _, _, cf_at = _int_offsets(model)
    df_zero = model.vocabulary.df[0]

    assert df_zero > 1, "the premise: there is room below df to be wrong in"
    with pytest.raises(TfsxFormatError, match=r"cf\[0\]"):
        model_from_bytes(_poke(payload, cf_at, df_zero - 1))


def test_a_length_below_the_row_it_describes_is_refused() -> None:
    """`lengths[i] >= nnz(row i)`, since each distinct term occurs at least once.

    A lower bound only: a term occurring three times and one occurring once are
    one non-zero either way, so the exact length cannot be recovered from the
    container and a length above the row's non-zero count is still accepted.
    """
    model = _three_token_model()
    payload = model_bytes(model)
    lengths_at, _, _ = _int_offsets(model)
    row_nnz = model.matrix.indptr[1] - model.matrix.indptr[0]

    for below in range(row_nnz):
        with pytest.raises(TfsxFormatError, match=r"lengths\[0\]"):
            model_from_bytes(_poke(payload, lengths_at, below))

    assert model_from_bytes(_poke(payload, lengths_at, row_nnz)) is not None, (
        "equal to the row's non-zero count is the boundary and is legitimate"
    )
    assert model_from_bytes(_poke(payload, lengths_at, row_nnz + 5)) is not None, (
        "and a longer document is not something this format can refute"
    )


def test_an_untouched_container_still_loads_with_its_integer_arrays_intact() -> None:
    """An untouched container loads with its integer arrays intact."""
    model = _three_token_model()

    restored = model_from_bytes(model_bytes(model))

    assert restored.lengths == model.lengths
    assert restored.vocabulary.df == model.vocabulary.df
    assert restored.vocabulary.cf == model.vocabulary.cf


# ---------------------------------------------------------------------------
# The schema: an orphan module describing the container above
# ---------------------------------------------------------------------------


# `persistence/model.py` declares the shape of a `.tfsx` payload: what
# `tfidf schema` prints and what a reader of the format consults. A mismatch
# with the writer misdescribes every file the project produces.
def test_the_schema_names_every_array_the_container_carries() -> None:
    """The schema names every field the container carries, in order.

    Ten fields: three CSR arrays, five per-term or per-document vectors and two
    string blocks. A dropped field would describe a smaller format.
    """
    names = [f.name for f in MODEL_FIELDS]

    assert names == [
        "indptr",
        "indices",
        "values",
        "idf",
        "norms",
        "lengths",
        "df",
        "cf",
        "tokens",
        "doc_ids",
    ]


def test_every_field_declares_a_type_the_container_actually_writes() -> None:
    """Every field declares a dtype the container writes.

    A reader derives `struct` formats from these names, so a dtype the writer
    never emits would be unimplementable.
    """
    assert {f.dtype for f in MODEL_FIELDS} == {"int32", "int64", "float64", "utf-8"}


def test_the_csr_arrays_declare_the_lengths_the_format_check_enforces() -> None:
    """The CSR arrays declare the lengths that `_check_csr` enforces.

    `_check_csr` rejects an `indptr` that is not `n_docs + 1` long and indices
    that do not match the values; the schema publishes those lengths in words.
    """
    by_name = {f.name: f for f in MODEL_FIELDS}

    assert by_name["indptr"].length == "n_docs + 1"
    assert by_name["indices"].length == by_name["values"].length == "nnz"
    assert by_name["indices"].dtype == "int32", "term ids, as the header packs them"


def test_the_per_term_and_per_document_lengths_are_declared_separately() -> None:
    """Per-term and per-document lengths are declared separately.

    Four vectors are sized by the vocabulary and three by the corpus. One wrong
    size reads a file with the arrays transposed and yields a plausible model.
    """
    by_name = {f.name: f for f in MODEL_FIELDS}

    assert {by_name[n].length for n in ("idf", "df", "cf", "tokens")} == {"n_terms"}
    assert {by_name[n].length for n in ("norms", "lengths", "doc_ids")} == {"n_docs"}


def test_every_field_says_what_it_is_for() -> None:
    """Every field states its purpose.

    The purpose column makes the schema readable without the source.
    """
    assert all(f.purpose for f in MODEL_FIELDS)
    assert all(len(f.purpose) > 5 for f in MODEL_FIELDS)


def test_the_schema_renders_as_plain_data_for_the_manifest() -> None:
    """The schema renders as plain data for the manifest.

    `describe_schema` is what the CLI prints and a manifest can embed, so it is
    JSON-able rather than a list of dataclasses.
    """
    described = describe_schema()

    assert len(described) == len(MODEL_FIELDS)
    assert all(sorted(row) == ["dtype", "length", "name", "purpose"] for row in described)
    assert all(isinstance(v, str) for row in described for v in row.values())


def test_the_rendered_schema_carries_the_same_fields_in_the_same_order() -> None:
    """The rendered schema keeps the same fields in the same order.

    The printed table is read top to bottom against the byte layout.
    """
    assert [row["name"] for row in describe_schema()] == [f.name for f in MODEL_FIELDS]


# ---------------------------------------------------------------------------
# The header's reduction word is the format, not the enum's declaration order
# ---------------------------------------------------------------------------
def test_the_reduction_wire_values_are_pinned_to_the_format() -> None:
    """Each `Reduction` member has a wire code fixed by the format.

    Deriving the code from `list(Reduction)` order would let a reordering in
    `numerics.py` reinterpret every existing container. The mapping is restated
    here, so changing it in `save_load.py` alone fails.
    """
    # Such a reinterpretation would be silent. `FORMAT_VERSION` and the
    # recorded `container_sha256` do not move, re-encoding a loaded container
    # reproduces the same bytes, and `TfidfModel.digest()` does not hash the
    # reduction.
    assert _REDUCTION_CODES == {
        Reduction.NAIVE: 0,
        Reduction.NEUMAIER: 1,
        Reduction.PAIRWISE: 2,
        Reduction.EXACT: 3,
    }
    assert _REDUCTION_BY_CODE == {
        0: Reduction.NAIVE,
        1: Reduction.NEUMAIER,
        2: Reduction.PAIRWISE,
        3: Reduction.EXACT,
    }
    # Every member has a code, and the codes are distinct: a member added
    # without one fails here rather than at the first save.
    assert set(_REDUCTION_CODES) == set(Reduction)
    assert len(set(_REDUCTION_CODES.values())) == len(Reduction)


@pytest.mark.parametrize("policy", list(Reduction))
def test_the_reduction_survives_the_round_trip_by_its_pinned_code(
    mini_model,  # type: ignore[no-untyped-def]
    policy: Reduction,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """The byte written is the pinned code, and it reads back as that member."""
    model = dataclasses.replace(mini_model, reduction=policy)
    path = tmp_path / "model.tfsx"
    save_model(model, path)

    header = _HEADER.unpack_from(path.read_bytes())
    assert header[6] == _REDUCTION_CODES[policy], "the byte on disk is the pinned code"
    assert load_model(path).reduction is policy


# ---------------------------------------------------------------------------
# The sidecar must not be written over the container
# ---------------------------------------------------------------------------
def test_a_json_container_path_is_refused_before_anything_is_written(
    mini_model,  # type: ignore[no-untyped-def]
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """A `.json` container path is refused before any byte is written.

    `Path("m.json").with_suffix(".json")` is `m.json`, so the sidecar would overwrite the
    container and its `container_sha256` would name bytes not on disk.
    `tfidf-stability build-corpus -o model.json` reaches the case: `--output` is free-form.
    """
    target = tmp_path / "model.json"
    with pytest.raises(TfsxFormatError, match=r"must not end in \.json"):
        save_model(mini_model, target)
    assert not target.exists(), "the container must not be written and then rejected"

    # The same name with the container's own suffix is fine, and its sidecar
    # lands beside it rather than on it.
    good = tmp_path / "model.tfsx"
    save_model(mini_model, good)
    assert good.exists()
    assert (tmp_path / "model.json").exists()
    assert load_model(good).digest() == mini_model.digest()


def test_the_sidecar_may_be_declined_for_a_json_path(
    mini_model,  # type: ignore[no-untyped-def]
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """The guard covers the collision, not the suffix.

    With `sidecar=False` nothing would overwrite the container, so a `.json`
    path is allowed."""
    target = tmp_path / "model.json"
    save_model(mini_model, target, sidecar=False)
    assert load_model(target).digest() == mini_model.digest()


# ---------------------------------------------------------------------------
# The schema describes the whole container, not only its arrays
# ---------------------------------------------------------------------------
def test_the_header_schema_matches_the_struct_that_writes_it() -> None:
    """`HEADER_FIELDS` describes every scalar `_HEADER` writes.

    `MODEL_FIELDS` covers only the arrays, so a reader built from
    `tfidf-stability schema` gets `flags` and `reduction` from `HEADER_FIELDS`.
    The count comes from `_HEADER`, so an undescribed scalar fails.
    """
    # `flags` is G13's logarithm switch: it changes about 15% of the idf
    # entries and the weights taken from them. `reduction` carries the
    # reduction policy: it changes `norms` and the cosine scores computed
    # under it.

    # Counted by unpacking a zero header rather than by reading the format
    # string, where "8s" is two characters for one field and "2I" is two fields.
    field_count = len(_HEADER.unpack(bytes(_HEADER.size)))
    assert len(HEADER_FIELDS) == field_count
    assert [f.name for f in HEADER_FIELDS][:5] == [
        "magic",
        "format_version",
        "n_docs",
        "n_terms",
        "nnz",
    ]
    assert [f.name for f in HEADER_FIELDS][-2:] == ["reserved_a", "reserved_b"]
    assert all(f.purpose for f in HEADER_FIELDS)

    # The separator is part of the layout and a reader that assumes the token
    # and document-id blocks abut misparses every container.
    assert BLOCK_SEPARATOR == b"\n"


def test_the_two_schema_halves_are_disjoint_and_both_described() -> None:
    """`MODEL_FIELDS` and `HEADER_FIELDS` describe disjoint halves.

    `MODEL_FIELDS` names the arrays and `HEADER_FIELDS` the header scalars, so
    no name belongs in both. `describe_header` renders the scalars in file
    order."""
    arrays = {f.name for f in MODEL_FIELDS}
    scalars = {f.name for f in HEADER_FIELDS}
    assert not arrays & scalars

    described = describe_header()
    assert [row["name"] for row in described] == [f.name for f in HEADER_FIELDS]
    assert all(set(row) == {"name", "dtype", "purpose"} for row in described)
