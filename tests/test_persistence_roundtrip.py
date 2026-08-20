"""Model serialisation: round-trip fidelity and byte determinism.

Round-trip fidelity: a loaded model must be bit-identical to the saved one.
Everything downstream compares floats bitwise, so a format that lost one ulp
would break every differential test while looking fine.

Byte determinism: saving the same model twice must produce identical bytes.
Hence the hand-rolled container in place of ``numpy.savez``: a zip embeds a
per-member modification timestamp, so two saves of one model differ and the
reproducibility snapshot could not exist.
"""

from __future__ import annotations

import struct

import pytest

from tfidf_stability.persistence.save_load import (
    _FLAG_MASK,
    FORMAT_VERSION,
    MAGIC,
    _check_csr,
    _decode_block,
    load_model,
    model_bytes,
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
    """The strongest single check: the digest covers every weight bitwise."""
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path)
    assert load_model(path).digest() == mini_model.digest()
    assert load_model(path).vocabulary.digest() == mini_model.vocabulary.digest()


@pytest.mark.parametrize("policy", list(Reduction))
def test_the_reduction_policy_round_trips(mini_features, policy, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The policy is part of the model's identity: the same corpus under a
    different policy has different norms and is a different model."""
    model = TfidfVectoriser(reduction=policy).fit(list(mini_features))
    path = tmp_path / f"{policy.value}.tfsx"
    save_model(model, path)
    assert load_model(path).reduction is policy


@pytest.mark.parametrize("impl", list(LogImpl))
def test_the_log_implementation_round_trips(mini_features, impl, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """G13: which logarithm produced the IDF is part of the provenance, because
    the two differ in ~15% of entries."""
    model = TfidfVectoriser(log_impl=impl).fit(list(mini_features))
    path = tmp_path / f"{impl.value}.tfsx"
    save_model(model, path)
    assert load_model(path).idf.log_impl is impl


def test_a_zero_norm_document_round_trips(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The mini corpus has one (all-stopword d5); an empty CSR row is the case
    an off-by-one in the row boundaries would corrupt."""
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path)
    restored = load_model(path)
    assert restored.zero_norm_documents == mini_model.zero_norm_documents
    assert restored.matrix.row(4).nnz == 0


# ---------------------------------------------------------------------------
# Byte determinism
# ---------------------------------------------------------------------------
def test_saving_twice_gives_identical_bytes(mini_model) -> None:  # type: ignore[no-untyped-def]
    """The property ``numpy.savez`` cannot provide, and the reason for the
    hand-rolled container: a zip stores a modification time per member."""
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
    """This parser is the project's only reader of untrusted bytes.

    Every length comes from the header and is checked against the file size
    before any slice is taken, so a truncated file raises rather than yielding a
    model assembled from adjacent bytes.
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
    """A future file must fail loudly rather than be misread by an older parser."""
    path = tmp_path / "future.tfsx"
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, len(MAGIC), FORMAT_VERSION + 99)
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="format version"):
        load_model(path)


def test_a_newline_in_a_token_is_refused(mini_features) -> None:  # type: ignore[no-untyped-def]
    """The token block is LF-separated, so a token containing LF makes the
    encoding ambiguous. Normalisation strips control characters, so real text
    cannot get here; the format refuses it without relying on that."""
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
    """Only one flag is defined, and it selects the correctly-rounded logarithm.

    A future writer that sets a second bit is describing a model this build
    cannot reconstruct. Masking the bit off would produce a plausible model
    computed under the wrong assumption, which is the failure the whole container
    format exists to prevent.
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
# length disagrees with its header is rejected before the arrays are read. They
# guard the direct caller instead, which is what `row()`, the reference scorer
# and the native core rely on when they skip re-checking. Reaching them means
# calling them, and the alternative -- deleting them -- would leave the assumption
# unstated.
def test_an_indptr_of_the_wrong_length_is_rejected() -> None:
    """One offset per row plus a final one. A short indptr makes the last row's
    extent depend on whatever the array happens to end with."""
    with pytest.raises(TfsxFormatError, match=r"indptr has 2 entries; expected n_docs \+ 1 = 3"):
        _check_csr([0, 1], [0], [1.0], n_rows=2, n_cols=4)


def test_a_weight_for_every_column_index_is_required() -> None:
    """The two arrays are read independently with only their lengths fixed, so a
    mismatch pairs a weight with the wrong column in every dot product."""
    with pytest.raises(TfsxFormatError, match="1 weights for 2 column indices"):
        _check_csr([0, 2], [0, 1], [1.0], n_rows=1, n_cols=4)


def test_a_canonical_matrix_passes_every_arm() -> None:
    """The negative cases above only mean something if the positive one passes."""
    _check_csr([0, 2, 3], [0, 2, 1], [1.0, 2.0, 3.0], n_rows=2, n_cols=4)


def test_an_empty_block_decodes_to_no_entries_rather_than_one_empty_string() -> None:
    """Splitting an empty block yields one empty string, so a zero count has to
    be settled before the split or an empty model would gain a phantom token."""
    assert _decode_block(b"", 0, "token") == ()


def test_a_non_empty_block_under_a_zero_count_is_rejected_rather_than_dropped() -> None:
    """Those bytes cannot be reproduced on the way back out, so silently
    discarding them would break the model-to-bytes bijection."""
    with pytest.raises(TfsxFormatError, match="count is 0 but the block holds"):
        _decode_block(b"orphan", 0, "token")


def test_a_block_holding_the_wrong_number_of_entries_is_rejected() -> None:
    """The header states the count and the block carries the bytes; the two are
    written independently, so a file can satisfy every length check and still
    decode to the wrong number of tokens."""
    with pytest.raises(TfsxFormatError, match="expected 3 token entries, decoded 2"):
        _decode_block(b"a\nb", 3, "token")


# ---------------------------------------------------------------------------
# The header, field by field
# ---------------------------------------------------------------------------
# A .tfsx file is read by builds other than the one that wrote it, so the byte
# layout and the exact rejection boundaries are the contract. Mutation testing
# found the boundaries unpinned: `column < n_cols` could become `<=`, the
# reserved-word check could accept a file with one of the two set, and the
# correctly-rounded flag could move to a different bit, all with the suite green.
_RESERVED_A, _RESERVED_B = 52, 56
_REDUCTION_OFFSET = 32


def test_the_correctly_rounded_flag_is_bit_zero(mini_model) -> None:  # type: ignore[no-untyped-def]
    """The bit position is on-disk format. A build that moved it would write
    files this one reads as "platform logarithm" -- a model that differs from the
    normative one in about 15% of its idf entries, loaded without complaint."""
    payload = model_bytes(mini_model)
    (flags,) = struct.unpack_from("<I", payload, _FLAGS_OFFSET)
    assert flags == 1
    assert _FLAG_MASK == 1


def test_a_file_of_exactly_header_length_is_measured_not_dismissed(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The guard is `len(data) < header_size`. At exactly the header size the
    header is complete and readable, so the failure that follows names the real
    problem -- the body is missing -- rather than blaming the header."""
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
    """Both are checked, not just one. A reserved word carries meaning a future
    version defines, so a file that sets one is describing something this build
    cannot honour -- and with `and` in place of `or`, setting exactly one slipped
    through."""
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, offset, 0xABCD)
    path = tmp_path / f"reserved_{which}.tfsx"
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="reserved header words must be zero"):
        load_model(path)


def test_the_reduction_policy_is_rejected_one_past_the_last_one(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`0 <= reduction < len(Reduction)`. The first invalid value is the count
    itself, which is the one an off-by-one in a future writer produces."""
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, _REDUCTION_OFFSET, len(Reduction))
    path = tmp_path / "reduction.tfsx"
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="out of range"):
        load_model(path)

    # And the last valid one is accepted, so the boundary is a boundary.
    struct.pack_into("<I", payload, _REDUCTION_OFFSET, len(Reduction) - 1)
    path.write_bytes(bytes(payload))
    assert load_model(path).matrix.n_rows == mini_model.matrix.n_rows


def test_a_column_index_equal_to_the_vocabulary_size_is_out_of_range() -> None:
    """Columns are 0-based, so `n_cols` is the first invalid one. It is also the
    value an off-by-one produces, and it would index one past the idf array."""
    with pytest.raises(TfsxFormatError, match=r"outside 0\.\.3"):
        _check_csr([0, 1], [4], [1.0], n_rows=1, n_cols=4)
    # One below is fine.
    _check_csr([0, 1], [3], [1.0], n_rows=1, n_cols=4)


def test_a_row_that_repeats_a_column_is_not_strictly_increasing() -> None:
    """Strict increase makes a row's support a set; a duplicate double-counts
    that term in every dot product. Equal neighbours are the case a `<` in place
    of `<=` lets through."""
    with pytest.raises(TfsxFormatError, match="not strictly increasing: 1 then 1"):
        _check_csr([0, 2], [1, 1], [1.0, 2.0], n_rows=1, n_cols=4)


def test_the_sidecar_is_written_unless_it_is_declined(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The default carries the digests a manifest quotes, so silently dropping
    it leaves a model whose provenance has to be recomputed to be checked."""
    with_sidecar = tmp_path / "a.tfsx"
    save_model(mini_model, with_sidecar)
    assert with_sidecar.with_suffix(".json").is_file()

    without = tmp_path / "b.tfsx"
    save_model(mini_model, without, sidecar=False)
    assert not without.with_suffix(".json").exists()


def test_a_loaded_model_reports_no_discarded_terms(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`n_discarded` counts what `max_features` dropped at fit time. It is not in
    the file because it is not part of the model's identity, so a load reports
    zero rather than inventing a count that would then travel into a manifest."""
    path = tmp_path / "model.tfsx"
    save_model(mini_model, path)
    assert load_model(path).vocabulary.n_discarded == 0


def test_the_format_version_this_build_writes_is_one(mini_model) -> None:  # type: ignore[no-untyped-def]
    """It is stamped into every file and compared on every read, so it is the
    number another build's parser is matched against. Asserted both as the
    constant and as the bytes actually written, since only the second is what
    another build sees."""
    assert FORMAT_VERSION == 1
    (written,) = struct.unpack_from("<I", model_bytes(mini_model), len(MAGIC))
    assert written == FORMAT_VERSION


def test_a_rejected_indptr_start_and_end_name_the_value_they_found() -> None:
    """A format error is read by someone holding a file they cannot open. One
    that quotes the wrong number sends them looking in the wrong place."""
    with pytest.raises(TfsxFormatError, match="indptr must start at 0, got 1"):
        _check_csr([1, 2], [0, 1], [1.0, 2.0], n_rows=1, n_cols=4)
    with pytest.raises(TfsxFormatError, match="indptr ends at 9 but there are 2 indices"):
        _check_csr([0, 9], [0, 1], [1.0, 2.0], n_rows=1, n_cols=4)


def test_an_out_of_range_reduction_names_the_range_it_checked(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`0..len(Reduction) - 1`, computed from the enum rather than written down,
    so the message stays right as policies are added."""
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, _REDUCTION_OFFSET, 99)
    path = tmp_path / "reduction99.tfsx"
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match=r"reduction policy 99 is out of range \(0\.\.3\)"):
        load_model(path)


def test_a_descending_pair_is_reported_in_the_order_it_was_found() -> None:
    """The duplicate case above cannot tell the two operands apart -- "1 then 1"
    reads the same whichever index the message quotes. A descending pair can,
    and getting the order backwards in a format diagnostic is how someone ends
    up editing the wrong end of a file."""
    with pytest.raises(TfsxFormatError, match="not strictly increasing: 3 then 1"):
        _check_csr([0, 2], [3, 1], [1.0, 2.0], n_rows=1, n_cols=4)
