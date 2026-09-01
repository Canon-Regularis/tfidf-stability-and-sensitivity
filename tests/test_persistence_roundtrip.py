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

from tfidf_stability.persistence.model import MODEL_FIELDS, describe_schema
from tfidf_stability.persistence.save_load import (
    _FLAG_MASK,
    _HEADER,
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


# ---------------------------------------------------------------------------
# Erroneous: the token block gets the same scrutiny as the document-id block
# ---------------------------------------------------------------------------
# `load_model` is, by its own docstring, the only parser here reading untrusted
# input, and it validated the two identifier blocks unequally: `check_unique_ids`
# on `doc_ids`, nothing on `tokens`. `Vocabulary` is frozen with no
# `__post_init__`, so constructing one directly bypassed the `is_sorted` guard
# that `build_vocabulary` runs.
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
    """One byte, and every score afterwards is against the wrong column.

    Measured before the guard existed, on the 301-byte container this builds:
    flipping the ``b`` of ``ab`` gives the token block ``aa aa cc``, which
    ``model_from_bytes`` accepted. ``_index`` is built by enumeration, so the
    later ``aa`` wins and the token maps to column 1. The query ``["aa"]`` then
    scored ``d0`` -- which contains "aa" twice -- at exactly 0.0, and ``d1``,
    which contains no "aa" at all, at 0.707.

    Nothing downstream would have caught it: the mutated container re-serialises
    to itself, so the round-trip property the fuzz suite checks still holds.
    """
    payload = model_bytes(_three_token_model())
    mutated, at = _flip_to_duplicate_a_token(payload)

    assert mutated != payload, f"the flip at byte {at} changed nothing"
    assert len(mutated) == len(payload), "one byte overwritten, not inserted"
    with pytest.raises(TfsxFormatError, match="not in UTF-8 byte order"):
        model_from_bytes(mutated)


def test_a_vocabulary_merely_out_of_order_is_refused_too_not_only_a_repeated_one() -> None:
    """Uniqueness is not the invariant; ascent is.

    A permuted block passes any uniqueness test while breaking the binary
    searches in ``tf.py`` and the merge in ``align_models``, both of which are
    correct only on a sorted vocabulary. This is why the guard checks the order
    rather than reusing ``check_unique_ids`` a second time.
    """
    payload = model_bytes(_three_token_model())
    mutated = payload.replace(b"aa\nab\ncc", b"ab\naa\ncc", 1)

    assert mutated != payload, "the permutation changed nothing"
    assert len(mutated) == len(payload), "a permutation, not a resize"
    with pytest.raises(TfsxFormatError, match="not in UTF-8 byte order"):
        model_from_bytes(mutated)


def test_the_document_id_block_was_already_guarded_which_is_the_asymmetry_closed() -> None:
    """The contrast that makes the two tests above a fix rather than a new rule.

    The equivalent flip one block along -- turning ``d1`` into a second ``d0`` --
    was always rejected. The parser was not lax; it was inconsistent.
    """
    payload = model_bytes(_three_token_model())
    mutated = payload.replace(b"d0\nd1\nd2", b"d0\nd0\nd2", 1)

    assert mutated != payload, "the flip changed nothing"
    assert len(mutated) == len(payload), "one byte overwritten, not inserted"
    with pytest.raises(TfsxFormatError, match="appears at positions 0 and 1"):
        model_from_bytes(mutated)


def test_an_untouched_container_still_loads_so_the_guard_admits_valid_files() -> None:
    """The guard's other half: a check that rejects everything is not a check."""
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
    """It cannot come from a fit, and it inverts the ranking if it is let in.

    Section 2.2 fixes ``idf(t) = log((1 + N) / (1 + df(t))) + 1`` and ``df <= N``
    always, so the ratio is at least 1 and ``idf`` at least 1. A negative entry
    is not a value this project can produce; it is a corrupt file.

    The parser checked ``idf`` for finiteness and, unlike the two arrays beside
    it, not for sign. Measured before the guard: flipping the eight bytes of
    ``idf[0]`` to ``-5.0`` in a 301-byte container was accepted, and the query
    ``["aa"]`` scored ``[-0.894, 0.0, -0.707]`` against a genuine
    ``[+0.894, 0.0, +0.707]``. Every score is outside the ``[0, 1]`` that section
    2.3 documents, and no error was raised on load, on scoring or on ranking --
    ``ranker.py`` checks scores for finiteness but not for sign.
    """
    model = _three_token_model()
    payload = model_bytes(model)

    mutated = _flip_idf(payload, model, -5.0)
    assert len(mutated) == len(payload), "eight bytes overwritten, not inserted"
    with pytest.raises(TfsxFormatError, match=r"idf.*negative"):
        model_from_bytes(mutated)


def test_the_weights_and_norms_were_already_guarded_which_is_the_asymmetry_closed() -> None:
    """The contrast that makes the test above a fix rather than a new rule.

    ``check_non_negative`` was already applied to the weight and norm arrays. It
    was the third array of the same kind that went unchecked, so the parser was
    not lax -- it was inconsistent, the same shape as the token block against the
    document-id block one field along.
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
    """The guard's other half: the boundary a real fit reaches must still load.

    A term appearing in every document has ``df == N``, so the ratio is exactly 1
    and ``idf`` exactly ``1.0``. A guard rejecting that would refuse ordinary
    corpora. ``cc`` is not in every document here, so the value is constructed
    rather than fitted, which is the point -- the parser must accept the whole
    legitimate range, not merely the values this fixture happens to produce.
    """
    model = _three_token_model()
    payload = model_bytes(model)

    restored = model_from_bytes(_flip_idf(payload, model, 1.0))

    assert restored.idf.values[0] == 1.0
    assert model_from_bytes(model_bytes(restored)) is not None, "and it round-trips"


# ---------------------------------------------------------------------------
# The integer arrays get the same scrutiny as the float ones
# ---------------------------------------------------------------------------
# `model_bytes` writes ten fields. Seven were validated -- the CSR triple, the
# three float arrays, and both string blocks -- and the three int64 arrays
# (`lengths`, `df`, `cf`) were not, so values no fit can produce loaded without
# complaint. Same shape as the token block and the idf sign before them: a
# sibling treated differently, in the one parser that reads untrusted input.
def _int_offsets(model: object) -> tuple[int, int, int]:
    """Byte offsets of `lengths`, `df` and `cf`, derived from the layout.

    Computed rather than hard-coded, so a field added ahead of them moves these
    with it. Searching the bytes for a value does not work here: several of
    these arrays hold the same small integers, and a search finds whichever
    comes first.
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

    Zero would say the term is in the vocabulary and in no document; greater
    than `N` would say it is in more documents than exist. Both were accepted.
    """
    model = _three_token_model()
    payload = model_bytes(model)
    _, df_at, _ = _int_offsets(model)

    for impossible in (-1, 0, model.n_documents + 1, 999):
        with pytest.raises(TfsxFormatError, match=r"df\[0\]"):
            model_from_bytes(_poke(payload, df_at, impossible))


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

    A lower bound and no more: the exact length cannot be recovered from the
    container, because a term occurring three times and one occurring once are
    one non-zero either way. Measured before the guard: `lengths[0]` moved from
    3 to 7 was accepted and `intermediates(0)` published `count=5, tf=0.714`
    against the genuine `count=2, tf=0.667`. That value is still accepted, and
    the test says so rather than pretending otherwise.
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
    """The guards' other half: three new checks that reject nothing valid."""
    model = _three_token_model()

    restored = model_from_bytes(model_bytes(model))

    assert restored.lengths == model.lengths
    assert restored.vocabulary.df == model.vocabulary.df
    assert restored.vocabulary.cf == model.vocabulary.cf


# ---------------------------------------------------------------------------
# The schema: an orphan module describing the container above
# ---------------------------------------------------------------------------
# `persistence/model.py` had no owning test file. It is the declared shape of a
# `.tfsx` payload -- the thing `tfidf schema` prints and a reader of the format
# consults -- so a drift between it and the writer would misdescribe every file
# this project has produced.
def test_the_schema_names_every_array_the_container_carries() -> None:
    """Ten fields: three CSR arrays, four per-term or per-document vectors, and
    the two string blocks. A field dropped here would leave a reader believing
    the format is smaller than it is."""
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
    """The reader reaches for `struct` formats from these names, so a dtype the
    writer never emits would be unimplementable."""
    assert {f.dtype for f in MODEL_FIELDS} == {"int32", "int64", "float64", "utf-8"}


def test_the_csr_arrays_declare_the_lengths_the_format_check_enforces() -> None:
    """`_check_csr` rejects an `indptr` that is not `n_docs + 1` long and
    indices that do not match the values. The schema is where those lengths are
    published, so the two have to agree in words as well as in behaviour."""
    by_name = {f.name: f for f in MODEL_FIELDS}

    assert by_name["indptr"].length == "n_docs + 1"
    assert by_name["indices"].length == by_name["values"].length == "nnz"
    assert by_name["indices"].dtype == "int32", "term ids, as the header packs them"


def test_the_per_term_and_per_document_lengths_are_declared_separately() -> None:
    """Four vectors are sized by the vocabulary and three by the corpus. Getting
    one wrong is exactly the mistake that reads a file with the arrays
    transposed and produces a plausible model."""
    by_name = {f.name: f for f in MODEL_FIELDS}

    assert {by_name[n].length for n in ("idf", "df", "cf", "tokens")} == {"n_terms"}
    assert {by_name[n].length for n in ("norms", "lengths", "doc_ids")} == {"n_docs"}


def test_every_field_says_what_it_is_for() -> None:
    """The purpose column is what makes the schema readable without the source.
    An empty one would leave a field named and unexplained."""
    assert all(f.purpose for f in MODEL_FIELDS)
    assert all(len(f.purpose) > 5 for f in MODEL_FIELDS)


def test_the_schema_renders_as_plain_data_for_the_manifest() -> None:
    """`describe_schema` is what the CLI prints and what a manifest can embed,
    so it must be JSON-able rather than a list of dataclasses."""
    described = describe_schema()

    assert len(described) == len(MODEL_FIELDS)
    assert all(sorted(row) == ["dtype", "length", "name", "purpose"] for row in described)
    assert all(isinstance(v, str) for row in described for v in row.values())


def test_the_rendered_schema_carries_the_same_fields_in_the_same_order() -> None:
    """Positional agreement, not just set agreement: the printed table is read
    top to bottom against the byte layout."""
    assert [row["name"] for row in describe_schema()] == [f.name for f in MODEL_FIELDS]
