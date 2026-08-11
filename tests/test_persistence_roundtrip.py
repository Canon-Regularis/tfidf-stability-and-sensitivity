"""Model serialisation: round-trip fidelity and byte determinism.

Two properties, and the second is the one that is easy to lose.

**Round-trip fidelity** -- a loaded model must be bit-identical to the saved one,
not merely close. Everything downstream compares floats bitwise, so a format
that lost a single ulp would break every differential test while looking fine.

**Byte determinism** -- saving the same model twice must produce identical
bytes. This is why the format is hand-rolled rather than ``numpy.savez``: a zip
embeds a per-member modification timestamp, so two saves of an identical model
differ, and the reproducibility snapshot could not exist.
"""

from __future__ import annotations

import struct

import pytest

from tfidf_stability.persistence.save_load import (
    FORMAT_VERSION,
    MAGIC,
    load_model,
    model_bytes,
    save_model,
)
from tfidf_stability.utils.hashing import hash_bytes, hash_floats
from tfidf_stability.utils.numerics import Reduction, same_bits
from tfidf_stability.vectorisation.idf import LogImpl
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

TfsxFormatError = pytest.importorskip("tfidf_stability.persistence.save_load").TfsxFormatError


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
def test_round_trip_is_bit_identical(mini_model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Not "close" -- identical. Downstream comparisons are bitwise."""
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
    """The container must be sensitive to exactly what the project cares about."""
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
    *before* any slice is taken, so a truncated file raises rather than quietly
    producing a model assembled from adjacent bytes.
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
    """A future file must fail loudly, not be misread by an older parser."""
    path = tmp_path / "future.tfsx"
    payload = bytearray(model_bytes(mini_model))
    struct.pack_into("<I", payload, len(MAGIC), FORMAT_VERSION + 99)
    path.write_bytes(bytes(payload))
    with pytest.raises(TfsxFormatError, match="format version"):
        load_model(path)


def test_a_newline_in_a_token_is_refused(mini_features) -> None:  # type: ignore[no-untyped-def]
    """The token block is LF-separated, so a token containing LF would make the
    encoding ambiguous. Normalisation strips control characters, so this cannot
    arise from real text -- but the format refuses it rather than relying on
    that."""
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
