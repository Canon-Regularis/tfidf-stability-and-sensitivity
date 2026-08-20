"""Fuzzing the two parsers that consume bytes this package did not write.

A ``.tfsx`` file can arrive from anywhere and document text is arbitrary; every
other input in the suite comes from our own code.

The property, stated once
-------------------------
Either it round-trips correctly, or it raises a typed exception from
``tfidf_stability.utils.validation``. The three failures being hunted:

* a silently wrong model, worse than a crash, because a plausible ranking
  computed from corrupt weights looks like a real one;
* an untyped exception (``struct.error``, ``IndexError``,
  ``UnicodeDecodeError``, ``MemoryError``), which a caller cannot handle;
* unbounded time or memory on a small input, which a hostile ``nnz`` field could
  otherwise arrange.

A truncation bug was found and fixed here once: the document-id block was read
with an open-ended slice, so cutting the final byte corrupted the last identifier
while satisfying every count check.
"""

from __future__ import annotations

import itertools
import struct

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tfidf_stability.persistence.save_load import (
    _HEADER,
    MAGIC,
    model_bytes,
    model_from_bytes,
)
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline
from tfidf_stability.utils.numerics import same_bits
from tfidf_stability.utils.validation import TfidfStabilityError

#: The n-gram joiner. If a raw token could contain it, a bigram could collide
#: with a unigram and the vocabulary would stop being a bijection.
_JOINER = "\x1f"


@pytest.fixture(scope="module")
def valid_container(mini_model) -> bytes:
    return model_bytes(mini_model)


#: The header layout, in the order ``_read_header`` unpacks it. Offsets are
#: derived from it because two hand-written ones were wrong: ``token_bytes`` as 32
#: (which is ``reduction``) and ``doc_id_bytes`` as 40 (the upper half of
#: ``token_bytes``). The ``doc_id_bytes`` case never touched ``doc_id_bytes`` and
#: passed anyway, since clobbering a neighbouring field also gets the container
#: rejected.
_HEADER_LAYOUT: tuple[tuple[str, str], ...] = (
    ("magic", "8s"),
    ("version", "I"),
    ("n_docs", "I"),
    ("n_terms", "I"),
    ("nnz", "Q"),
    ("flags", "I"),
    ("reduction", "I"),
    ("token_bytes", "Q"),
    ("doc_id_bytes", "Q"),
    ("reserved_a", "I"),
    ("reserved_b", "I"),
)


def _header_field(name: str) -> tuple[int, str]:
    """Byte offset and struct code of one header field."""
    offset = 0
    for field, code in _HEADER_LAYOUT:
        if field == name:
            return offset, f"<{code}"
        offset += struct.calcsize(f"<{code}")
    raise KeyError(name)


def test_the_header_layout_this_file_assumes_is_the_real_one() -> None:
    """Guards every offset-based test below against a format change."""
    total = sum(struct.calcsize(f"<{code}") for _, code in _HEADER_LAYOUT)
    assert total == _HEADER.size, f"layout sums to {total}, _HEADER.size is {_HEADER.size}"
    assert _header_field("magic")[0] == 0
    assert _header_field("token_bytes")[0] == 36, "the offset the old test got wrong"
    assert _header_field("doc_id_bytes")[0] == 44, "the offset the old test got wrong"


def _assert_typed(call, data: bytes) -> None:
    """The parser must round-trip or raise a typed error.

    Returning normally used to be enough to pass, so "accepted" meant
    "unexamined": before ``_check_csr`` existed, 425 of the 2715 mutated
    containers this parser accepted carried an ``indptr`` that was non-monotonic
    or did not start at zero, and nothing downstream re-checks it.
    """
    try:
        restored = call(data)
    except TfidfStabilityError:
        return
    except Exception as exc:
        raise AssertionError(
            f"{type(exc).__name__} escaped the parser: {exc!r}. Every rejection "
            f"must be a TfidfStabilityError so a caller can handle it."
        ) from exc

    # Accepting a mutation is legitimate (a flipped mantissa bit gives a different
    # valid model); accepting one that violates the CSR invariants is a defect.
    # The invariants are spelled out here rather than delegated to the parser's own
    # ``_check_csr``, which would take this test with it if it were deleted.
    #
    # Idempotence is the wrong property: a model whose indptr starts at 255
    # re-serialises and reparses to identical bytes, so a fixed-point assertion
    # passes straight over it.
    matrix = restored.matrix
    indptr, indices = list(matrix.indptr), list(matrix.indices)
    assert len(indptr) == matrix.n_rows + 1, f"indptr has {len(indptr)} entries"
    assert indptr[0] == 0, f"indptr starts at {indptr[0]}"
    assert indptr[-1] == len(indices), f"indptr ends at {indptr[-1]}, nnz is {len(indices)}"
    assert len(matrix.values) == len(indices), "one weight per column index"
    for row in range(matrix.n_rows):
        lo, hi = indptr[row], indptr[row + 1]
        assert lo <= hi, f"indptr decreases at row {row}"
        segment = indices[lo:hi]
        assert all(0 <= c < matrix.n_cols for c in segment), f"row {row} column out of range"
        assert all(b > a for a, b in itertools.pairwise(segment)), (
            f"row {row} column indices are not strictly increasing"
        )


# ---------------------------------------------------------------------------
# The .tfsx container
# ---------------------------------------------------------------------------
def test_a_valid_container_round_trips(mini_model, valid_container) -> None:
    """The baseline. Without it the rest proves only that nothing crashes."""
    restored = model_from_bytes(valid_container)
    assert restored.doc_ids == mini_model.doc_ids
    assert all(same_bits(a, b) for a, b in zip(restored.norms, mini_model.norms, strict=True))


def test_truncation_at_every_offset_is_rejected(valid_container) -> None:
    """The regression that motivated this file.

    Every proper prefix of a valid container is invalid. Checked exhaustively: the
    bug was at a single offset, the last byte, which a sampler would likely have
    missed.
    """
    for cut in range(len(valid_container)):
        _assert_typed(model_from_bytes, valid_container[:cut])


def test_trailing_bytes_are_rejected(valid_container) -> None:
    """Accepting a container plus junk breaks the bijection between models and
    byte strings that the reproducibility snapshot depends on: two files would
    decode to one model.
    """
    _assert_typed(model_from_bytes, valid_container + b"\x00")
    _assert_typed(model_from_bytes, valid_container + b"trailing garbage")


@pytest.mark.property
@settings(
    max_examples=400, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(index=st.integers(min_value=0, max_value=4095), value=st.integers(0, 255))
def test_a_single_byte_flip_never_yields_a_silently_wrong_model(
    valid_container, index: int, value: int
) -> None:
    """Corrupt one byte and require a correct model or a typed error."""
    if index >= len(valid_container):
        return
    mutated = bytearray(valid_container)
    if mutated[index] == value:
        return
    mutated[index] = value
    _assert_typed(model_from_bytes, bytes(mutated))


@pytest.mark.parametrize("field", ["n_docs", "n_terms", "nnz", "token_bytes", "doc_id_bytes"])
def test_an_absurd_count_is_rejected_without_allocating(valid_container, field: str) -> None:
    """A hostile length field must not become a huge allocation. ``nnz`` sizes an
    array read, so if it were trusted a twelve-byte edit could ask the parser for
    terabytes: denial of service reachable by anyone who can hand over a file.
    """
    offset, width = _header_field(field)
    mutated = bytearray(valid_container)
    packed = struct.pack(width, 2**31 if width == "<I" else 2**48)
    mutated[offset : offset + len(packed)] = packed
    _assert_typed(model_from_bytes, bytes(mutated))


def test_a_bad_magic_is_rejected(valid_container) -> None:
    _assert_typed(model_from_bytes, b"NOTATFSX" + valid_container[8:])


def test_an_unknown_flag_bit_is_rejected(valid_container) -> None:
    """Ignoring an unknown bit would let two files decode to one model."""
    mutated = bytearray(valid_container)
    mutated[24:28] = struct.pack("<I", 0xFFFF_FFFF)
    _assert_typed(model_from_bytes, bytes(mutated))


def test_the_empty_input_is_rejected() -> None:
    _assert_typed(model_from_bytes, b"")
    _assert_typed(model_from_bytes, MAGIC)


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(data=st.binary(min_size=0, max_size=512))
def test_arbitrary_bytes_never_escape_an_untyped_exception(data: bytes) -> None:
    """Random input almost never reaches deep parsing; the check is free."""
    _assert_typed(model_from_bytes, data)


# ---------------------------------------------------------------------------
# The tokeniser
# ---------------------------------------------------------------------------
@pytest.mark.property
@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=400))
def test_the_tokeniser_accepts_any_text(text: str) -> None:
    """No input is a parse error; text is data rather than a format."""
    features = PreprocessingPipeline().preprocess(text)
    assert isinstance(features, list)
    assert all(isinstance(f, str) for f in features)


@pytest.mark.property
@settings(max_examples=300, deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(codec="utf-8", min_codepoint=1),
        max_size=200,
    )
)
def test_no_token_ever_contains_the_ngram_joiner(text: str) -> None:
    """The invariant the n-gram representation rests on.

    Bigrams are stored as ``a + U+001F + b``, so a unigram containing U+001F would
    be indistinguishable from a bigram and the vocabulary would stop being a
    bijection between strings and feature identities. Normalisation strips the
    character; checked here over arbitrary Unicode, where the unit tests use
    ASCII.
    """
    for feature in PreprocessingPipeline().preprocess(text):
        # A bigram legitimately contains exactly one joiner; a unigram none.
        assert feature.count(_JOINER) <= 1
        for part in feature.split(_JOINER):
            assert _JOINER not in part
            assert part, "an empty side of a bigram means the joiner leaked into a token"


#: Adversarial inputs, built with ``chr`` rather than written as literals: a
#: bidirectional override or a null byte pasted into source reorders how the
#: surrounding code displays (ruff PLE2502). The tokeniser receives the same
#: characters either way.
_ADVERSARIAL_TEXT = [
    _JOINER,  # the joiner alone
    "a" + _JOINER + "b",  # the joiner between letters
    chr(0) + chr(1) + chr(0x1B) + chr(0x7F),  # control characters
    "\U0001f600 \U0001f4a9",  # astral plane
    "e" + chr(0x301) + "gal",  # combining acute
    "égal",  # the same word precomposed
    "a" * 5000,  # far over the token length bound
    chr(0x202E) + "reversed",  # a bidirectional override
    chr(0xFEFF) + "bom",  # a byte-order mark mid-text
    chr(0x200B).join("zero width"),  # zero-width spaces
]


@pytest.mark.parametrize("text", _ADVERSARIAL_TEXT)
def test_adversarial_text_is_handled_without_raising(text: str) -> None:
    features = PreprocessingPipeline().preprocess(text)
    assert all(_JOINER not in part for f in features for part in [f.replace(_JOINER, "", 1)])


def test_combining_and_precomposed_forms_agree() -> None:
    """Normalisation must fold them together, or the same word gets two ids: a
    corpus mixing the two encodings of an accented word splits its document
    frequency across two vocabulary entries and moves every idf that depends on it.
    """
    pipeline = PreprocessingPipeline()
    assert pipeline.preprocess("égal") == pipeline.preprocess("égal")


def test_the_block_separator_is_validated_not_merely_skipped(valid_container) -> None:
    """The container must be a bijection between models and byte strings.

    Both block lengths are fixed by the header, so the separating LF carries no
    information and any value for it was accepted: all 255 alternatives decoded to
    a byte-identical model, 256 distinct files sharing one meaning. ``_FLAG_MASK``
    rejects unknown flag bits one byte further down for that same reason.
    """
    header = _HEADER.unpack_from(valid_container, 0)
    n_docs, n_terms, nnz, token_bytes = header[2], header[3], header[4], header[7]
    offset = (
        _HEADER.size
        + (n_docs + 1) * 8  # indptr
        + nnz * 4  # indices
        + nnz * 8  # values
        + n_terms * 8  # idf
        + n_docs * 8  # norms
        + n_docs * 8  # lengths
        + n_terms * 8  # df
        + n_terms * 8  # cf
        + token_bytes
    )
    assert valid_container[offset : offset + 1] == b"\n", "the separator is where we think"

    for value in (0x00, 0x20, 0x0D, 0xFF):
        mutated = bytearray(valid_container)
        mutated[offset] = value
        with pytest.raises(TfidfStabilityError, match="between the token and document-id blocks"):
            model_from_bytes(bytes(mutated))
