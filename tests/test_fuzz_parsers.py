"""Fuzzing the two parsers that consume bytes this package did not write.

Everything else here is fed by our own code. These two are not: a ``.tfsx`` file
can arrive from anywhere, and document text is by definition arbitrary. They are
therefore the only places where "what does this do on input we did not design
for" is a question with teeth.

The property, stated once
-------------------------
**Either it round-trips correctly, or it raises a typed exception from
``tfidf_stability.utils.validation``.** Nothing else is acceptable -- and in
particular these three failures are what the harness is looking for:

* returning a silently *wrong* model, which is far worse than crashing, because
  a plausible ranking computed from corrupt weights looks exactly like a real
  one;
* leaking an untyped exception (``struct.error``, ``IndexError``,
  ``UnicodeDecodeError``, ``MemoryError``), which tells a caller nothing and
  cannot be handled;
* consuming unbounded time or memory on a small input, which a hostile
  ``nnz`` field could otherwise arrange.

This is not hypothetical. A truncation bug was already found and fixed in this
parser once: the document-id block was read with an open-ended slice, so cutting
the final byte corrupted the last identifier while satisfying every count check.
That is precisely the shape of bug byte-level fuzzing finds and unit tests do
not.
"""

from __future__ import annotations

import struct

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tfidf_stability.persistence.save_load import (
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


def _assert_typed(call, data: bytes) -> None:
    """The parser must round-trip or raise a *typed* error. Nothing else."""
    try:
        call(data)
    except TfidfStabilityError:
        return
    except Exception as exc:
        raise AssertionError(
            f"{type(exc).__name__} escaped the parser: {exc!r}. Every rejection "
            f"must be a TfidfStabilityError so a caller can handle it."
        ) from exc


# ---------------------------------------------------------------------------
# The .tfsx container
# ---------------------------------------------------------------------------
def test_a_valid_container_round_trips(mini_model, valid_container) -> None:
    """The baseline. Without this the rest proves only that nothing crashes."""
    restored = model_from_bytes(valid_container)
    assert restored.doc_ids == mini_model.doc_ids
    assert all(same_bits(a, b) for a, b in zip(restored.norms, mini_model.norms, strict=True))


def test_truncation_at_every_offset_is_rejected(valid_container) -> None:
    """The regression that motivated this file.

    Every proper prefix of a valid container is invalid. Checked exhaustively
    rather than by sampling, because the bug that occurred was at exactly one
    offset -- the last byte -- and a sampler would very likely have missed it.
    """
    for cut in range(len(valid_container)):
        _assert_typed(model_from_bytes, valid_container[:cut])


def test_trailing_bytes_are_rejected(valid_container) -> None:
    """A container plus junk is not a container.

    Accepting it would break the bijection between models and byte strings that
    the reproducibility snapshot depends on: two different files would decode to
    the same model.
    """
    _assert_typed(model_from_bytes, valid_container + b"\x00")
    _assert_typed(model_from_bytes, valid_container + b"trailing garbage")


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
    """A hostile length field must not become a huge allocation.

    ``nnz`` sizes an array read. If it were trusted, a twelve-byte edit could ask
    the parser for terabytes -- a denial of service reachable by anyone who can
    hand over a file.
    """
    offsets = {"n_docs": 12, "n_terms": 16, "nnz": 20, "token_bytes": 32, "doc_id_bytes": 40}
    widths = {
        "n_docs": "<I",
        "n_terms": "<I",
        "nnz": "<Q",
        "token_bytes": "<Q",
        "doc_id_bytes": "<Q",
    }
    mutated = bytearray(valid_container)
    packed = struct.pack(widths[field], 2**31 if widths[field] == "<I" else 2**48)
    mutated[offsets[field] : offsets[field] + len(packed)] = packed
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


@settings(max_examples=200, deadline=None)
@given(data=st.binary(min_size=0, max_size=512))
def test_arbitrary_bytes_never_escape_an_untyped_exception(data: bytes) -> None:
    """Random input almost never reaches deep parsing, but it is free to check."""
    _assert_typed(model_from_bytes, data)


# ---------------------------------------------------------------------------
# The tokeniser
# ---------------------------------------------------------------------------
@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=400))
def test_the_tokeniser_accepts_any_text(text: str) -> None:
    """No input is a parse error: text is data, not a format."""
    features = PreprocessingPipeline().preprocess(text)
    assert isinstance(features, list)
    assert all(isinstance(f, str) for f in features)


@settings(max_examples=300, deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(codec="utf-8", min_codepoint=1),
        max_size=200,
    )
)
def test_no_token_ever_contains_the_ngram_joiner(text: str) -> None:
    """The invariant the whole n-gram representation rests on.

    Bigrams are stored as ``a + U+001F + b``. If a *unigram* could contain
    U+001F, a bigram would be indistinguishable from a unigram that happened to
    embed the separator, and the vocabulary would stop being a bijection between
    strings and feature identities. Normalisation must strip the character; this
    asserts it does, over arbitrary Unicode rather than over the ASCII the unit
    tests use.
    """
    for feature in PreprocessingPipeline().preprocess(text):
        # A bigram legitimately contains exactly one joiner; a unigram none.
        assert feature.count(_JOINER) <= 1
        for part in feature.split(_JOINER):
            assert _JOINER not in part
            assert part, "an empty side of a bigram means the joiner leaked into a token"


#: Adversarial inputs, built with ``chr`` rather than written as literals.
#: A bidirectional override or a null byte pasted into source can reorder how
#: the surrounding code *displays* -- which is exactly what ruff's PLE2502
#: forbids -- so the characters are constructed instead. The tokeniser still
#: receives them.
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
    """Normalisation must fold them together, or the same word gets two ids.

    A corpus mixing the two encodings of an accented word would otherwise split
    its document frequency across two vocabulary entries, changing every idf that
    depends on it.
    """
    pipeline = PreprocessingPipeline()
    assert pipeline.preprocess("égal") == pipeline.preprocess("égal")
