"""Tests for the bytes a run writes and the digest that names them.

`utils/hashing.py` digests bit patterns, normalises CRLF in text and streams
files; `utils/io.py` writes atomically and renders strict JSON. Fault injection
patches the OS boundary only. Rationale: docs/engineering_notes.md#canonical-bytes-and-digests
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import unicodedata
from pathlib import Path

import pytest

from tfidf_stability.persistence.manifest import RunManifest
from tfidf_stability.utils.hashing import (
    _CHUNK,
    hash_bytes,
    hash_file,
    hash_floats,
    hash_ints,
    hash_json,
    hash_manifest_lines,
    hash_text,
    short,
)
from tfidf_stability.utils.io import (
    VOLATILE_KEYS,
    atomic_write_bytes,
    atomic_write_text,
    canonical_json,
    read_jsonl,
    strip_volatile,
    write_json,
    write_jsonl,
)


def _refuse_constant(name: str) -> object:
    """A `parse_constant` hook that rejects the non-standard JSON tokens.

    `json.loads` accepts `NaN`, `Infinity` and `-Infinity` by default, so a
    strict reader must refuse them explicitly.
    """
    raise ValueError(name)


_SHA256_OF_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ---------------------------------------------------------------------------
# Digests: normal
# ---------------------------------------------------------------------------
def test_hashing_empty_bytes_gives_the_known_sha256_of_nothing() -> None:
    """The digest of no bytes equals the published SHA-256 of the empty string."""
    assert hash_bytes(b"") == _SHA256_OF_EMPTY


def test_text_hashing_is_the_utf8_bytes_of_the_normalised_text() -> None:
    assert hash_text("abc") == hash_bytes(b"abc")
    assert hash_text("") == _SHA256_OF_EMPTY


def test_crlf_and_lf_text_hash_identically_so_a_checkout_platform_cannot_move_it() -> None:
    """CRLF and LF text hash identically.

    Results are byte-identical across operating systems, so a digest must
    not vary with the line endings git checked out.
    """
    assert hash_text("a\r\nb\r\n") == hash_text("a\nb\n")
    assert hash_text("a\r\nb") != hash_text("a\rb"), "a lone CR is not a line ending here"


def test_a_lone_cr_is_left_alone_because_only_crlf_is_normalised() -> None:
    assert hash_text("a\rb") == hash_bytes(b"a\rb")


# ---------------------------------------------------------------------------
# Digests: the bit patterns
# ---------------------------------------------------------------------------
def test_float_hashing_packs_the_bit_pattern_rather_than_a_rendering() -> None:
    values = [0.1, 2.0, -3.5]
    expected = hash_bytes(b"".join(struct.pack("<d", v) for v in values))
    assert hash_floats(values) == expected


def test_two_floats_one_ulp_apart_hash_differently() -> None:
    """Two floats one ulp apart hash differently.

    A digest over `repr` could collide them; a digest over the bit pattern
    cannot.
    """
    a = 1.0
    b = math.nextafter(1.0, math.inf)
    assert a != b
    assert hash_floats([a]) != hash_floats([b])


def test_positive_and_negative_zero_hash_differently() -> None:
    """Positive and negative zero hash differently.

    `0.0 == -0.0` is True, so a digest over values would call them the same.
    """
    assert 0.0 == -0.0
    assert hash_floats([0.0]) != hash_floats([-0.0])


def test_two_distinct_nans_hash_by_their_bits_not_by_comparison() -> None:
    """Two NaNs with identical bits hash identically.

    NaN compares unequal to every value including another NaN, so only a
    digest over bytes can group them.
    """
    first, second = float("nan"), float("nan")
    assert first is not second, "two separately constructed objects"
    assert first != second, "the premise: NaN is unequal to NaN"
    assert hash_floats([first]) == hash_floats([second]), "identical bit patterns must agree"


def test_integer_hashing_packs_fixed_width_little_endian() -> None:
    assert hash_ints([1, 2, 3]) == hash_bytes(b"".join(struct.pack("<q", v) for v in [1, 2, 3]))


@pytest.mark.parametrize(("width", "code"), [(4, "<i"), (8, "<q")])
def test_both_integer_widths_pack_as_documented(width: int, code: str) -> None:
    values = [0, -1, 7]
    assert hash_ints(values, width=width) == hash_bytes(
        b"".join(struct.pack(code, v) for v in values)
    )


def test_the_two_integer_widths_disagree_so_the_width_is_part_of_the_identity() -> None:
    assert hash_ints([1], width=4) != hash_ints([1], width=8)


def test_an_unsupported_integer_width_is_rejected_not_silently_widened() -> None:
    with pytest.raises(KeyError, match=r"^2$"):
        hash_ints([1], width=2)


def test_an_integer_beyond_the_chosen_width_is_rejected_rather_than_truncated() -> None:
    """Truncation would make two different corpora hash alike."""
    # Two spellings: CPython 3.13 and earlier raise "argument out of range",
    # 3.14 raises "'i' format requires -2147483648 <= number <= 2147483647".
    # The alternation keeps the assertion about the refusal rather than about
    # which interpreter ran it; CI spans both.
    with pytest.raises(struct.error, match=r"format requires|argument out of range"):
        hash_ints([2**31], width=4)


def test_hashing_no_values_at_all_is_the_digest_of_nothing() -> None:
    assert hash_floats([]) == _SHA256_OF_EMPTY
    assert hash_ints([]) == _SHA256_OF_EMPTY


# ---------------------------------------------------------------------------
# Digests: files and manifests
# ---------------------------------------------------------------------------
def test_hashing_a_file_as_binary_matches_hashing_its_bytes(tmp_path: Path) -> None:
    """The binary branch of `hash_file` digests the raw bytes."""
    target = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 8
    target.write_bytes(payload)
    assert hash_file(target, text=False) == hash_bytes(payload)


def test_a_file_larger_than_one_chunk_hashes_the_same_as_its_bytes(tmp_path: Path) -> None:
    """A file above the 1 MiB chunk size hashes as its bytes.

    The read loop runs more than once only above the chunk size.
    """
    target = tmp_path / "big.bin"
    payload = bytes(range(256)) * 8192  # 2 MiB, so at least two chunks
    target.write_bytes(payload)
    assert len(payload) > (1 << 20)
    assert hash_file(target, text=False) == hash_bytes(payload)


def test_hashing_a_file_as_text_normalises_line_endings(tmp_path: Path) -> None:
    crlf, lf = tmp_path / "crlf.txt", tmp_path / "lf.txt"
    crlf.write_bytes(b"a\r\nb\r\n")
    lf.write_bytes(b"a\nb\n")
    assert hash_file(crlf, text=True) == hash_file(lf, text=True)
    assert hash_file(crlf, text=False) != hash_file(lf, text=False), (
        "as binaries they genuinely differ, which is why the two modes exist"
    )


def test_hashing_a_missing_file_raises_rather_than_returning_a_digest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        hash_file(tmp_path / "absent", text=False)


def test_a_manifest_digest_is_order_independent_but_content_sensitive() -> None:
    """A manifest digest ignores entry order but not entry content.

    The listing is sorted before hashing.
    """
    a = [("one.txt", "aa"), ("two.txt", "bb")]
    assert hash_manifest_lines(a) == hash_manifest_lines(list(reversed(a)))
    assert hash_manifest_lines(a) != hash_manifest_lines([("one.txt", "aa"), ("two.txt", "cc")])


def test_a_manifest_digest_uses_the_two_space_sha256sum_layout() -> None:
    """The format sha256sum writes, so a manifest stays checkable by the tool."""
    assert hash_manifest_lines([("f", "abc")]) == hash_text("abc  f\n")


def test_an_empty_manifest_is_the_digest_of_nothing() -> None:
    assert hash_manifest_lines([]) == _SHA256_OF_EMPTY


def test_json_hashing_is_insensitive_to_key_order_but_not_to_values() -> None:
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})
    assert hash_json({"a": 1}) != hash_json({"a": 2})


def test_short_truncates_without_claiming_to_identify() -> None:
    digest = hash_bytes(b"x")
    assert short(digest) == digest[:12]
    assert len(short(digest)) == 12
    assert short(digest, 4) == digest[:4]


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------
def test_an_atomic_write_leaves_the_bytes_and_no_temporary_behind(tmp_path: Path) -> None:
    target = atomic_write_bytes(tmp_path / "out.bin", b"payload")
    assert target.read_bytes() == b"payload"
    assert list(tmp_path.iterdir()) == [target], "a temporary file survived the write"


def test_an_atomic_write_creates_missing_parent_directories(tmp_path: Path) -> None:
    target = atomic_write_bytes(tmp_path / "a" / "b" / "out.bin", b"x")
    assert target.read_bytes() == b"x"


def test_a_failed_rename_removes_the_temporary_and_leaves_the_original(tmp_path: Path) -> None:
    """A failed rename removes the temporary and leaves the original intact.

    `os.replace` is patched at the OS boundary, not inside the package.
    Rationale: docs/engineering_notes.md#canonical-bytes-and-digests
    """
    target = tmp_path / "out.bin"
    atomic_write_bytes(target, b"original")

    def exploding_replace(src: object, dst: object) -> None:
        raise OSError("rename refused")

    real_replace = os.replace
    os.replace = exploding_replace  # type: ignore[assignment]
    try:
        with pytest.raises(OSError, match="rename refused"):
            atomic_write_bytes(target, b"replacement")
    finally:
        os.replace = real_replace  # type: ignore[assignment]

    assert target.read_bytes() == b"original", "a failed write must not damage the original"
    assert list(tmp_path.iterdir()) == [target], "the temporary file outlived the failure"


def test_cleanup_does_not_mask_the_failure_that_caused_it(tmp_path: Path) -> None:
    """Cleanup of a temporary that has already gone does not mask the error.

    `missing_ok=True` on the cleanup unlink matters only once the temporary is
    gone, which the test above never reaches; without it FileNotFoundError
    replaces the rename failure. The patched `os.replace` deletes the file.
    """
    target = tmp_path / "out.bin"

    def vanishing_replace(src: object, dst: object) -> None:
        os.unlink(str(src))
        raise OSError("rename refused after the temporary went away")

    real_replace = os.replace
    os.replace = vanishing_replace  # type: ignore[assignment]
    try:
        # FileNotFoundError is itself an OSError, so the message rather than the
        # type is what separates the real error from a masked one.
        with pytest.raises(OSError, match="rename refused after the temporary went away"):
            atomic_write_bytes(target, b"payload")
    finally:
        os.replace = real_replace  # type: ignore[assignment]

    assert not list(tmp_path.glob("*.tmp")), "and nothing is left behind either way"


def test_text_is_written_with_lf_endings_on_every_platform(tmp_path: Path) -> None:
    target = atomic_write_text(tmp_path / "out.txt", "a\nb\n")
    assert target.read_bytes() == b"a\nb\n", "a CRLF here would move every text digest"


def test_writing_crlf_text_still_lands_as_lf(tmp_path: Path) -> None:
    target = atomic_write_text(tmp_path / "out.txt", "a\r\nb\r\n")
    assert b"\r\n" not in target.read_bytes()


# ---------------------------------------------------------------------------
# Canonical JSON and JSONL
# ---------------------------------------------------------------------------
def test_canonical_json_sorts_keys_so_two_equal_payloads_render_alike() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_a_non_finite_value_becomes_null_because_json_has_no_token_for_it() -> None:
    """A non-finite value renders as `null`.

    RFC 8259 has no token for NaN or Infinity.
    """
    rendered = canonical_json({"x": math.nan, "y": math.inf, "z": -math.inf}, indent=None)
    assert json.loads(rendered) == {"x": None, "y": None, "z": None}
    assert "NaN" not in rendered
    assert "Infinity" not in rendered


def test_non_finite_values_are_replaced_at_every_depth(tmp_path: Path) -> None:
    payload = {"a": [{"b": [math.nan]}], "c": {"d": {"e": math.inf}}}
    assert json.loads(canonical_json(payload, indent=None)) == {
        "a": [{"b": [None]}],
        "c": {"d": {"e": None}},
    }


def test_a_jsonl_round_trip_preserves_every_record(tmp_path: Path) -> None:
    records = [{"i": i, "name": f"d{i}"} for i in range(4)]
    path = write_jsonl(tmp_path / "out.jsonl", records)
    assert list(read_jsonl(path)) == records


def test_reading_jsonl_skips_blank_lines_between_records(tmp_path: Path) -> None:
    """A blank line is not a record, and it must not become an empty dict."""
    path = tmp_path / "gappy.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_an_empty_jsonl_file_yields_no_records(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert list(read_jsonl(path)) == []


def test_writing_no_records_produces_an_empty_file_not_a_missing_one(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "none.jsonl", [])
    assert path.is_file()
    assert path.read_bytes() == b""


def test_write_json_lands_canonical_bytes(tmp_path: Path) -> None:
    path = write_json(tmp_path / "out.json", {"b": 1, "a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2, "b": 1}


# ---------------------------------------------------------------------------
# strip_volatile
# ---------------------------------------------------------------------------
def test_volatile_keys_are_dropped_so_two_identical_runs_hash_alike() -> None:
    a = {"payload": {"x": 1}, "timestamp": "2026-01-01", "hostname": "one"}
    b = {"payload": {"x": 1}, "timestamp": "2026-06-30", "hostname": "two"}
    assert strip_volatile(a) == strip_volatile(b)
    assert hash_json(strip_volatile(a)) == hash_json(strip_volatile(b))


def test_stripping_reaches_every_depth_not_just_the_top_level() -> None:
    payload = {"outer": {"inner": {"timestamp": "now", "kept": 1}}}
    assert strip_volatile(payload) == {"outer": {"inner": {"kept": 1}}}


def test_stripping_reaches_inside_lists() -> None:
    payload = {"runs": [{"timestamp": "now", "value": 1}, {"timestamp": "later", "value": 2}]}
    assert strip_volatile(payload) == {"runs": [{"value": 1}, {"value": 2}]}


def test_extra_keys_can_be_stripped_without_editing_the_default_set() -> None:
    payload = {"keep": 1, "drop": 2}
    assert strip_volatile(payload, extra=("drop",)) == {"keep": 1}


def test_a_payload_with_nothing_volatile_survives_unchanged() -> None:
    payload = {"a": 1, "b": [1, 2], "c": {"d": "e"}}
    assert strip_volatile(payload) == payload


def test_a_scalar_payload_is_returned_as_is() -> None:
    assert strip_volatile(7) == 7
    assert strip_volatile("text") == "text"
    assert strip_volatile(None) is None


# ---------------------------------------------------------------------------
# The two canonical renderings, and where they disagree
# ---------------------------------------------------------------------------

# `canonical_json` writes non-finite floats as `null`, `hash_json` emits `NaN`;
# they agree on finite payloads and disagree on an undefined margin.
# Rationale: docs/engineering_notes.md#two-canonical-renderings
_UNDEFINED_PAYLOAD = {"tau": 1e-9, "median_margin": math.nan}


def test_the_written_form_and_the_hashed_form_agree_on_every_finite_payload() -> None:
    """The written form and the hashed form agree on a finite payload.

    This is the control for the divergence tests below.
    """
    finite = {"tau": 1e-9, "k": 10, "name": "profile", "ratios": [0.5, 1.0]}
    assert hash_text(canonical_json(finite, indent=None)) == hash_json(finite)


def test_the_two_renderings_disagree_once_a_value_is_not_finite() -> None:
    """The two renderings disagree once a value is not finite.

    `canonical_json` writes `null`; `hash_json` hashes the `NaN` token.
    Neither raises.
    """
    written = canonical_json(_UNDEFINED_PAYLOAD, indent=None)
    assert '"median_margin":null' in written

    assert hash_text(written) != hash_json(_UNDEFINED_PAYLOAD), (
        "the file digest and the payload digest agreed; the divergence is gone"
    )


def test_the_hashed_form_emits_a_token_no_strict_reader_accepts() -> None:
    """The written form parses strictly; a document holding `NaN` does not.

    `json.loads` accepts the `NaN` token `hash_json` renders by default, so
    the divergence is visible only to a strict parser such as this hook.
    """
    strict = {"parse_constant": _refuse_constant}

    assert json.loads(canonical_json(_UNDEFINED_PAYLOAD, indent=None), **strict) == {
        "tau": 1e-9,
        "median_margin": None,
    }

    with pytest.raises(ValueError, match="NaN"):
        json.loads('{"m":NaN}', **strict)


def test_a_manifest_holding_an_undefined_value_verifies_against_its_own_file(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """The digest a manifest embeds is taken over the bytes the file holds.

    Verification recomputes it from the file, so `digest()` renders with
    `canonical_json` like `write()`. `hash_json` would emit the non-standard
    `NaN` token where the file holds `null`."""
    manifest = RunManifest("stability_profile", parameters=dict(_UNDEFINED_PAYLOAD))
    path = tmp_path / "manifest.json"
    recorded = manifest.write(path)["manifest_digest"]

    loaded = json.loads(path.read_text(encoding="utf-8"))
    loaded.pop("manifest_digest")
    payload = _as_the_manifest_hashes_it(loaded)

    assert recorded == hash_text(canonical_json(payload, indent=None))
    assert loaded["parameters"]["median_margin"] is None, "the file itself holds null"

    # The in-memory payload still holds the NaN that the file renders as null,
    # so digesting it with `hash_json` gives a different hex string.
    in_memory = strip_volatile(manifest.to_dict(), extra=manifest._MACHINE_KEYS)
    in_memory.pop("notes", None)
    assert hash_json(in_memory) != recorded


def test_the_two_renderers_agree_on_every_finite_payload() -> None:
    """`canonical_json(..., indent=None)` and `hash_json` agree on finite input.

    The two differ only by the non-finite sanitiser, so a payload of finite
    numbers, strings and containers renders to the same bytes either way."""
    finite = {
        "tau": 1e-9,
        "ks": [5, 10, 20],
        "name": "café",
        "nested": {"b": 2, "a": 1},
        "flag": True,
        "absent": None,
    }
    assert hash_json(finite) == hash_text(canonical_json(finite, indent=None))


def _as_the_manifest_hashes_it(loaded: dict[str, object]) -> object:
    """Apply the exact stripping rule `RunManifest.digest` applies.

    The rule is reproduced rather than calling `digest()`, so the tests model an
    outside verifier working from the written file. `_MACHINE_KEYS` is read from
    the class, so a key added there is covered here without an edit."""
    payload = strip_volatile(loaded, extra=RunManifest._MACHINE_KEYS)
    assert isinstance(payload, dict)
    payload.pop("notes", None)
    return payload


def test_a_manifest_of_finite_values_does_verify_against_its_own_file(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """A manifest of finite values verifies against its own file.

    This is the control for the non-finite round trip above.
    """
    manifest = RunManifest("stability_profile", parameters={"tau": 1e-9, "ks": [5, 10]})
    path = tmp_path / "manifest.json"
    recorded = manifest.write(path)["manifest_digest"]

    loaded = json.loads(path.read_text(encoding="utf-8"))
    loaded.pop("manifest_digest")
    payload = _as_the_manifest_hashes_it(loaded)

    assert recorded == hash_json(payload)


# ---------------------------------------------------------------------------
# hash_ints: the width is part of the identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [0, 1, 2, 3, 16, -1, None, "8", True])
def test_a_width_other_than_four_or_eight_is_refused(width: object) -> None:
    """A width other than four or eight raises `KeyError`.

    The format table has two entries; a width falling through to a default
    would digest the same integers into a different string.
    """
    with pytest.raises(KeyError, match=rf"^{re.escape(repr(width))}$"):
        hash_ints([1], width=width)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("width", "value"),
    [(4, 2**31 - 1), (4, -(2**31)), (8, 2**63 - 1), (8, -(2**63))],
)
def test_each_width_accepts_the_whole_range_it_names(width: int, value: int) -> None:
    """Each width accepts both ends of the range it names.

    These are the largest and smallest identifiers each width carries;
    `struct` refuses one past either end.
    """
    assert len(hash_ints([value], width=width)) == 64


@pytest.mark.parametrize(
    ("width", "value"),
    [(4, 2**31), (4, -(2**31) - 1), (8, 2**63), (8, -(2**63) - 1)],
)
def test_one_past_the_range_is_refused_rather_than_truncated(width: int, value: int) -> None:
    """One past either end of the range raises `struct.error`.

    Truncation would fold two distinct identifiers onto one digest.
    """
    # The interpreter's wording differs across the versions CI spans; see the
    # note on the width-domain test above.
    with pytest.raises(struct.error, match=r"format requires|argument out of range"):
        hash_ints([value], width=width)


def test_a_float_width_is_accepted_because_it_equals_an_integer_key() -> None:
    """A float width equal to an integer key finds the table entry.

    The format table is a dict keyed on `int`, and `8.0 == 8` with the same
    hash. `True` equals 1, which is not a key, so a boolean width is refused.
    """
    assert hash_ints([7], width=8.0) == hash_ints([7], width=8)  # type: ignore[arg-type]
    assert hash_ints([7], width=4.0) == hash_ints([7], width=4)  # type: ignore[arg-type]


def test_a_boolean_packs_as_the_integer_it_equals() -> None:
    """A boolean packs as the integer it equals.

    `bool` is an `int`, and a JSON corpus can carry `true` where a count is
    expected.
    """
    assert hash_ints([True, False]) == hash_ints([1, 0])


# ---------------------------------------------------------------------------
# hash_file: the streaming boundary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [_CHUNK - 1, _CHUNK, _CHUNK + 1, 2 * _CHUNK])
def test_a_file_at_the_streaming_boundary_hashes_as_its_bytes(size: int, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A file at the streaming boundary hashes as its bytes.

    Exactly at the boundary the second read returns empty and ends the loop;
    one byte over, it returns a single byte. An off-by-one would hash a prefix.
    """
    data = bytes(range(256)) * (size // 256) + bytes(size % 256)
    path = tmp_path / "blob.bin"
    path.write_bytes(data)

    assert len(data) == size
    assert hash_file(path) == hash_bytes(data)


def test_an_empty_file_hashes_as_nothing_rather_than_failing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An empty file hashes as the digest of no bytes.

    The loop body never runs; a zero-length artefact is a legitimate result.
    """
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert hash_file(path) == hash_bytes(b"")


def test_hashing_a_binary_file_as_text_refuses_rather_than_mangling(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A binary file hashed as text raises `UnicodeDecodeError`.

    `text=True` decodes as UTF-8 first; a digest over replacement characters
    is not produced.
    """
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(UnicodeDecodeError, match="codec can't decode byte 0xff"):
        hash_file(path, text=True)


# ---------------------------------------------------------------------------
# hash_text and hash_floats: what the digest is sensitive to
# ---------------------------------------------------------------------------
def test_text_that_cannot_be_encoded_is_refused() -> None:
    """A lone surrogate raises `UnicodeEncodeError`.

    It has no UTF-8 encoding and can reach here from a corpus read with
    `errors="surrogateescape"`.
    """
    with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
        hash_text("\ud800")


def test_two_unicode_normalisations_of_one_string_hash_differently() -> None:
    """NFC and NFD spellings of one string hash differently.

    The digest is over bytes, not graphemes; preprocessing normalises to NFKC
    before anything is hashed.
    """
    composed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")

    assert composed != decomposed, "the premise: two spellings of one word"
    assert hash_text(composed) != hash_text(decomposed)


def test_the_order_of_floats_is_part_of_their_digest() -> None:
    """The order of floats is part of their digest.

    A digest over a multiset would call two different vectors equal.
    """
    assert hash_floats([1.0, 2.0]) != hash_floats([2.0, 1.0])


def test_hashing_an_iterator_consumes_it_once() -> None:
    """Hashing an iterator consumes it.

    `hash_floats` takes an `Iterable`, so a second call on the same generator
    digests an empty sequence.
    """
    values = iter([1.0, 2.0, 3.0])
    first = hash_floats(values)
    second = hash_floats(values)

    assert first != second
    assert second == hash_floats([]), "the second pass saw nothing at all"


# ---------------------------------------------------------------------------
# short: truncating a digest for display
# ---------------------------------------------------------------------------
def test_a_negative_length_is_refused_rather_than_dropping_from_the_end() -> None:
    """`short` refuses a negative length.

    `digest[:-1]` is a legal slice, so -1 would return almost the whole digest
    where a short prefix is expected. The value reaches log lines and filenames.
    Zero is a length; an over-long one is the whole digest."""
    digest = "a" * 64
    with pytest.raises(ValueError, match="length must be non-negative"):
        short(digest, -1)

    assert len(short(digest, 0)) == 0
    assert len(short(digest, 1)) == 1
    assert short(digest, 999) == digest


# ---------------------------------------------------------------------------
# canonical_json: rendering choices that reach disk
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("indent", [None, 0, 2, -1])
def test_every_indent_carries_the_same_data(indent: int | None) -> None:
    """Every indent renders the same data.

    The compact form is hashed and the readable form is written; the two must
    not differ in content.
    """
    payload = {"b": [1, 2], "a": {"z": 1.5}}
    assert json.loads(canonical_json(payload, indent=indent)) == payload


def test_only_the_compact_form_omits_the_trailing_newline() -> None:
    """Only the compact form omits the trailing newline.

    The written form ends with a newline so a file is POSIX-clean; the hashed
    form does not, because a trailing byte would enter the digest.
    """
    assert canonical_json({"a": 1}, indent=None) == '{"a":1}'
    assert canonical_json({"a": 1}, indent=2).endswith("}\n")


def test_negative_zero_is_written_as_negative_zero() -> None:
    """Negative zero is written as `-0.0`.

    It is finite, so the sanitiser leaves it, and its bit pattern differs
    from `0.0`.
    """
    assert canonical_json({"z": -0.0}, indent=None) == '{"z":-0.0}'


def test_a_value_json_cannot_represent_is_rendered_through_str() -> None:
    """A value JSON cannot represent is rendered through `str`.

    `default=str` means nothing raises mid-experiment; the rendering is lossy,
    so two objects with the same `str` become the same bytes.
    """
    assert canonical_json({"b": b"hi"}, indent=None) == '{"b":"b\'hi\'"}'


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("at the top of a dict", {"m": math.inf}),
        ("inside a list", {"m": [math.nan]}),
        ("inside a nested list", {"m": [[-math.inf]]}),
        ("inside a tuple", {"m": (1.0, math.nan)}),
        ("as a dict value two levels down", {"a": {"b": math.nan}}),
    ],
)
def test_a_non_finite_value_is_replaced_wherever_it_sits(label: str, payload: object) -> None:
    """A non-finite value is replaced in dicts, lists and tuples alike.

    One missed container leaves a document no strict parser can read.
    """
    rendered = canonical_json(payload, indent=None)
    assert "NaN" not in rendered, label
    assert "Infinity" not in rendered, label
    assert json.loads(rendered, parse_constant=_refuse_constant) is not None


def test_a_tuple_becomes_a_list_because_json_has_no_tuple() -> None:
    """A tuple renders as a JSON array.

    A payload that round-trips through a file comes back with lists where it
    had tuples, so an equality comparison against the original fails on type.
    """
    assert canonical_json({"t": (1, 2)}, indent=None) == '{"t":[1,2]}'


# ---------------------------------------------------------------------------
# Atomic writes: the cleanup arm, without mocking anything
# ---------------------------------------------------------------------------
def test_writing_onto_an_existing_directory_leaves_no_temporary_behind(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Writing onto an existing directory leaves no temporary behind.

    The `except BaseException` arm is reached by a real failed rename, with
    nothing patched. The temporary sits in the target's directory, where a
    later glob would find it.
    """
    occupied = tmp_path / "results"
    occupied.mkdir()

    # The exception type is platform-dependent -- PermissionError on Windows,
    # IsADirectoryError on Linux -- so the target's name is what is matched.
    with pytest.raises(OSError, match="results"):
        atomic_write_bytes(occupied, b"payload")

    assert not list(tmp_path.glob("*.tmp")), "a temporary survived the failed rename"
    assert occupied.is_dir(), "and the existing entry is untouched"


@pytest.mark.parametrize(("writer", "empty"), [(atomic_write_bytes, b""), (atomic_write_text, "")])
def test_writing_nothing_produces_an_empty_file_rather_than_no_file(
    writer: object, empty: object, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """Writing nothing produces an empty file, not a missing one.

    A missing file and an empty one differ to the next stage of a pipeline.
    """
    path = tmp_path / "out"
    writer(path, empty)  # type: ignore[operator]
    assert path.is_file()
    assert path.read_bytes() == b""


def test_a_lone_carriage_return_survives_an_atomic_text_write(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A lone carriage return survives an atomic text write.

    Only CRLF is normalised, matching `hash_text`; a bare CR is document data.
    """
    path = tmp_path / "cr.txt"
    atomic_write_text(path, "a\rb\r\nc")
    assert path.read_bytes() == b"a\rb\nc"


# ---------------------------------------------------------------------------
# read_jsonl
# ---------------------------------------------------------------------------
def test_a_record_written_with_an_undefined_value_reads_back_as_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An undefined value written as `null` reads back as `None`, not `NaN`.

    This is the one lossy point in the round trip, and it holds the undefined
    margin G16 requires; a reader checks `is None`, not equality with NaN.
    """
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"doc_id": "d0", "margin": math.nan}])

    (record,) = read_jsonl(path)
    assert record["margin"] is None
    assert not isinstance(record["margin"], float)


def test_a_line_that_is_not_json_is_refused_rather_than_skipped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A line that is not JSON raises `JSONDecodeError`.

    Blank lines are skipped; a corrupt line is not blank, and skipping it
    would shorten a corpus.
    """
    path = tmp_path / "broken.jsonl"
    path.write_bytes(b'{"a": 1}\nnot json at all\n')

    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        list(read_jsonl(path))


def test_a_file_without_a_trailing_newline_still_yields_its_last_record(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A file without a trailing newline still yields its last record.

    A reader that required the terminator would drop one document from a
    hand-edited corpus.
    """
    path = tmp_path / "no_terminator.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}')
    assert [r["a"] for r in read_jsonl(path)] == [1, 2]


def test_whitespace_only_lines_are_skipped_like_blank_ones(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Whitespace-only lines are skipped like blank ones.

    The guard is `line.strip()`, so a line of spaces is as empty as an empty
    one.
    """
    path = tmp_path / "spaced.jsonl"
    path.write_bytes(b'{"a": 1}\n   \n\t\n{"a": 2}\n')
    assert [r["a"] for r in read_jsonl(path)] == [1, 2]


# ---------------------------------------------------------------------------
# strip_volatile
# ---------------------------------------------------------------------------
def test_a_value_that_looks_like_a_volatile_key_is_left_alone() -> None:
    """A value equal to a volatile key name is kept.

    Keys are dropped, not values; a column name of "timestamp" must survive.
    """
    assert strip_volatile({"sort_by": "timestamp"}) == {"sort_by": "timestamp"}


def test_stripping_turns_a_tuple_into_a_list() -> None:
    """Stripping turns a tuple into a list.

    `hash_json` renders both as an array, so the change is invisible in the
    digest and visible in an equality check.
    """
    assert strip_volatile({"ks": (5, 10)}) == {"ks": [5, 10]}


def test_stripping_reaches_a_volatile_key_three_levels_down() -> None:
    """A volatile key three levels down is dropped.

    A non-recursive implementation that handled only the top two levels
    would pass the shallower tests.
    """
    payload = {"a": {"b": {"c": {"timestamp": 1, "keep": 2}}}}
    assert strip_volatile(payload) == {"a": {"b": {"c": {"keep": 2}}}}


@pytest.mark.parametrize("key", sorted(VOLATILE_KEYS))
def test_every_declared_volatile_key_is_actually_dropped(key: str) -> None:
    """Every key in `VOLATILE_KEYS` is dropped.

    A listed name that is not honoured would leave a hostname or a working
    directory inside a published digest.
    """
    assert strip_volatile({key: "x", "keep": 1}) == {"keep": 1}


# ---------------------------------------------------------------------------
# Renderings that are recorded rather than incidental
# ---------------------------------------------------------------------------
def test_a_non_ascii_token_is_hashed_as_itself_rather_than_as_an_escape() -> None:
    """A non-ASCII token is hashed as itself, not as an escape sequence.

    `hash_json` renders with `ensure_ascii=False`, so the digest covers the
    UTF-8 character, not its six-character ASCII escape.
    """
    token = "caf" + chr(0xE9)
    digest = hash_json({"token": token})

    assert digest == hash_text('{"token":"' + token + '"}')
    assert digest != hash_text('{"token":"caf' + chr(92) + 'u00e9"}')


def test_a_non_ascii_value_is_written_to_disk_as_itself(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A non-ASCII value is written to disk as itself.

    A results file is read by people as well as parsers.
    """
    token = "caf" + chr(0xE9)
    path = tmp_path / "out.json"
    write_json(path, {"token": token})

    written = path.read_text(encoding="utf-8")
    assert token in written
    assert chr(92) + "u00e9" not in written


def test_the_readable_form_is_indented_by_two_spaces() -> None:
    """The readable form is indented by two spaces.

    The default indent is part of the bytes the snapshot test compares.
    """
    assert canonical_json({"a": {"b": 1}}) == '{\n  "a": {\n    "b": 1\n  }\n}\n'


def test_a_written_file_uses_the_same_two_space_indent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A written file uses the same two-space indent as `canonical_json`.

    `write_json` carries its own default; the two must not differ.
    """
    path = tmp_path / "out.json"
    write_json(path, {"a": {"b": 1}})
    assert path.read_text(encoding="utf-8") == canonical_json({"a": {"b": 1}})
    assert '\n  "a"' in path.read_text(encoding="utf-8")


def test_an_absent_native_block_is_the_reference_and_certifies() -> None:
    """An absent or null `native` block is the pure-Python reference.

    No build flags are involved, so the reference is reproducible on its own;
    refusing it would refuse every result from a machine without a compiler
    (CLI side: `test_a_model_built_without_a_native_backend_verifies_as_reproducible`).
    """
    for environment in ({}, {"native": None}):
        manifest = RunManifest("stability_profile", environment=environment)
        assert manifest.is_reproducible_build is True
        manifest.require_reproducible()


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ({"reproducible": True}, True),
        ({"reproducible": False}, False),
        ({}, False),
        ("not-a-mapping", False),
        (42, False),
    ],
)
def test_the_native_block_must_say_reproducible_to_certify(native: object, expected: bool) -> None:
    """A present `native` block certifies only when it says `reproducible`.

    A block that is not a mapping, or that omits the flag, is not a claim of
    reproducibility; a non-mapping raises `RuntimeError`. An absent block is
    the reference and certifies instead."""
    manifest = RunManifest("stability_profile", environment={"native": native})
    if not isinstance(native, dict):
        with pytest.raises(RuntimeError, match="not a mapping"):
            manifest.require_reproducible()
    assert manifest.is_reproducible_build is expected
