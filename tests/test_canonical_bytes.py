"""What bytes a run emits, and what digest names them.

`utils/hashing.py` and `utils/io.py` decide the two things every reproducibility
claim in this repository rests on: the exact bytes written to disk, and the digest
that identifies them. Neither had an owning test file. `hashing.py` sat at 67%,
the lowest non-zero figure in the package, with three whole function bodies never
executed.

The properties hunted are the ones that would let a digest agree when the values
behind it do not.

A digest over floats must see the bit pattern. `hash_floats` packs `<d` rather
than formatting, because a digest over `repr` would depend on the interpreter's
float formatting and could collide two values differing in the last bit, which is
the difference this whole project exists to detect.

Text hashing must not vary with the checkout platform. `hash_text` normalises
CRLF to LF, so a digest computed on Windows equals one computed on Linux for the
same tracked file. A binary must not get that treatment, because normalisation
would corrupt it; the two paths are separate for that reason and only the text
one was ever exercised.

A write is atomic or it does not happen. `atomic_write_bytes` writes to a
temporary file in the destination directory, fsyncs, then renames. If anything
fails in between, the temporary file is removed and the original is left intact.
A half-written report that still parses is worse than no report.

Fault injection here patches the OS boundary (`os.replace`, `os.fsync`) and never
a function inside this package, so the test cannot pass by agreeing with a mock
of the thing under test.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import pytest

from tfidf_stability.utils.hashing import (
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
    atomic_write_bytes,
    atomic_write_text,
    canonical_json,
    read_jsonl,
    strip_volatile,
    write_json,
    write_jsonl,
)

_SHA256_OF_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ---------------------------------------------------------------------------
# Digests: normal
# ---------------------------------------------------------------------------
def test_hashing_empty_bytes_gives_the_known_sha256_of_nothing() -> None:
    """A fixed external constant, so the whole family is anchored to SHA-256
    rather than merely to itself."""
    assert hash_bytes(b"") == _SHA256_OF_EMPTY


def test_text_hashing_is_the_utf8_bytes_of_the_normalised_text() -> None:
    assert hash_text("abc") == hash_bytes(b"abc")
    assert hash_text("") == _SHA256_OF_EMPTY


def test_crlf_and_lf_text_hash_identically_so_a_checkout_platform_cannot_move_it() -> None:
    """The repository claims byte-identical results across three operating
    systems; a digest varying with the line endings git checked out would break
    that claim without touching a single computed value."""
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
    """The collision a repr-based digest would allow, and the difference this
    project exists to detect."""
    a = 1.0
    b = math.nextafter(1.0, math.inf)
    assert a != b
    assert hash_floats([a]) != hash_floats([b])


def test_positive_and_negative_zero_hash_differently() -> None:
    """`0.0 == -0.0` is True, so a digest comparing values rather than bytes
    would call these the same."""
    assert 0.0 == -0.0
    assert hash_floats([0.0]) != hash_floats([-0.0])


def test_two_distinct_nans_hash_by_their_bits_not_by_comparison() -> None:
    """NaN compares unequal to every value including another NaN, so equality
    cannot group these. The digest works on bytes, so it can."""
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
    with pytest.raises(KeyError):
        hash_ints([1], width=2)


def test_an_integer_beyond_the_chosen_width_is_rejected_rather_than_truncated() -> None:
    """Truncation would make two different corpora hash alike."""
    with pytest.raises(struct.error):
        hash_ints([2**31], width=4)


def test_hashing_no_values_at_all_is_the_digest_of_nothing() -> None:
    assert hash_floats([]) == _SHA256_OF_EMPTY
    assert hash_ints([]) == _SHA256_OF_EMPTY


# ---------------------------------------------------------------------------
# Digests: files and manifests
# ---------------------------------------------------------------------------
def test_hashing_a_file_as_binary_matches_hashing_its_bytes(tmp_path: Path) -> None:
    """The binary branch, which nothing exercised: every existing caller passes
    text=True."""
    target = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 8
    target.write_bytes(payload)
    assert hash_file(target, text=False) == hash_bytes(payload)


def test_a_file_larger_than_one_chunk_hashes_the_same_as_its_bytes(tmp_path: Path) -> None:
    """The read loop runs more than once only above the 1 MiB chunk size, so a
    smaller fixture would leave the loop's second iteration untested."""
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
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "absent", text=False)


def test_a_manifest_digest_is_order_independent_but_content_sensitive() -> None:
    """The listing is sorted before hashing, so the order entries were collected
    in cannot change the identity of the asset set."""
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
    """The cleanup arm. Patched at the OS boundary rather than inside this
    package, so the test cannot pass by agreeing with a mock of the code it is
    about.
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
    """Every published result was unparseable outside Python before this
    existed: RFC 8259 has no NaN or Infinity."""
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
