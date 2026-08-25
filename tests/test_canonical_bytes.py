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
    document Python writes is a document Python reads -- and the fact that no
    other parser would is invisible unless a test refuses them explicitly.
    """
    raise ValueError(name)


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
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
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


# ---------------------------------------------------------------------------
# The two canonical renderings, and where they disagree
# ---------------------------------------------------------------------------
# There are two paths from a payload to bytes and they are not the same path.
# `canonical_json` sanitises non-finite floats to `null` before rendering, so
# what lands on disk is strict JSON. `hash_json` does not: it calls `json.dumps`
# directly, and Python's encoder emits the non-standard `NaN` and `Infinity`
# tokens by default.
#
# For every finite payload the two agree, which is why nothing had noticed. For a
# payload holding an undefined margin -- which G16 *requires* be reported as
# undefined rather than coerced -- they disagree, and the digest of a file stops
# matching the digest of the thing written into it.
_UNDEFINED_PAYLOAD = {"tau": 1e-9, "median_margin": math.nan}


def test_the_written_form_and_the_hashed_form_agree_on_every_finite_payload() -> None:
    """The control. Without it the divergence below reads as a general fact
    about the two functions rather than as a property of non-finite values."""
    finite = {"tau": 1e-9, "k": 10, "name": "profile", "ratios": [0.5, 1.0]}
    assert hash_text(canonical_json(finite, indent=None)) == hash_json(finite)


def test_the_two_renderings_disagree_once_a_value_is_not_finite() -> None:
    """`canonical_json` writes `null`; `hash_json` hashes the `NaN` token. Same
    payload, two different digests, and no error from either."""
    written = canonical_json(_UNDEFINED_PAYLOAD, indent=None)
    assert '"median_margin":null' in written

    assert hash_text(written) != hash_json(_UNDEFINED_PAYLOAD), (
        "the file digest and the payload digest agreed; the divergence is gone"
    )


def test_the_hashed_form_emits_a_token_no_strict_reader_accepts() -> None:
    """The reason the sanitiser exists at all. `json.loads` reads its own
    extension back, so the divergence is invisible from Python; every other
    parser rejects the document."""
    strict = {"parse_constant": _refuse_constant}

    assert json.loads(canonical_json(_UNDEFINED_PAYLOAD, indent=None), **strict) == {
        "tau": 1e-9,
        "median_margin": None,
    }

    with pytest.raises(ValueError, match="NaN"):
        json.loads('{"m":NaN}', **strict)


def test_a_manifest_holding_an_undefined_value_cannot_verify_against_its_own_file(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """The consequence, pinned rather than repaired.

    `RunManifest.digest()` goes through `hash_json`, and `write()` goes through
    `canonical_json`. So the `manifest_digest` embedded in the file is taken over
    bytes the file does not contain, and a verifier that reads the manifest back
    and recomputes the digest -- which is precisely what
    ``docs/experiments.md`` describes as the check on a published number --
    gets a different hex string.

    Not reachable from a manifest today: the hashed fields carry `tau`, the `k`
    set and artefact digests, all finite or textual. It is one undefined margin
    away from being live, and the sanitiser's own docstring records that both
    experiment result files did contain `NaN`.
    """
    manifest = RunManifest("stability_profile", parameters=dict(_UNDEFINED_PAYLOAD))
    path = tmp_path / "manifest.json"
    recorded = manifest.write(path)["manifest_digest"]

    loaded = json.loads(path.read_text(encoding="utf-8"))
    loaded.pop("manifest_digest")
    payload = _as_the_manifest_hashes_it(loaded)

    assert recorded != hash_json(payload), "the manifest now verifies; update this test"
    assert loaded["parameters"]["median_margin"] is None, "the file itself holds null"


def _as_the_manifest_hashes_it(loaded: dict[str, object]) -> object:
    """Apply the exact stripping rule `RunManifest.digest` applies.

    Local by house convention, and shared between the two tests around it so
    they cannot drift apart. It reproduces the rule rather than calling
    `digest()`, because what these tests model is an outside verifier working
    from the written file -- calling the method would assert only that it equals
    itself.

    `_MACHINE_KEYS` is read from the class rather than restated, so a key added
    there is covered here without an edit. Its absence was not a cosmetic
    mismatch: without it the sibling above would still report `!=` and would
    have gone on passing for the wrong reason, recording a machine identity
    where it means to record the NaN round trip.
    """
    payload = strip_volatile(loaded, extra=RunManifest._MACHINE_KEYS)
    assert isinstance(payload, dict)
    payload.pop("notes", None)
    return payload


def test_a_manifest_of_finite_values_does_verify_against_its_own_file(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """The same round trip on ordinary parameters, so the failure above is
    attributable to the non-finite value and not to the round trip."""
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
    """The format table has two entries. A width that fell through to a default
    would digest the same integers into a different string, silently breaking
    every comparison against a previously recorded digest."""
    with pytest.raises(KeyError, match=rf"^{re.escape(repr(width))}$"):
        hash_ints([1], width=width)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("width", "value"),
    [(4, 2**31 - 1), (4, -(2**31)), (8, 2**63 - 1), (8, -(2**63))],
)
def test_each_width_accepts_the_whole_range_it_names(width: int, value: int) -> None:
    """Both ends, exactly. `struct` refuses one past either, so these are the
    largest and smallest identifiers a corpus can carry at each width."""
    assert len(hash_ints([value], width=width)) == 64


@pytest.mark.parametrize(
    ("width", "value"),
    [(4, 2**31), (4, -(2**31) - 1), (8, 2**63), (8, -(2**63) - 1)],
)
def test_one_past_the_range_is_refused_rather_than_truncated(width: int, value: int) -> None:
    """Truncation would fold two distinct identifiers onto one digest, which is
    the one failure a content hash exists to prevent."""
    # The interpreter's wording differs across the versions CI spans; see the
    # note on the width-domain test above.
    with pytest.raises(struct.error, match=r"format requires|argument out of range"):
        hash_ints([value], width=width)


def test_a_float_width_is_accepted_because_it_equals_an_integer_key() -> None:
    """The format table is a dict keyed on `int`, and `8.0 == 8` with the same
    hash, so a width arriving as a float from a parsed config finds the entry.

    Pinned rather than guarded, and worth knowing in both directions: `True`
    equals 1, which is *not* a key, so a boolean width is refused while a float
    one is not.
    """
    assert hash_ints([7], width=8.0) == hash_ints([7], width=8)  # type: ignore[arg-type]
    assert hash_ints([7], width=4.0) == hash_ints([7], width=4)  # type: ignore[arg-type]


def test_a_boolean_packs_as_the_integer_it_equals() -> None:
    """`bool` is an `int`, so `True` digests as 1. Pinned because a JSON corpus
    can carry `true` where a count is expected."""
    assert hash_ints([True, False]) == hash_ints([1, 0])


# ---------------------------------------------------------------------------
# hash_file: the streaming boundary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [_CHUNK - 1, _CHUNK, _CHUNK + 1, 2 * _CHUNK])
def test_a_file_at_the_streaming_boundary_hashes_as_its_bytes(size: int, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The read loop is `while chunk := handle.read(_CHUNK)`. Exactly at the
    boundary the second read returns empty and ends the loop; one byte over, it
    returns a single byte. An off-by-one there would silently hash a prefix.
    """
    data = bytes(range(256)) * (size // 256) + bytes(size % 256)
    path = tmp_path / "blob.bin"
    path.write_bytes(data)

    assert len(data) == size
    assert hash_file(path) == hash_bytes(data)


def test_an_empty_file_hashes_as_nothing_rather_than_failing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The loop body never runs. The digest of no bytes is a real digest, and a
    zero-length artefact is a legitimate result."""
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert hash_file(path) == hash_bytes(b"")


def test_hashing_a_binary_file_as_text_refuses_rather_than_mangling(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`text=True` decodes as UTF-8 first. A binary file is not text, and the
    refusal is better than a digest over replacement characters."""
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(UnicodeDecodeError, match="codec can't decode byte 0xff"):
        hash_file(path, text=True)


# ---------------------------------------------------------------------------
# hash_text and hash_floats: what the digest is sensitive to
# ---------------------------------------------------------------------------
def test_text_that_cannot_be_encoded_is_refused() -> None:
    """A lone surrogate has no UTF-8 encoding. It can reach here from a corpus
    read with `errors="surrogateescape"`, so the refusal is the boundary between
    "text this project can hash" and "bytes pretending to be text"."""
    with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
        hash_text("\ud800")


def test_two_unicode_normalisations_of_one_string_hash_differently() -> None:
    """The digest is over bytes, not over graphemes. Preprocessing normalises to
    NFKC precisely so this difference is resolved before anything is hashed --
    the hash itself does not resolve it."""
    composed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")

    assert composed != decomposed, "the premise: two spellings of one word"
    assert hash_text(composed) != hash_text(decomposed)


def test_the_order_of_floats_is_part_of_their_digest() -> None:
    """A digest over a multiset would call two different vectors equal. These
    are sequences, and the sequence is the thing."""
    assert hash_floats([1.0, 2.0]) != hash_floats([2.0, 1.0])


def test_hashing_an_iterator_consumes_it_once() -> None:
    """`hash_floats` takes an `Iterable`, so a generator is spent by the call.
    Pinned because hashing the same generator twice to compare digests would
    silently compare a full sequence against an empty one."""
    values = iter([1.0, 2.0, 3.0])
    first = hash_floats(values)
    second = hash_floats(values)

    assert first != second
    assert second == hash_floats([]), "the second pass saw nothing at all"


# ---------------------------------------------------------------------------
# short: a display helper with a slicing trap
# ---------------------------------------------------------------------------
def test_a_negative_length_drops_from_the_end_instead_of_truncating() -> None:
    """`digest[:-1]` is a legal slice, so a length arriving as -1 returns almost
    the whole digest rather than an empty string or an error. Pinned: the
    function is documented as "never for identity comparison", and this is the
    shape that most looks like a full digest while not being one.
    """
    digest = "a" * 64
    assert len(short(digest, -1)) == 63
    assert len(short(digest, 0)) == 0
    assert short(digest, 999) == digest


# ---------------------------------------------------------------------------
# canonical_json: rendering choices that reach disk
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("indent", [None, 0, 2, -1])
def test_every_indent_carries_the_same_data(indent: int | None) -> None:
    """Indentation is presentation. The compact form is what gets hashed and the
    readable form is what gets written, and they must not differ in content."""
    payload = {"b": [1, 2], "a": {"z": 1.5}}
    assert json.loads(canonical_json(payload, indent=indent)) == payload


def test_only_the_compact_form_omits_the_trailing_newline() -> None:
    """The written form ends with a newline so a file is POSIX-clean; the hashed
    form does not, because a trailing byte would be in the digest."""
    assert canonical_json({"a": 1}, indent=None) == '{"a":1}'
    assert canonical_json({"a": 1}, indent=2).endswith("}\n")


def test_negative_zero_is_written_as_negative_zero() -> None:
    """It is finite, so the sanitiser leaves it, and `-0.0` is a different bit
    pattern from `0.0` that the rest of the project compares on."""
    assert canonical_json({"z": -0.0}, indent=None) == '{"z":-0.0}'


def test_a_value_json_cannot_represent_is_rendered_through_str() -> None:
    """`default=str` means nothing raises mid-experiment. It also means the
    rendering is lossy and one-way, so two distinct objects with the same `str`
    become the same bytes."""
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
    """The sanitiser recurses through dicts, lists and tuples alike. One missed
    container leaves a document no strict parser can read."""
    rendered = canonical_json(payload, indent=None)
    assert "NaN" not in rendered, label
    assert "Infinity" not in rendered, label
    assert json.loads(rendered, parse_constant=_refuse_constant) is not None


def test_a_tuple_becomes_a_list_because_json_has_no_tuple() -> None:
    """A type change the caller does not choose. Worth stating: a payload that
    round-trips through a file comes back with lists where it had tuples, so an
    equality comparison against the original fails on type alone."""
    assert canonical_json({"t": (1, 2)}, indent=None) == '{"t":[1,2]}'


# ---------------------------------------------------------------------------
# Atomic writes: the cleanup arm, without mocking anything
# ---------------------------------------------------------------------------
def test_writing_onto_an_existing_directory_leaves_no_temporary_behind(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The `except BaseException` arm, reached by a real failure rather than a
    patched `os.replace`.

    The temporary is created in the target's own directory, so a rename that
    fails and does not clean up leaves a `.tmp` file beside the real outputs --
    where the next run's globbing would find it.
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
    """An empty result is a result. A missing file and an empty one are
    different things to the next stage of a pipeline."""
    path = tmp_path / "out"
    writer(path, empty)  # type: ignore[operator]
    assert path.is_file()
    assert path.read_bytes() == b""


def test_a_lone_carriage_return_survives_an_atomic_text_write(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Only CRLF is normalised, matching `hash_text`. A bare CR is data -- it
    can appear inside a document -- and rewriting it would change the corpus."""
    path = tmp_path / "cr.txt"
    atomic_write_text(path, "a\rb\r\nc")
    assert path.read_bytes() == b"a\rb\nc"


# ---------------------------------------------------------------------------
# read_jsonl
# ---------------------------------------------------------------------------
def test_a_record_written_with_an_undefined_value_reads_back_as_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The round trip is lossy in exactly one place, and it is the place G16
    cares about: an undefined margin goes out as `null` and comes back as
    `None`, not as `NaN`. A reader comparing against `float("nan")` finds
    nothing; a reader checking `is None` finds it."""
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"doc_id": "d0", "margin": math.nan}])

    (record,) = read_jsonl(path)
    assert record["margin"] is None
    assert not isinstance(record["margin"], float)


def test_a_line_that_is_not_json_is_refused_rather_than_skipped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Blank lines are skipped deliberately; a corrupt line is not blank.
    Skipping it would silently shorten a corpus."""
    path = tmp_path / "broken.jsonl"
    path.write_bytes(b'{"a": 1}\nnot json at all\n')

    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        list(read_jsonl(path))


def test_a_file_without_a_trailing_newline_still_yields_its_last_record(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Iterating a file yields the final partial line. A reader that required
    the terminator would drop one document from a hand-edited corpus."""
    path = tmp_path / "no_terminator.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}')
    assert [r["a"] for r in read_jsonl(path)] == [1, 2]


def test_whitespace_only_lines_are_skipped_like_blank_ones(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The guard is `line.strip()`, so a line of spaces is as empty as an empty
    one -- which is what a hand-edited file tends to contain."""
    path = tmp_path / "spaced.jsonl"
    path.write_bytes(b'{"a": 1}\n   \n\t\n{"a": 2}\n')
    assert [r["a"] for r in read_jsonl(path)] == [1, 2]


# ---------------------------------------------------------------------------
# strip_volatile
# ---------------------------------------------------------------------------
def test_a_value_that_looks_like_a_volatile_key_is_left_alone() -> None:
    """Keys are dropped, not values. A configuration recording the string
    "timestamp" as a column name must survive into the digest."""
    assert strip_volatile({"sort_by": "timestamp"}) == {"sort_by": "timestamp"}


def test_stripping_turns_a_tuple_into_a_list() -> None:
    """A type change the caller does not ask for, and one that matters: the
    stripped payload is what gets hashed, and `hash_json` renders both as an
    array, so this is invisible in the digest and visible in an equality check.
    """
    assert strip_volatile({"ks": (5, 10)}) == {"ks": [5, 10]}


def test_stripping_reaches_a_volatile_key_three_levels_down() -> None:
    """Manifests nest. The existing tests reach two levels; three is where a
    non-recursive implementation that special-cased the top two would pass."""
    payload = {"a": {"b": {"c": {"timestamp": 1, "keep": 2}}}}
    assert strip_volatile(payload) == {"a": {"b": {"c": {"keep": 2}}}}


@pytest.mark.parametrize("key", sorted(VOLATILE_KEYS))
def test_every_declared_volatile_key_is_actually_dropped(key: str) -> None:
    """The set is the contract. A name listed but not honoured would leave a
    hostname or a working directory inside a published digest."""
    assert strip_volatile({key: "x", "keep": 1}) == {"keep": 1}


# ---------------------------------------------------------------------------
# Renderings that are recorded rather than incidental
# ---------------------------------------------------------------------------
def test_a_non_ascii_token_is_hashed_as_itself_rather_than_as_an_escape() -> None:
    """`ensure_ascii=False` is why a token digests the same however it arrived.

    With ASCII escaping on, the accented character renders as a six-character
    ASCII escape sequence instead, so a config written by a tool that escapes and
    one written by a tool that does not would carry different digests for the
    same vocabulary.
    """
    token = "caf" + chr(0xE9)
    digest = hash_json({"token": token})

    assert digest == hash_text('{"token":"' + token + '"}')
    assert digest != hash_text('{"token":"caf' + chr(92) + 'u00e9"}')


def test_a_non_ascii_value_is_written_to_disk_as_itself(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The same choice on the writing side. A results file is read by people as
    well as parsers, and an escaped token is unrecognisable in both."""
    token = "caf" + chr(0xE9)
    path = tmp_path / "out.json"
    write_json(path, {"token": token})

    written = path.read_text(encoding="utf-8")
    assert token in written
    assert chr(92) + "u00e9" not in written


def test_the_readable_form_is_indented_by_two_spaces() -> None:
    """The default is what every results file on disk is written with, so it is
    part of the bytes the snapshot test compares -- not a formatting
    preference."""
    assert canonical_json({"a": {"b": 1}}) == '{\n  "a": {\n    "b": 1\n  }\n}\n'


def test_a_written_file_uses_the_same_two_space_indent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`write_json` carries its own default. If the two drifted apart, a file
    written through one path would not match one written through the other."""
    path = tmp_path / "out.json"
    write_json(path, {"a": {"b": 1}})
    assert path.read_text(encoding="utf-8") == canonical_json({"a": {"b": 1}})
    assert '\n  "a"' in path.read_text(encoding="utf-8")
