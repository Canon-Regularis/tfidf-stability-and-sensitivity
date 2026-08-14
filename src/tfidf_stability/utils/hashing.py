"""Canonical hashing.

Tracing a published number back to its data, config and build needs "the same
input" to have one digest, so this module fixes one way to hash each kind of
thing and everything else uses it.

Hash bytes rather than renderings: a float goes through its raw binary64 bit
pattern. ``repr`` is lossless in CPython today, but that is a language guarantee
rather than a file-format one, and the bit pattern keeps a one-ulp difference
visible.

Hash a canonical form: JSON with sorted keys and no incidental whitespace, text
normalised to LF so a Windows checkout and a Linux checkout of the same file
agree. ``.gitattributes`` normalises on write, but a digest taken before that
would not.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "hash_bytes",
    "hash_file",
    "hash_floats",
    "hash_ints",
    "hash_json",
    "hash_text",
    "short",
]

_CHUNK = 1 << 20


def hash_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes, as lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 of text, normalised to LF and encoded UTF-8.

    The repository claims byte-identical results across Linux, macOS and Windows,
    which a digest varying with the checkout platform would break.
    """
    return hash_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def hash_file(path: Path | str, *, text: bool = False) -> str:
    """SHA-256 of a file.

    Args:
        path: File to hash.
        text: Normalise line endings before hashing. Use for anything tracked as
            text; leave off for binaries, where normalisation would corrupt.
    """
    p = Path(path)
    if text:
        return hash_text(p.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def hash_floats(values: Iterable[float]) -> str:
    """SHA-256 over the raw binary64 bit patterns.

    A digest over ``repr`` would depend on the interpreter's float formatting and
    could collide two values differing in the last bit, the difference this
    project exists to detect.
    """
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def hash_ints(values: Iterable[int], *, width: int = 8) -> str:
    """SHA-256 over fixed-width little-endian integers."""
    fmt = {4: "<i", 8: "<q"}[width]
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack(fmt, value))
    return digest.hexdigest()


def hash_json(payload: Any) -> str:
    """SHA-256 over a canonical JSON rendering.

    Sorted keys, no incidental whitespace, and ``ensure_ascii=False`` so a
    non-ASCII token digests the same whether or not it arrived escaped. Two
    configs differing only in key order share a digest.
    """
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hash_text(blob)


def short(digest: str, length: int = 12) -> str:
    """A truncated digest, for log lines and filenames.

    Never for identity comparison: 12 hex characters is 48 bits, enough to
    eyeball and far too few to rely on.
    """
    return digest[:length]


def hash_manifest_lines(entries: Sequence[tuple[str, str]]) -> str:
    """Digest a ``(name, digest)`` listing, as used by the vendored-asset manifests."""
    return hash_text("".join(f"{d}  {n}\n" for n, d in sorted(entries)))
