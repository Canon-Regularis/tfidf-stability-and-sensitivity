"""Deterministic, atomic file IO.

The snapshot test hashes output files, so written bytes carry no timestamps,
absolute paths, hostname or locale-dependent formatting. Writes are atomic: an
interrupted run leaves the previous file or the new one, never a partial file.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "canonical_json",
    "read_jsonl",
    "strip_volatile",
    "write_json",
    "write_jsonl",
]

#: Keys stripped before a report is hashed. Useful to a human reading the file
#: and fatal to a test hashing it, so they are written and then excluded.
VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "timestamp",
        "created_at",
        "duration_seconds",
        "elapsed",
        "hostname",
        "username",
        "cwd",
        # Provenance blocks write this key; `datasets/loaders.py` stores the
        # corpus file path in it for a jsonl corpus, absolute when given so.
        # Stripping it keeps a manifest digest independent of the corpus
        # location, as `RunManifest.digest` requires.
        "path",
        "pid",
    }
)


def _json_safe(payload: Any) -> Any:
    """Replace non-finite floats with ``null``, recursively.

    RFC 8259 has no token for ``NaN`` or ``Infinity``; strict parsers reject both. Non-finite
    means undefined (G16); ``defined``, ``n_nan`` and ``*_hex`` fields keep undefined,
    not-applicable and infinite apart. Rationale: docs/engineering_notes.md#non-finite-json
    """
    if isinstance(payload, float):
        return None if not math.isfinite(payload) else payload
    if isinstance(payload, dict):
        return {k: _json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(v) for v in payload]
    return payload


def canonical_json(payload: Any, *, indent: int | None = 2) -> str:
    """Render JSON canonically: sorted keys, LF endings, UTF-8 literals.

    ``indent=None`` is the compact hashing form, the default the readable disk form; the two
    differ only in whitespace. Non-finite floats become ``null`` (:func:`_json_safe`);
    ``allow_nan`` stays on so a value the sanitiser misses is written, not raised.
    """
    text = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else (",", ": "),
        default=str,
    )
    return text if indent is None else text + "\n"


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Write bytes atomically via a temporary file in the same directory.

    Same directory because ``os.replace`` is atomic only within a filesystem, and
    the system temp directory may be on another one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Write text atomically, always with LF endings.

    ``.gitattributes`` normalises tracked files on write, so emitting CRLF on
    Windows would make a freshly generated file differ from the committed one and
    break the snapshot comparison.
    """
    return atomic_write_bytes(path, text.replace("\r\n", "\n").encode("utf-8"))


def write_json(path: Path | str, payload: Any, *, indent: int | None = 2) -> Path:
    """Write canonical JSON atomically."""
    return atomic_write_text(path, canonical_json(payload, indent=indent))


def write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write JSON Lines, one canonical compact object per line."""
    body = "".join(canonical_json(record, indent=None) + "\n" for record in records)
    return atomic_write_text(path, body)


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Read JSON Lines, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def strip_volatile(payload: Any, extra: Iterable[str] = ()) -> Any:
    """Recursively drop keys whose value varies between identical runs.

    Applied before hashing a report, so the digest covers the results alone.
    Recursive because manifests nest: a timestamp two levels down breaks a
    snapshot as thoroughly as one at the top.
    """
    # `extra` is materialised once and the frozenset recurses, so a generator
    # argument, exhausted after one pass, still strips keys at every depth.
    drop = VOLATILE_KEYS | frozenset(extra)
    return _strip(payload, drop)


def _strip(payload: Any, drop: frozenset[str]) -> Any:
    """Recurse with the key set already resolved."""
    if isinstance(payload, Mapping):
        return {k: _strip(v, drop) for k, v in payload.items() if k not in drop}
    if isinstance(payload, (list, tuple)):
        return [_strip(v, drop) for v in payload]
    return payload
