"""Deterministic file IO.

Everything written by this project has to be byte-reproducible: the snapshot
test hashes output files, and a format that embeds a timestamp or iterates a
dictionary in insertion order would make that test fail for reasons unrelated to
the mathematics.

Two rules:

* **no ambient state in the bytes** -- no timestamps, no absolute paths, no
  hostname, no locale-dependent formatting;
* **atomic writes** -- an interrupted experiment leaves either the previous file
  or the new one, never a half-written file that a later run would hash happily.
"""

from __future__ import annotations

import json
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

#: Keys removed before hashing a report. These are genuinely useful in a file a
#: human reads and genuinely fatal in one a test hashes, so they are written and
#: then excluded rather than omitted.
VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "timestamp",
        "created_at",
        "duration_seconds",
        "elapsed",
        "hostname",
        "username",
        "cwd",
        "output_path",
        "pid",
    }
)


def canonical_json(payload: Any, *, indent: int | None = 2) -> str:
    """Render JSON canonically: sorted keys, LF endings, UTF-8 literals.

    ``indent=None`` gives the compact form used for hashing; the default gives
    the readable form used for files on disk. Both sort keys, so the two differ
    only in whitespace.
    """
    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else (",", ": "),
        default=str,
    )
    return text if indent is None else text + "\n"


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Write bytes atomically via a temporary file in the same directory.

    Same directory because ``os.replace`` is only atomic within a filesystem,
    and a temp directory may well be on another one.
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

    LF unconditionally: ``.gitattributes`` normalises tracked files on write, so
    emitting CRLF on Windows would make a freshly generated file differ from the
    committed one and break the snapshot comparison.
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

    Applied before hashing a report, so the digest covers the *results* and not
    when or where they were produced. Recursive because a manifest nests, and a
    timestamp two levels down would break a snapshot just as effectively as one
    at the top.
    """
    drop = VOLATILE_KEYS | frozenset(extra)
    if isinstance(payload, Mapping):
        return {k: strip_volatile(v, extra) for k, v in payload.items() if k not in drop}
    if isinstance(payload, (list, tuple)):
        return [strip_volatile(v, extra) for v in payload]
    return payload
