#!/usr/bin/env python3
"""Check C++ sources against the ``.clang-format`` rules that need no tool.

Reads ``ColumnLimit`` and ``UseTab`` from ``.clang-format`` and rejects lines
over the limit, tab characters and trailing whitespace; vendored code is skipped.
Rationale: docs/engineering_notes.md#clang-format-subset-check
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Directories holding vendored code, excluded from the check.
_EXCLUDED = ("third_party",)

_SUFFIXES = (".hpp", ".cpp", ".h", ".cc")


def _style(root: Path) -> tuple[int, bool]:
    """Return ``(column_limit, tabs_forbidden)`` read from ``.clang-format``.

    The values come from the style file rather than constants here, so an edit
    to ``.clang-format`` changes this check with it.
    """
    config = root / ".clang-format"
    text = config.read_text(encoding="utf-8")

    limit = re.search(r"^ColumnLimit:\s*(\d+)", text, re.M)
    if limit is None:
        raise SystemExit(f"{config} sets no ColumnLimit; this gate has nothing to enforce")

    use_tab = re.search(r"^UseTab:\s*(\w+)", text, re.M)
    return int(limit.group(1)), (use_tab is not None and use_tab.group(1) == "Never")


def sources(root: Path) -> list[Path]:
    return sorted(
        path
        for path in (root / "cpp").rglob("*")
        if path.suffix in _SUFFIXES
        and path.is_file()
        and not any(part in _EXCLUDED for part in path.parts)
    )


def problems(root: Path) -> list[str]:
    """Return every violation as one message each, in file then line order."""
    limit, tabs_forbidden = _style(root)
    found: list[str] = []
    for path in sources(root):
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if len(line) > limit:
                found.append(f"{relative}:{number}: {len(line)} columns, limit is {limit}")
            if tabs_forbidden and "\t" in line:
                found.append(f"{relative}:{number}: tab character, UseTab is Never")
            if line != line.rstrip():
                found.append(f"{relative}:{number}: trailing whitespace")
    return found


def main() -> int:
    found = problems(REPO)
    if found:
        for message in found:
            print(message, file=sys.stderr)
        print(f"{len(found)} formatting violations", file=sys.stderr)
        return 1
    # The file count makes a run that scanned nothing visible in the output.
    print(f"{len(sources(REPO))} C++ files match .clang-format on width, tabs and trailing space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
