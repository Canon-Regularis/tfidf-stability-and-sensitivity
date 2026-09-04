#!/usr/bin/env python3
"""Enforce the parts of ``.clang-format`` that need no clang-format.

``.clang-format`` sets the C++ style and nothing applied it: no CI job runs
clang-format, no gate reads the file, and five lines had drifted past its own
``ColumnLimit`` of 100 without anything objecting. A style file that nothing
enforces is a claim about the tree rather than a property of it.

Running clang-format itself would be better and is not what this does. It is not
installed on every machine that builds this project, and a gate that silently
skips when its tool is missing is the same defect one level down. What is checked
here is the subset that can be checked anywhere, from the values in the file
rather than from constants repeated here:

  ColumnLimit   no line wider than the limit
  UseTab        no tab character in a line
  (implied)     no trailing whitespace, which clang-format always removes

Vendored code is excluded: it is upstream's to format, and ``check_vendored.py``
already asserts it matches upstream byte for byte.

Usage::

    python scripts/check_cpp_format.py

Exits non-zero on any violation, so it can gate CI beside the other checks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Directories whose contents are not ours to format.
_EXCLUDED = ("third_party",)

_SUFFIXES = (".hpp", ".cpp", ".h", ".cc")


def _style(root: Path) -> tuple[int, bool]:
    """``(column_limit, tabs_forbidden)`` read from ``.clang-format``.

    Read rather than hardcoded, so editing the style file moves this check with
    it instead of leaving the two to disagree.
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
    """Every violation, as one message each, in file then line order."""
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
    # Counted, because a gate that scanned nothing would otherwise pass.
    print(f"{len(sources(REPO))} C++ files match .clang-format on width, tabs and trailing space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
