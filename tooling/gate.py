"""How a repository gate reports what it found.

Three gates spelled the same seven lines: a headline on stderr, one indented
line per problem, exit 1. They are one thing, so they live here.

`check_layout.py`, `check_python_floor.py`, `check_cpp_format.py` and
`check_test_vacuity.py` are deliberately left out. They report findings rather
than problem strings, and two of them take arguments; bending them to this shape
costs more than the four lines it saves.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

__all__ = ["report"]


def report(problems: Sequence[str], headline: str) -> int:
    """Print every problem under ``headline`` and return the exit code.

    Args:
        problems: One line per problem. Empty means the gate passed.
        headline: What failed, as a noun phrase: "version check", not "failed".
    """
    if not problems:
        return 0
    print(f"{headline} FAILED:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return 1
