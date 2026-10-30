"""How a repository gate reports what it found.

A headline on stderr, one indented line per problem, exit 1. Called by
`check_docs.py`, `check_vendored.py`, `check_versions.py` and
`render_summary.py`, each of which already puts the repository root on
`sys.path` in order to import this.

The other gates print their own. `check_layout.py` reports one formatted
problem rather than a list; `check_python_floor.py`, `check_cpp_format.py` and
`check_test_vacuity.py` each also print a success line naming what was scanned.
`check_dependencies.py` is the one whose shape matches exactly, and it reaches
nothing else under `tooling`, so importing this would add the `sys.path` line
the four callers above need and save five.
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
