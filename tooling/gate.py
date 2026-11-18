"""Report a gate's problems on stderr and return its exit code.

Called by `check_docs.py`, `check_vendored.py`, `check_versions.py` and
`render_summary.py`. `check_dependencies.py` matches the shape but stays out:
importing this would add a `sys.path` line it does not otherwise need.
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
