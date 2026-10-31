#!/usr/bin/env python3
"""Verify every C++ subpackage under cpp/include/tfidf has a Python mirror.

Each C++ subpackage needs a counterpart under src/tfidf_stability; the converse is
not asserted, and ``core`` is exempt because its numeric policy comes from the
interpreter. Python-only subpackages are printed, not listed, so this stays current.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: C++ directories with no Python counterpart.
_CPP_ONLY = {"core"}


def main() -> int:
    cpp_root = REPO / "cpp" / "include" / "tfidf"
    py_root = REPO / "src" / "tfidf_stability"

    cpp = {p.name for p in cpp_root.iterdir() if p.is_dir()}
    py = {p.name for p in py_root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))}

    orphaned = cpp - py - _CPP_ONLY
    if orphaned:
        print(
            f"cpp/include/tfidf/{sorted(orphaned)} has no Python counterpart. "
            f"Either add the Python module or move the header under core/.",
            file=sys.stderr,
        )
        return 1

    mirrored = sorted(cpp & py)
    print(f"mirrored: {mirrored}")
    print(f"C++ only (by design): {sorted(cpp & _CPP_ONLY)}")
    print(f"Python only (orchestration, not on a hot path): {sorted(py - cpp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
