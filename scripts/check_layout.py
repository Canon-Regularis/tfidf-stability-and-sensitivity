#!/usr/bin/env python3
"""Verify the C++ tree still mirrors the Python packages.

``cpp/CMakeLists.txt`` claims ``cpp/include/tfidf/<sub>/`` corresponds one-for-one
to ``src/tfidf_stability/<sub>/`` and that CI asserts it. This is that check; an
unchecked layout claim decays as modules are added on one side only.

The correspondence is checked in one direction only: every C++ subpackage must
have a Python counterpart. The converse is not required and is not asserted --
Python subpackages with no C++ mirror are expected, since only the hot paths were
ported. ``core`` is exempt even in that direction: numeric policy, the
floating-point guard and the build configuration, all of which the reference
backend inherits from the interpreter.

The set of Python-only subpackages is printed rather than listed here. It was
listed, as five names, and had grown to nine without the docstring noticing --
which is the decay this script exists to prevent, occurring in the script's own
description of it.
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
    py = {
        p.name
        for p in py_root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and p.name != "backends"
    }

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
