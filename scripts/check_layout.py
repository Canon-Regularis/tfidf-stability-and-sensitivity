#!/usr/bin/env python3
"""Verify the C++ tree still mirrors the Python packages.

``cpp/CMakeLists.txt`` states that ``cpp/include/tfidf/<sub>/`` corresponds
one-for-one to ``src/tfidf_stability/<sub>/``, and that a CI check asserts it.
This is that check.

The claim is worth enforcing rather than merely stating: the whole differential
architecture assumes a reader can find the C++ counterpart of a Python module by
name, and a layout claim left unchecked decays silently as modules are added on
one side only.

``core`` is exempt in one direction -- it holds the numeric policy, the
floating-point guard and the build configuration, which have no Python
counterpart because the reference backend inherits those from the interpreter.
Python subpackages without a C++ mirror are fine and expected: orchestration
(``analysis``, ``cli``, ``datasets``, ``persistence``, ``profiles``) never needs
one, because it is not on any hot path.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: C++ directories with no Python counterpart, by design.
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
