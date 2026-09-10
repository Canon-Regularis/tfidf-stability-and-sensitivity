"""Repository tooling: the code the gates and campaigns are built from.

Separate from ``src/tfidf_stability`` because it is not shipped. The wheel takes
``wheel.packages`` only, so nothing here reaches an installed package, and the
100% coverage floor that governs the library does not govern a campaign driver
whose branches include "the build failed". ``mypy`` checks it; coverage does not.
"""

from __future__ import annotations

__all__: list[str] = []
