"""Repository tooling: the code the gates and campaigns are built from.

Kept outside ``src/tfidf_stability`` because it is not shipped: the wheel
packages ``wheel.packages`` only. The 100% coverage floor that governs the
library does not govern this package; ``mypy`` checks it, coverage does not.
"""

from __future__ import annotations

__all__: list[str] = []
