"""Locate a file the package ships, in a wheel or in a source checkout.

A wheel places data inside the package (``tfidf_stability/<parts>``) and a
checkout keeps it at the repository root (``<repo>/<parts>``), so one rule
serves every asset. Rationale: docs/engineering_notes.md#packaged-asset-resolution
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

__all__ = ["resolve_packaged_path"]


def resolve_packaged_path(module: Path, *parts: str, present: Callable[[Path], bool]) -> Path:
    """Return the packaged ``parts`` if present, else the checkout's.

    ``module`` is a module in a subpackage: ``parents[1]`` is the package and ``parents[3]`` a
    checkout's root. The wheel copy wins, because from a wheel ``parents[3]`` is the directory
    above ``site-packages``. ``present`` is ``Path.is_dir`` or ``Path.is_file``.
    """
    packaged = module.resolve().parents[1].joinpath(*parts)
    if present(packaged):
        return packaged
    return module.resolve().parents[3].joinpath(*parts)
