"""Asset resolution from an installed distribution and from a checkout.

``data/assets`` and ``configs/default.yaml`` sit beside ``src/`` in a checkout
and inside the package in a wheel; each resolver takes the module's path.
Rationale: docs/engineering_notes.md#packaged-asset-resolution
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tfidf_stability.cli.commands import _resolve_default_config
from tfidf_stability.preprocessing.stopwords import _resolve_asset_dir


def _installed_layout(root: Path, subpackage: str, asset: str) -> Path:
    """A wheel's layout: the asset inside the package, beside the subpackage.

    ``site-packages/tfidf_stability/<subpackage>/mod.py`` with the asset at
    ``site-packages/tfidf_stability/<asset>``. Returns the module path.
    """
    package = root / "site-packages" / "tfidf_stability"
    (package / subpackage).mkdir(parents=True)
    (package / asset).mkdir(parents=True, exist_ok=True)
    module = package / subpackage / "mod.py"
    module.write_text("", encoding="utf-8")
    return module


def _checkout_layout(root: Path, subpackage: str, asset: str) -> Path:
    """A source checkout: the asset at the repository root, outside ``src/``."""
    package = root / "repo" / "src" / "tfidf_stability"
    (package / subpackage).mkdir(parents=True)
    (root / "repo" / asset).mkdir(parents=True, exist_ok=True)
    module = package / subpackage / "mod.py"
    module.write_text("", encoding="utf-8")
    return module


# ---------------------------------------------------------------------------
# The stopword asset
# ---------------------------------------------------------------------------
def test_the_stopword_asset_resolves_inside_an_installed_package(tmp_path: Path) -> None:
    """The wheel layout resolves to the asset directory inside the package."""
    module = _installed_layout(tmp_path, "preprocessing", "data/assets")

    resolved = _resolve_asset_dir(module)

    assert resolved == module.resolve().parents[1] / "data" / "assets"
    assert resolved.is_dir()
    assert "site-packages" in resolved.parts


def test_the_stopword_asset_still_resolves_from_a_checkout(tmp_path: Path) -> None:
    """A checkout resolves to the asset directory beside ``src/``.

    The packaged directory does not exist in a checkout, so the fallback answers.
    """
    module = _checkout_layout(tmp_path, "preprocessing", "data/assets")

    resolved = _resolve_asset_dir(module)

    assert resolved == module.resolve().parents[3] / "data" / "assets"
    assert resolved.is_dir()
    assert "src" not in resolved.parts, "the assets sit beside src/, not inside it"


def test_the_packaged_asset_wins_when_both_layouts_exist(tmp_path: Path) -> None:
    """The packaged copy is preferred when both layouts exist.

    An editable install can present both at once. Preferring the packaged copy
    keeps an installed distribution self-contained instead of reading whatever
    tree sits above ``site-packages``.
    """
    module = _installed_layout(tmp_path, "preprocessing", "data/assets")
    outer = module.resolve().parents[3] / "data" / "assets"
    outer.mkdir(parents=True, exist_ok=True)

    assert _resolve_asset_dir(module) == module.resolve().parents[1] / "data" / "assets"


def test_the_live_asset_directory_holds_the_frozen_list() -> None:
    """The repository's own stopword assets sit at the resolved path and load."""
    from tfidf_stability.preprocessing.stopwords import (
        _ASSET_DIR,
        DEFAULT_STOPWORD_ASSET,
        load_stopwords,
    )

    assert (_ASSET_DIR / DEFAULT_STOPWORD_ASSET).is_file()
    assert (_ASSET_DIR / "MANIFEST.sha256").is_file()
    assert len(load_stopwords()) > 100, "the frozen English list, not an empty set"


# ---------------------------------------------------------------------------
# The default configuration
# ---------------------------------------------------------------------------
def test_the_default_config_resolves_inside_an_installed_package(tmp_path: Path) -> None:
    """The wheel layout resolves to the packaged ``default.yaml``.

    ``tfidf-stability build-corpus`` with no ``-c`` reads this file, so an
    installed CLI depends on finding it.
    """
    module = _installed_layout(tmp_path, "cli", "configs")
    packaged = module.resolve().parents[1] / "configs" / "default.yaml"
    packaged.write_text("preprocessing: {}\n", encoding="utf-8")

    assert _resolve_default_config(module) == packaged


def test_the_default_config_still_resolves_from_a_checkout(tmp_path: Path) -> None:
    """A checkout resolves to the ``default.yaml`` beside ``src/``."""
    module = _checkout_layout(tmp_path, "cli", "configs")
    in_tree = module.resolve().parents[3] / "configs" / "default.yaml"
    in_tree.write_text("preprocessing: {}\n", encoding="utf-8")

    assert _resolve_default_config(module) == in_tree


def test_the_config_falls_back_when_the_package_holds_no_copy(tmp_path: Path) -> None:
    """The file's presence decides the result, not the directory's.

    A packaged ``configs/`` without ``default.yaml`` does not shadow the
    checkout's copy.
    """
    module = _installed_layout(tmp_path, "cli", "configs")
    in_tree = module.resolve().parents[3] / "configs" / "default.yaml"
    in_tree.parent.mkdir(parents=True, exist_ok=True)
    in_tree.write_text("preprocessing: {}\n", encoding="utf-8")

    assert _resolve_default_config(module) == in_tree


def test_the_live_default_config_is_the_normative_one() -> None:
    """The resolver points at this repository's own ``default.yaml``."""
    from tfidf_stability.cli.commands import _DEFAULT_CONFIG, load_config

    assert _DEFAULT_CONFIG.is_file()
    assert _DEFAULT_CONFIG.name == "default.yaml"
    assert load_config()["_source"] == "default.yaml"


@pytest.mark.parametrize("installed", [True, False], ids=["installed", "checkout"])
@pytest.mark.parametrize(
    ("subpackage", "asset"),
    [("preprocessing", "data/assets"), ("cli", "configs")],
)
def test_neither_layout_reaches_outside_the_tree_it_names(
    subpackage: str, asset: str, installed: bool, tmp_path: Path
) -> None:
    """With a copy offered just outside, each resolver stays inside its tree.

    The tree is the package when installed, with the packaged copy present, and
    the checkout root otherwise. Just outside lies the directory holding
    ``site-packages``, or the directory holding the checkout.
    """
    module = (_installed_layout if installed else _checkout_layout)(tmp_path, subpackage, asset)
    tree = module.resolve().parents[1 if installed else 3]
    outside = module.resolve().parents[3 if installed else 4]
    resolver = _resolve_asset_dir if subpackage == "preprocessing" else _resolve_default_config
    for root in (tree, outside):
        (root / asset).mkdir(parents=True, exist_ok=True)
        if subpackage == "cli":
            (root / asset / "default.yaml").write_text("{}\n", "utf-8")

    resolved = resolver(module)

    assert tree in resolved.resolve().parents
