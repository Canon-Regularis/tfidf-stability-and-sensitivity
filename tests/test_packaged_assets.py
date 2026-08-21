"""Asset resolution from an installed distribution, not just a checkout.

Two files the package cannot work without live outside the package directory:
the frozen stopword list under ``data/assets`` and the normative
``configs/default.yaml``. Both were resolved through ``parents[3]``, which is the
repository root from ``src/tfidf_stability/<sub>/`` and the directory *above*
``site-packages`` from an installed distribution. An installed package therefore
could not load its own hash-verified stopword list -- the asset that fixes the
vocabulary and so every number this project publishes.

Nothing caught it because ``tests/conftest.py`` prepended ``src/`` to
``sys.path`` unconditionally. ``release.yml``'s ``CIBW_TEST_COMMAND`` runs the
suite against a freshly built wheel for exactly this purpose and got the source
tree instead, so no wheel has ever been under test.

The resolvers now take the module's path as an argument. In a checkout only the
repository layout exists, so the branch that matters for a wheel is unreachable
from here unless a layout is built to order -- which is what these do.
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
    """The wheel layout, which the repository-root form could never reach."""
    module = _installed_layout(tmp_path, "preprocessing", "data/assets")

    resolved = _resolve_asset_dir(module)

    assert resolved == module.resolve().parents[1] / "data" / "assets"
    assert resolved.is_dir()
    assert "site-packages" in resolved.parts


def test_the_stopword_asset_still_resolves_from_a_checkout(tmp_path: Path) -> None:
    """The layout every developer and every CI job uses. Unchanged: the packaged
    directory does not exist in a checkout, so the fallback is what answers."""
    module = _checkout_layout(tmp_path, "preprocessing", "data/assets")

    resolved = _resolve_asset_dir(module)

    assert resolved == module.resolve().parents[3] / "data" / "assets"
    assert resolved.is_dir()
    assert "src" not in resolved.parts, "the assets sit beside src/, not inside it"


def test_the_packaged_asset_wins_when_both_layouts_exist(tmp_path: Path) -> None:
    """Order matters, and only one order is safe.

    An editable install can present both at once. Preferring the packaged copy
    means an installed distribution is self-contained; preferring the repository
    would make a wheel silently read whatever tree it happened to sit near.
    """
    module = _installed_layout(tmp_path, "preprocessing", "data/assets")
    outer = module.resolve().parents[3] / "data" / "assets"
    outer.mkdir(parents=True, exist_ok=True)

    assert _resolve_asset_dir(module) == module.resolve().parents[1] / "data" / "assets"


def test_the_live_asset_directory_holds_the_frozen_list() -> None:
    """The resolver is not merely returning a plausible path: this repository's
    own assets are where it says, and they load."""
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
    """`tfidf-stability build-corpus` with no `-c` reads this file, so an
    installed CLI that cannot find it is a CLI with no default behaviour."""
    module = _installed_layout(tmp_path, "cli", "configs")
    packaged = module.resolve().parents[1] / "configs" / "default.yaml"
    packaged.write_text("preprocessing: {}\n", encoding="utf-8")

    assert _resolve_default_config(module) == packaged


def test_the_default_config_still_resolves_from_a_checkout(tmp_path: Path) -> None:
    """The fallback, and the only path that existed before."""
    module = _checkout_layout(tmp_path, "cli", "configs")
    in_tree = module.resolve().parents[3] / "configs" / "default.yaml"
    in_tree.write_text("preprocessing: {}\n", encoding="utf-8")

    assert _resolve_default_config(module) == in_tree


def test_the_config_falls_back_when_the_package_holds_no_copy(tmp_path: Path) -> None:
    """Presence is tested, not the directory: a packaged ``configs/`` that does
    not contain ``default.yaml`` must not shadow the checkout's copy."""
    module = _installed_layout(tmp_path, "cli", "configs")
    in_tree = module.resolve().parents[3] / "configs" / "default.yaml"
    in_tree.parent.mkdir(parents=True, exist_ok=True)
    in_tree.write_text("preprocessing: {}\n", encoding="utf-8")

    assert _resolve_default_config(module) == in_tree


def test_the_live_default_config_is_the_normative_one() -> None:
    """As above: the resolver points at this repository's real config."""
    from tfidf_stability.cli.commands import _DEFAULT_CONFIG, load_config

    assert _DEFAULT_CONFIG.is_file()
    assert _DEFAULT_CONFIG.name == "default.yaml"
    assert load_config()["_source"] == "default.yaml"


@pytest.mark.parametrize(
    ("subpackage", "asset"),
    [("preprocessing", "data/assets"), ("cli", "configs")],
)
def test_neither_layout_reaches_outside_the_tree_it_names(
    subpackage: str, asset: str, tmp_path: Path
) -> None:
    """The failure that shipped: `parents[3]` from an installed package points
    above `site-packages`, at whatever happens to be there. Both resolvers must
    return a path under the root they were given, never one beside it."""
    module = _installed_layout(tmp_path, subpackage, asset)
    resolver = _resolve_asset_dir if subpackage == "preprocessing" else _resolve_default_config
    if subpackage == "cli":
        (module.resolve().parents[1] / "configs" / "default.yaml").write_text("{}\n", "utf-8")

    resolved = resolver(module)

    assert tmp_path.resolve() in resolved.resolve().parents
