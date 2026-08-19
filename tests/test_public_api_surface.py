"""The declared surface, and the claims it makes about the C++ side.

Three properties, none of which any other test file asserts.

First, ``types.py`` describes how its aliases correspond to
``cpp/include/tfidf/core/types.hpp``. That description is checkable, and it was
wrong until this file existed: the docstring claimed a one-for-one mirror while
``DocId`` denoted a row index in C++ and an external string in Python. The same
name meaning different things across a language boundary is the sort of defect
that produces a plausible wrong answer rather than a crash, so the correspondence
is pinned here rather than described in prose alone.

Second, every enum in the package is a ``class X(str, Enum)`` with an explicit
``__str__``. ``pyproject.toml`` justifies not using ``StrEnum`` on the grounds
that the explicit method "makes the serialised form part of the visible contract
rather than an inherited behaviour". Those values land in run manifests and YAML
configs, so a member whose ``str()`` stopped being its ``value`` would change a
published file while every existing test still passed.

Third, ``__all__`` is the package's statement about what it supports. A name
listed there that does not resolve is a broken promise, and it is the kind of
thing that survives indefinitely because nothing imports the module.

What this file deliberately does not do is import a module to make its lines
execute. ``types.py`` is ten ``TypeAlias`` assignments that run at import; a bare
``import tfidf_stability.types`` would score it fully covered while asserting
nothing at all.

One consequence to be aware of when reading a coverage report. The last two
tests walk the package and import every module, which is the only way to check
that ``__all__`` resolves. That marks import-time statements across the whole
package as covered without exercising any behaviour, so a module can report
coverage on the strength of this file alone. Measured at the time of writing,
deselecting it moves the project total by nine statements; the figure shrinks as
modules gain their own tests, which is the point. A module whose coverage depends
on this file has gained nothing, so judge one by whether its own owning test file
exercises it.
"""

from __future__ import annotations

import enum
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import tfidf_stability
from tfidf_stability import types

REPO = Path(__file__).resolve().parents[1]
HEADER = REPO / "cpp" / "include" / "tfidf" / "core" / "types.hpp"

#: The correspondence `types.py` documents, as a table a test can check.
#: Python name -> the C++ alias it corresponds to, or None where there is none.
_CORRESPONDENCE: dict[str, str | None] = {
    "TermId": "TermId",
    "DocIndex": "DocId",  # the names differ on purpose; see the module docstring
    "DocId": None,  # the external string identifier; C++ has no counterpart
    "Rank": None,  # 1-indexed, Python-side reporting only
    "Offset": "Offset",
    "Real": "Real",
    "Score": "Score",
}


def _header_aliases() -> dict[str, str]:
    """Parse `using NAME = TYPE;` out of the C++ header."""
    pattern = re.compile(r"^using\s+(\w+)\s*=\s*([^;]+);", re.MULTILINE)
    return {
        name: spelling.strip()
        for name, spelling in pattern.findall(HEADER.read_text(encoding="utf-8"))
    }


# ---------------------------------------------------------------------------
# types.py against the C++ header
# ---------------------------------------------------------------------------
def test_the_header_this_module_describes_still_exists() -> None:
    """A correspondence test that silently stops finding its counterpart proves
    nothing, so the file's existence is asserted before anything is parsed."""
    assert HEADER.is_file(), f"{HEADER} is missing; types.py describes a file that is not there"


def test_every_python_alias_is_accounted_for_in_the_correspondence() -> None:
    """`__all__` and the documented table must not drift apart.

    Adding an alias without deciding whether C++ has one is exactly how the
    one-for-one claim became false.
    """
    assert set(types.__all__) == set(_CORRESPONDENCE), (
        "types.__all__ and the documented C++ correspondence disagree; "
        "a new alias needs a row in the table and in the module docstring"
    )


def test_each_alias_claiming_a_cpp_counterpart_has_one() -> None:
    aliases = _header_aliases()
    checked = 0
    for python_name, cpp_name in _CORRESPONDENCE.items():
        if cpp_name is None:
            continue
        assert cpp_name in aliases, (
            f"types.py says {python_name} corresponds to C++ {cpp_name}, "
            f"but the header declares {sorted(aliases)}"
        )
        checked += 1
    assert checked == 5, "the correspondence table lost a row without this test noticing"


def test_each_alias_claiming_no_counterpart_really_has_none() -> None:
    """The other direction, which is the one that caught the DocId collision.

    `DocId` exists in both files, so a test that only checked "the names I claim
    are shared are present" would pass while the two meant opposite things.
    """
    aliases = _header_aliases()
    assert _CORRESPONDENCE["Rank"] is None
    assert "Rank" not in aliases, "C++ grew a Rank alias; the correspondence needs revisiting"

    # DocId is the trap: present in both, same spelling, different concept.
    assert "DocId" in aliases, "the header no longer declares DocId"
    assert aliases["DocId"] == "std::int32_t", (
        "C++ DocId is a row index. If it became a string type the two sides would "
        "have converged and DocIndex could be retired."
    )
    assert types.DocId is str, "Python DocId is the external identifier, not an index"
    assert types.DocIndex is int, "Python DocIndex is the row index C++ calls DocId"


def test_the_widths_the_header_pins_are_the_ones_the_docstring_reports() -> None:
    aliases = _header_aliases()
    assert aliases["TermId"] == "std::int32_t"
    assert aliases["Offset"] == "std::int64_t", (
        "Offset is int64 because total non-zeros can exceed 2^31 even when "
        "document and term counts cannot"
    )
    assert aliases["Real"] == "double"
    assert aliases["Score"] == "double"


def test_every_alias_resolves_to_a_runtime_type() -> None:
    for name in types.__all__:
        resolved = getattr(types, name)
        assert isinstance(resolved, type), f"{name} does not resolve to a type"


def test_all_is_sorted_so_a_diff_shows_only_what_changed() -> None:
    assert list(types.__all__) == sorted(types.__all__)


# ---------------------------------------------------------------------------
# The enum serialisation contract
# ---------------------------------------------------------------------------
def _package_enums() -> list[type[enum.Enum]]:
    """Every Enum subclass reachable by importing the package's own modules."""
    found: dict[str, type[enum.Enum]] = {}
    for info in pkgutil.walk_packages(tfidf_stability.__path__, f"{tfidf_stability.__name__}."):
        if "_snowball" in info.name or "_native" in info.name:
            continue
        module = importlib.import_module(info.name)
        for attribute in vars(module).values():
            if (
                isinstance(attribute, type)
                and issubclass(attribute, enum.Enum)
                and attribute.__module__.startswith("tfidf_stability")
            ):
                found[f"{attribute.__module__}.{attribute.__qualname__}"] = attribute
    return list(found.values())


def test_every_enum_str_is_its_value_so_a_manifest_round_trips() -> None:
    """The reason `pyproject.toml` gives for not using StrEnum.

    Each of these classes writes an explicit ``__str__`` returning ``self.value``,
    and the justification recorded against the UP042 ignore is that doing so
    "makes the serialised form part of the visible contract". These values are
    written into run manifests and read back out of YAML configs, so a member
    whose ``str()`` drifted from its ``value`` would change a published file.
    """
    enums = _package_enums()
    assert len(enums) >= 10, f"only found {len(enums)} enums; the walk is not reaching the package"

    checked = 0
    for enum_class in enums:
        for member in enum_class:
            if not isinstance(member.value, str):
                continue
            assert str(member) == member.value, (
                f"{enum_class.__module__}.{enum_class.__qualname__}.{member.name} "
                f"serialises as {str(member)!r} but its value is {member.value!r}"
            )
            checked += 1
    assert checked > 30, f"only {checked} string enum members were checked; expected far more"


def test_a_string_enum_member_compares_equal_to_its_serialised_form() -> None:
    """`class X(str, Enum)` is chosen so a config value read from YAML matches."""
    from tfidf_stability.utils.numerics import Reduction

    assert Reduction.NAIVE == "naive"
    assert Reduction("naive") is Reduction.NAIVE


# ---------------------------------------------------------------------------
# __all__ resolves, everywhere
# ---------------------------------------------------------------------------
def _modules_declaring_all() -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for info in pkgutil.walk_packages(tfidf_stability.__path__, f"{tfidf_stability.__name__}."):
        if "_snowball" in info.name or "_native" in info.name:
            continue
        module = importlib.import_module(info.name)
        declared = getattr(module, "__all__", None)
        if declared:
            out.append((info.name, list(declared)))
    return out


def test_every_name_in_every_all_resolves() -> None:
    """A promise in `__all__` that does not resolve survives forever otherwise.

    Nothing imports several of these modules, so an `__all__` entry naming a
    symbol that was renamed would never raise until a user tried the documented
    import.
    """
    modules = _modules_declaring_all()
    assert len(modules) >= 30, (
        f"only {len(modules)} modules declare __all__; the walk is too narrow"
    )

    resolved = 0
    for name, declared in modules:
        module = importlib.import_module(name)
        for symbol in declared:
            assert hasattr(module, symbol), f"{name}.__all__ names {symbol!r}, which does not exist"
            resolved += 1
    assert resolved > 200, f"only {resolved} exported names were checked"


@pytest.mark.parametrize(
    "package",
    ["analysis", "persistence", "perturbation", "profiles", "ranking"],
)
def test_each_subpackage_reexport_resolves(package: str) -> None:
    module = importlib.import_module(f"tfidf_stability.{package}")
    declared = getattr(module, "__all__", [])
    assert declared, f"tfidf_stability.{package} declares no __all__"
    for symbol in declared:
        assert hasattr(module, symbol), f"tfidf_stability.{package} re-exports missing {symbol!r}"


def test_the_root_package_exports_only_its_version() -> None:
    """Deliberate: everything else is imported from a subpackage.

    Pinned because a convenience re-export added at the root would be a public
    API commitment made by accident.
    """
    assert tfidf_stability.__all__ == ["__version__"]
    assert isinstance(tfidf_stability.__version__, str)
