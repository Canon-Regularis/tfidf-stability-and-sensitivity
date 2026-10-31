#!/usr/bin/env python3
"""Check that pyproject.toml and requirements*.txt declare the same packages.

Runtime dependencies are compared with requirements.txt, and the extras in
_DEV_GROUPS plus build requires with requirements-dev.txt. Packages in _EXACT
must carry an exact pin in both files. Exits non-zero on any disagreement.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Extras a development install carries. `docs` is excluded because no CI
#: job builds the documentation.
#: Rationale: docs/engineering_notes.md#docs-extra-outside-dependency-check
_DEV_GROUPS = ("dev", "viz", "nb")

#: Packages whose version must be an exact pin, not a floor, in both files.
#: Rationale: docs/engineering_notes.md#linters-pinned-exactly
_EXACT = ("ruff", "mypy")

_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+(?:\[[^\]]+\])?)\s*(.*)$")


def _normalise(name: str) -> str:
    """Return the PEP 503 normalised name so that `a_b` and `A-B` compare equal.

    Any extras marker is kept.
    """
    base, _, extras = name.partition("[")
    canonical = re.sub(r"[-_.]+", "-", base).lower()
    return f"{canonical}[{extras}" if extras else canonical


def _split(requirement: str) -> tuple[str, str]:
    """Split a requirement into ``(normalised name, specifier)``."""
    match = _REQUIREMENT.match(requirement.strip())
    if match is None:
        return _normalise(requirement.strip()), ""
    return _normalise(match.group(1)), match.group(2).strip()


def _from_file(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    """Return the requirements one file states, following any ``-r`` include."""
    seen = seen if seen is not None else set()
    if path in seen or not path.exists():
        return {}
    seen.add(path)
    found: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r"):
            found.update(_from_file(path.parent / line[2:].strip(), seen))
            continue
        name, specifier = _split(line)
        found[name] = specifier
    return found


def _from_project(data: dict[str, object], groups: tuple[str, ...]) -> dict[str, str]:
    """Return pyproject.toml's dependencies, build requires and the given extras."""
    project = data.get("project", {})
    assert isinstance(project, dict)
    stated: list[str] = list(project.get("dependencies", []))
    extras = project.get("optional-dependencies", {})
    assert isinstance(extras, dict)
    for group in groups:
        stated.extend(extras.get(group, []))
    build = data.get("build-system", {})
    assert isinstance(build, dict)
    stated.extend(build.get("requires", []))
    return dict(_split(item) for item in stated)


def _compare(what: str, declared: dict[str, str], installed: dict[str, str]) -> list[str]:
    """Report packages named on one side only or pinned differently on each."""
    problems: list[str] = []
    for name in sorted(set(installed) - set(declared)):
        problems.append(f"{what}: {name} is in the requirements file but in no pyproject group")
    for name in sorted(set(declared) - set(installed)):
        problems.append(f"{what}: {name} is in pyproject but in no requirements file")
    for name in sorted(set(declared) & set(installed)):
        if declared[name] != installed[name]:
            problems.append(
                f"{what}: {name} is {declared[name] or 'unpinned'} in pyproject and "
                f"{installed[name] or 'unpinned'} in the requirements file"
            )
    return problems


def check(repo: Path = REPO) -> list[str]:
    """Return every disagreement between the two dependency declarations."""
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return ["pyproject.toml: not found"]
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    problems = _compare(
        "runtime",
        dict(_split(item) for item in data.get("project", {}).get("dependencies", [])),
        _from_file(repo / "requirements.txt"),
    )
    problems += _compare(
        "development",
        _from_project(data, _DEV_GROUPS),
        _from_file(repo / "requirements-dev.txt"),
    )

    development = _from_file(repo / "requirements-dev.txt")
    declared = _from_project(data, _DEV_GROUPS)
    for name in _EXACT:
        for where, table in (("requirements-dev.txt", development), ("pyproject.toml", declared)):
            specifier = table.get(name)
            if specifier is None:
                problems.append(f"linters: {name} is named nowhere in {where}")
            elif not specifier.startswith("=="):
                problems.append(f"linters: {name} is {specifier!r} in {where}, not an exact pin")

    if not problems:
        print(
            f"dependencies agree: {len(_from_file(repo / 'requirements.txt'))} runtime, "
            f"{len(development)} development"
        )
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("dependency check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
