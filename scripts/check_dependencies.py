#!/usr/bin/env python3
"""Verify that the two places this project declares dependencies agree.

Every package is named twice: once in `pyproject.toml`, which is what an install
of the distribution resolves, and once in `requirements*.txt`, which is what CI
installs. Nothing compared them, and they had drifted in both directions --
`snowballstemmer` was pinned only in the requirements file, and the `docs` extra
only in `pyproject.toml`, so the two commands gave different environments.

`docs` is deliberately outside the comparison: no CI job builds the
documentation, so requiring `mkdocs-material` in a development install would
make every contributor pay for a job that does not exist. Add it to
`_DEV_GROUPS` in the same change that adds a documentation job.

The linters are checked more strictly than the rest. A floor on a linter is not
a constraint but a promise that every future release agrees with this one, and
that promise fails: both files therefore pin an exact version, and the two pins
must be equal or CI runs a different linter from the one a contributor installs.

Usage::

    python scripts/check_dependencies.py

Exits non-zero on any disagreement, so it can gate CI beside check_versions.py.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Extras a development install is expected to carry. `docs` is excluded; the
#: module docstring says why.
_DEV_GROUPS = ("dev", "viz", "nb")

#: Packages whose version must be exact rather than floored, in both files.
_EXACT = ("ruff", "mypy")

_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+(?:\[[^\]]+\])?)\s*(.*)$")


def _normalise(name: str) -> str:
    """PEP 503 name, with any extras marker kept, so `a_b` and `A-B` compare."""
    base, _, extras = name.partition("[")
    canonical = re.sub(r"[-_.]+", "-", base).lower()
    return f"{canonical}[{extras}" if extras else canonical


def _split(requirement: str) -> tuple[str, str]:
    """A requirement as ``(normalised name, specifier)``."""
    match = _REQUIREMENT.match(requirement.strip())
    if match is None:
        return _normalise(requirement.strip()), ""
    return _normalise(match.group(1)), match.group(2).strip()


def _from_file(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    """Requirements stated by one file, following any ``-r`` it includes."""
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
    """Requirements stated by `pyproject.toml` for the given extras."""
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
    """Report every package one side names and the other does not, or pins differently."""
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
    """Every disagreement between the two declarations."""
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
