#!/usr/bin/env python3
"""Verify that every place this project states a version agrees.

Four files name it, listed in `_SOURCES`. The binding ABI is a second claim,
named by `cpp/bindings/module.cpp` and `src/tfidf_stability/_native/__init__.py`;
``--tag vX.Y.Z`` adds a release tag to the version set.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Python puts this script's own directory on `sys.path`, not the repository
# root, so `tooling` needs the root added before it can be imported.
sys.path.insert(0, str(REPO))

from tooling.gate import report  # noqa: E402 - needs REPO on the path above

#: file -> (regex capturing the version, human description of the line)
_SOURCES: dict[str, tuple[str, str]] = {
    "pyproject.toml": (r"^version\s*=\s*[\"']([^\"']+)[\"']", "the distribution version"),
    "src/tfidf_stability/__init__.py": (
        r"^__version__\s*[:=].*?[\"']([^\"']+)[\"']",
        "what the package reports at runtime",
    ),
    "CITATION.cff": (r"^version:\s*[\"']?([^\"'\s]+)", "what a citation of this work names"),
    # CMakeLists.txt's `VERSION` becomes `PROJECT_VERSION`, then `kVersion` in
    # build_config.hpp, then the native module's `__version__`, which is hashed
    # into every RunManifest. The pattern anchors on the stripped line, so
    # `cmake_minimum_required(VERSION 3.20)` cannot match.
    "CMakeLists.txt": (
        r"^VERSION\s+([0-9][^\s)]*)",
        "what the native extension reports and the run manifest hashes",
    ),
}


#: file -> (regex capturing the binding ABI, human description of the line).
#: The ABI is versioned on its own axis: a release can leave the binding
#: surface untouched, and a new binding can land inside one release.
#: Rationale: docs/engineering_notes.md#binding-abi-checked-separately
_ABI_SOURCES: dict[str, tuple[str, str]] = {
    "cpp/bindings/module.cpp": (
        r'^constexpr\s+const\s+char\*\s+kAbi\s*=\s*"([^"]+)"',
        "the surface the compiled extension declares",
    ),
    "src/tfidf_stability/_native/__init__.py": (
        r'^REQUIRED_ABI\s*=\s*"([^"]+)"',
        "the surface the Python package requires",
    ),
}

#: The line that carries `kAbi` to Python. Without it the two literals can
#: agree while the extension exports no ABI, so the check requires it.
_ABI_EXPORT = 'm.attr("__abi__") = kAbi;'


def _stated(relative: str, pattern: str, repo: Path = REPO) -> str | None:
    path = repo / relative
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1)
    return None


def check_abi(repo: Path = REPO) -> list[str]:
    """Whether the two statements of the binding ABI agree.

    The check is textual, so it holds on a checkout with nothing built; the
    reference CI legs never compile the extension.
    """
    problems: list[str] = []
    found: dict[str, str] = {}
    for relative, (pattern, description) in _ABI_SOURCES.items():
        stated = _stated(relative, pattern, repo)
        if stated is None:
            problems.append(f"{relative}: no ABI found ({description})")
            continue
        found[relative] = stated

    if len(set(found.values())) > 1:
        problems.append("the binding ABI is stated differently in different places:")
        for where, abi in sorted(found.items()):
            problems.append(f"    {where}: {abi}")

    binding = repo / "cpp/bindings/module.cpp"
    if binding.exists() and _ABI_EXPORT not in binding.read_text(encoding="utf-8"):
        problems.append(f"cpp/bindings/module.cpp: no `{_ABI_EXPORT}`, so kAbi reaches nothing")

    if not problems:
        shown = next(iter(set(found.values())), "?")
        print(f"binding ABI {shown} agrees across {len(found)} places")
    return problems


def check(tag: str | None = None) -> list[str]:
    found: dict[str, str] = {}
    problems: list[str] = []

    for relative, (pattern, description) in _SOURCES.items():
        version = _stated(relative, pattern)
        if version is None:
            problems.append(f"{relative}: no version found ({description})")
            continue
        found[relative] = version

    if tag:
        # `v0.2.0` and `0.2.0` are the same claim written two ways.
        found["git tag"] = tag.removeprefix("v")

    distinct = set(found.values())
    if len(distinct) > 1:
        problems.append("the version is stated differently in different places:")
        for where, version in sorted(found.items()):
            problems.append(f"    {where}: {version}")

    if not problems:
        shown = next(iter(distinct), "?")
        print(f"version {shown} agrees across {len(found)} places: {', '.join(sorted(found))}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Rationale: docs/engineering_notes.md#release-tag-joins-version-set
    parser.add_argument(
        "--tag",
        default=None,
        help="a release tag to include in the comparison, e.g. v0.2.0",
    )
    args = parser.parse_args()

    problems = check(args.tag) + check_abi()
    return report(problems, "version check")


if __name__ == "__main__":
    raise SystemExit(main())
