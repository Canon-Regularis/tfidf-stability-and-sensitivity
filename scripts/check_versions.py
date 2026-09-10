#!/usr/bin/env python3
"""Verify that every place this project states a version agrees.

Four files name it, and nothing made them agree: `pyproject.toml` and
`src/tfidf_stability/__init__.py` said 0.2.0 while `CITATION.cff` still said
0.1.0. A citation carrying the wrong version is worse than an unversioned one --
it is a claim about which code produced a published number, and this project's
whole argument is that a result is a function of the repository's contents.

The binding ABI is a second, independent claim, named by
`cpp/bindings/module.cpp` and `src/tfidf_stability/_native/__init__.py`. It sits
on its own axis: a release can leave the binding surface untouched, and a new
binding can land inside one release. A disagreement there makes every process
fall back to the reference backend, and the differential suite then skips rather
than fails, so the run stays green while testing nothing.

On a tag build the tag joins the set. `release.yml` triggers on `v*` and hands
the tag to nothing that checks it, so `git tag v0.3.0` on a tree declaring 0.2.0
would build, test, gate and publish 0.2.0 under a 0.3.0 release. Pass the tag as
an argument (CI: ``--tag "${{ github.ref_name }}"``) to include it.

Usage::

    python scripts/check_versions.py [--tag vX.Y.Z]

Exits non-zero on any disagreement, so it can gate CI beside check_docs.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: file -> (regex capturing the version, human description of the line)
_SOURCES: dict[str, tuple[str, str]] = {
    "pyproject.toml": (r"^version\s*=\s*[\"']([^\"']+)[\"']", "the distribution version"),
    "src/tfidf_stability/__init__.py": (
        r"^__version__\s*[:=].*?[\"']([^\"']+)[\"']",
        "what the package reports at runtime",
    ),
    "CITATION.cff": (r"^version:\s*[\"']?([^\"'\s]+)", "what a citation of this work names"),
    # The fourth. CMakeLists.txt's `VERSION` becomes `PROJECT_VERSION`, then
    # `kVersion` in build_config.hpp, then the native module's `__version__`
    # and `build_info()["version"]`, which is hashed into every RunManifest.
    #
    # Anchored on the stripped line, so `cmake_minimum_required(VERSION 3.20)`
    # cannot match: that line begins with the command name, not with VERSION.
    "CMakeLists.txt": (
        r"^VERSION\s+([0-9][^\s)]*)",
        "what the native extension reports and the run manifest hashes",
    ),
}


#: The ABI is a second version, on its own axis: a release can leave the binding
#: surface untouched, and a new binding can land inside one release. Two files
#: state it and nothing compared them.
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

#: The line that carries `kAbi` to Python. Without it the two literals could
#: agree while the extension exported something else, and the check above would
#: pass while saying nothing.
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

    Textual, so it holds on a checkout with nothing built: the reference CI legs
    never compile the extension, and a mismatch there degrades every process to
    the reference backend without saying so.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default=None,
        help="a release tag to include in the comparison, e.g. v0.2.0",
    )
    args = parser.parse_args()

    problems = check(args.tag) + check_abi()
    if problems:
        print("version check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
