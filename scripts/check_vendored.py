#!/usr/bin/env python3
"""Verify every vendored file against its recorded digest.

Each ``MANIFEST.sha256`` lists the SHA-256 digest of every file in its
directory. The check also reports a file present in a vendored directory but
absent from its manifest. ``THIRD_PARTY_NOTICES.md`` asserts the vendored sets.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Python puts this script's own directory on `sys.path`, not the repository
# root, so `tooling` needs the root added before it can be imported.
sys.path.insert(0, str(REPO))

from tooling.gate import report  # noqa: E402 - needs REPO on the path above

#: Files a vendored directory may contain without being listed in its manifest.
_EXEMPT = {"MANIFEST.sha256", "__init__.py", "__pycache__"}


def _unlisted(manifest: Path, manifests: list[Path], listed: set[str]) -> list[str]:
    """Files under a manifest's directory that the manifest does not name.

    Paths are compared relative to the manifest directory, the form the
    manifest uses. The walk is recursive because vendored files live in
    subdirectories such as ``doctest/`` and ``nanobench/``.
    """
    problems: list[str] = []
    for path in sorted(manifest.parent.rglob("*")):
        if path.is_dir():
            continue
        relative_parts = path.relative_to(manifest.parent).parts
        if any(part in _EXEMPT for part in relative_parts):
            continue
        # A nested manifest owns its own subtree, so a file there is reported
        # once, against the manifest that ought to list it.
        if any(other != manifest and other.parent in path.parents for other in manifests):
            continue
        if path.relative_to(manifest.parent).as_posix() not in listed:
            problems.append(
                f"{path.relative_to(REPO)}: present in a vendored directory but "
                f"absent from {manifest.name} -- add it or remove it"
            )
    return problems


def check() -> list[str]:
    """Return a list of problems; empty means everything verified."""
    problems: list[str] = []
    manifests = sorted(REPO.rglob("MANIFEST.sha256"))
    if not manifests:
        return ["no MANIFEST.sha256 found anywhere -- vendoring is unverified"]

    checked = 0
    for manifest in manifests:
        listed: set[str] = set()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            try:
                digest, name = line.split("  ", 1)
            except ValueError:
                problems.append(f"{manifest}: unparsable line {line!r}")
                continue

            listed.add(name)
            target = manifest.parent / name
            if not target.exists():
                problems.append(f"{manifest}: {name} is listed but missing")
                continue

            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != digest:
                problems.append(
                    f"{target.relative_to(REPO)}: digest changed\n"
                    f"    recorded {digest}\n"
                    f"    actual   {actual}"
                )
            checked += 1

        problems.extend(_unlisted(manifest, manifests, listed))

    if not problems:
        print(f"verified {checked} vendored files across {len(manifests)} manifests")
    return problems


def main() -> int:
    problems = check()
    return report(problems, "vendored asset verification")


if __name__ == "__main__":
    raise SystemExit(main())
