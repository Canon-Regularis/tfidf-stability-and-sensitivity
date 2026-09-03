#!/usr/bin/env python3
"""Verify every vendored file against its recorded digest.

Three vendored trees, each depending on the bytes being what upstream published:
the Snowball stemmer (Python and C++ backends run implementations generated from
one source), the doctest and nanobench headers (hermetic offline build), and the
Snowball test vectors (ground truth for the stemmer). ``THIRD_PARTY_NOTICES.md``
asserts all three.

Also checks the reverse direction, which is the failure that actually happens: a
file added to a vendored directory but never added to its manifest. A digest
check alone passes, since nothing points at the new file.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Files a vendored directory may contain without being listed in its manifest.
_EXEMPT = {"MANIFEST.sha256", "__init__.py", "__pycache__"}


def _unlisted(manifest: Path, manifests: list[Path], listed: set[str]) -> list[str]:
    """Files under a manifest's directory that the manifest does not name.

    An unlisted file is invisible to the digest comparison, so it ships
    unverified. Compared on the path relative to the manifest rather than on the
    bare name, because that is the form the manifest uses:
    ``cpp/third_party/MANIFEST.sha256`` lists ``doctest/doctest.h``.

    Walked recursively, because every file ``cpp/third_party`` vendors lives one
    level down, in ``doctest/`` and ``nanobench/``.
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
    if problems:
        print("vendored asset verification FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
