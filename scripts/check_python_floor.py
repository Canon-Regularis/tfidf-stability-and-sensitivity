#!/usr/bin/env python3
"""Refuse a stdlib call that the minimum supported Python cannot make.

Runs mypy at the ``requires-python`` floor from ``pyproject.toml`` over
``tests/``, ``scripts/`` and ``examples/`` and reports every ``call-arg``
finding. Rationale: docs/engineering_notes.md#python-floor-call-arg-check
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Checked in addition to `src/`, which the main mypy run covers.
_ROOTS = ("tests", "scripts", "examples")

#: The only mypy error code reported. Other codes concern typing style rather
#: than whether the call is possible at the floor.
_WANTED = "call-arg"


def floor(root: Path) -> str:
    """Return the ``major.minor`` floor from ``requires-python``."""
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*"[^0-9]*(\d+\.\d+)', text, re.M)
    if match is None:
        raise SystemExit(f"{pyproject} states no requires-python floor to check against")
    return match.group(1)


def findings(root: Path, version: str) -> list[str]:
    """List mypy's ``call-arg`` lines for the extra roots at ``version``."""
    roots = [str(root / name) for name in _ROOTS if (root / name).is_dir()]
    if not roots:
        raise SystemExit(f"none of {_ROOTS} exist under {root}; there is nothing to check")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--python-version",
            version,
            "--ignore-missing-imports",
            "--no-error-summary",
            "--no-incremental",
            *roots,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 and not result.stdout.strip():
        raise SystemExit(f"mypy did not run: {result.stderr.strip()[:400]}")
    return [line for line in result.stdout.splitlines() if f"[{_WANTED}]" in line]


def main() -> int:
    version = floor(REPO)
    found = findings(REPO, version)
    if found:
        print(
            f"calls that Python {version} cannot make, though a newer one can:",
            file=sys.stderr,
        )
        for line in found:
            print(f"  {line}", file=sys.stderr)
        print(
            f"\n{len(found)} such calls. requires-python is >={version}, and CI runs it.",
            file=sys.stderr,
        )
        return 1
    print(f"no call in {', '.join(_ROOTS)} needs a Python newer than {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
