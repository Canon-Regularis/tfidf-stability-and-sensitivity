#!/usr/bin/env python3
"""Refuse a stdlib call that the oldest supported interpreter cannot make.

``pyproject.toml`` says ``requires-python = ">=3.11"``, and CI runs 3.11, 3.12
and 3.13. A contributor on a newer interpreter cannot see the difference: a
call that their Python accepts and 3.11 does not passes every local check and
fails in five CI jobs at once.

That happened. ``Path.write_text`` has taken a ``newline`` argument since 3.10
and ``Path.read_text`` only since 3.13, so ``read_text(encoding=..., newline="")``
ran locally on 3.14 and raised ``TypeError`` on every supported version, in the
test suite, the coverage job and the native job.

``mypy`` already knows this -- typeshed models each signature per version -- and
the project already runs it at the floor. It just runs over ``src/`` only, and
this defect was in ``tests/`` and ``scripts/``. Pointing strict mypy at those
would report several hundred findings about pytest decorators and untyped test
helpers, which is a different piece of work; this asks mypy one question instead:

    is any call passing an argument the signature at the floor does not accept?

That is mypy's ``call-arg``, and it is the shape this class of defect takes. The
floor is read from ``pyproject.toml`` rather than repeated here, so raising
``requires-python`` moves the check with it.

Usage::

    python scripts/check_python_floor.py

Exits non-zero if any such call exists, so it can gate CI beside the others.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Checked in addition to `src/`, which the main mypy run already covers.
_ROOTS = ("tests", "scripts", "examples")

#: The one question. Every other mypy code is about typing style rather than
#: about whether the call is possible, and enabling them here would bury this.
_WANTED = "call-arg"


def floor(root: Path) -> str:
    """The minimum interpreter, as ``major.minor``, from ``requires-python``."""
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*"[^0-9]*(\d+\.\d+)', text, re.M)
    if match is None:
        raise SystemExit(f"{pyproject} states no requires-python floor to check against")
    return match.group(1)


def findings(root: Path, version: str) -> list[str]:
    """Every ``call-arg`` mypy reports over the extra roots at ``version``."""
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
