#!/usr/bin/env python3
"""Verify that relative links and code references in the docs resolve.

The implementation notes cross-reference the code heavily -- that is what makes
them useful rather than a second copy of the paper. It is also what makes them
rot: a module renamed in `src/` leaves a link in `docs/` pointing nowhere, and a
stale link is a silent lie about where something lives.

Three checks, in increasing order of how often they catch something:

1. **Relative links resolve.** `[text](path)` and `[text](path#anchor)` must name
   a file that exists.
2. **Anchors exist.** A `#g23` fragment must correspond to a heading in the
   target Markdown file, since `spec_addenda.md#g22` and `#g23` are cited
   throughout and an off-by-one would go unnoticed.
3. **Referenced source files exist.** A backticked path like
   `analysis/noise_floor.py` or `src/tfidf_stability/ranking/margins.py` is
   checked against the tree.

External links (http, https, mailto) are not fetched: the build must stay
hermetic and offline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
#: A backticked token that looks like a repository-relative source path.
_CODE_PATH = re.compile(r"`((?:[\w.-]+/)+[\w.-]+\.(?:py|hpp|cpp|md|yaml|yml|toml|cff))`")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _anchors(text: str) -> set[str]:
    """GitHub-style anchors for every heading in a Markdown document."""
    found = set()
    for heading in _HEADING.findall(text):
        slug = heading.strip().lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s]+", "-", slug)
        found.add(slug)
        # `## G23 -- title` should also be reachable as `#g23`, which is how the
        # addenda are cited everywhere.
        first = slug.split("-")[0]
        if re.fullmatch(r"g\d+", first):
            found.add(first)
    return found


def _check_link(document: Path, target: str, text: str, cache: dict[Path, set[str]]) -> str | None:
    """Validate one Markdown link. Returns a problem description, or None."""
    path_part, _, anchor = target.partition("#")

    if not path_part:
        # A same-document anchor.
        if anchor and anchor not in _anchors(text):
            return f"{document.name}: no heading for anchor #{anchor}"
        return None

    resolved = (document.parent / path_part).resolve()
    if not resolved.exists():
        return f"{document.name}: link target does not exist: {target}"

    if anchor and resolved.suffix == ".md":
        if resolved not in cache:
            cache[resolved] = _anchors(resolved.read_text(encoding="utf-8"))
        if anchor not in cache[resolved]:
            return f"{document.name}: {resolved.name} has no anchor #{anchor}"
    return None


def check() -> list[str]:
    problems: list[str] = []
    documents = sorted(DOCS.glob("*.md"))
    if not documents:
        return ["no Markdown found under docs/"]

    anchor_cache: dict[Path, set[str]] = {}
    n_links = n_paths = 0

    for document in documents:
        text = document.read_text(encoding="utf-8")

        for target in _LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            n_links += 1
            problem = _check_link(document, target, text, anchor_cache)
            if problem:
                problems.append(problem)

        for reference in _CODE_PATH.findall(text):
            n_paths += 1
            candidates = [
                REPO / reference,
                REPO / "src" / "tfidf_stability" / reference,
                DOCS / reference,
            ]
            if not any(c.exists() for c in candidates):
                problems.append(f"{document.name}: no such file referenced: `{reference}`")

    if not problems:
        print(
            f"checked {len(documents)} documents: {n_links} links and "
            f"{n_paths} source references all resolve"
        )
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("documentation check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
