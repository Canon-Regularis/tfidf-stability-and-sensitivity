#!/usr/bin/env python3
"""Verify that relative links and code references in the docs resolve.

The implementation notes cross-reference the code heavily, which is what makes
them rot: a module renamed in `src/` leaves a link in `docs/` pointing nowhere.

Three checks, in increasing order of how often they catch something:

1. Relative links resolve. `[text](path)` and `[text](path#anchor)` must name an
   existing file.
2. Anchors exist. A `#g23` fragment must correspond to a heading in the target
   Markdown file; `spec_addenda.md#g22` and `#g23` are cited throughout and an
   off-by-one would go unnoticed.
3. Referenced source files exist. A backticked path like
   `analysis/noise_floor.py` or `src/tfidf_stability/ranking/margins.py` is
   checked against the tree.

4. Source docstrings resolve. The modules cite each other far more densely than
    cites them, and a rename leaves a  role pointing at nothing.
   Sphinx would report it; nothing here runs Sphinx.

External links (http, https, mailto) go unfetched; the build stays hermetic and
offline.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import re
import sys
from collections.abc import Iterator, Sequence
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
        # A heading whose first word is `G23` must also resolve as `#g23`, the
        # form the addenda are cited in.
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


# ---------------------------------------------------------------------------
# Source docstrings
# ---------------------------------------------------------------------------
# The same rot, one directory over. `docs/` is checked because it cites the
# code; the docstrings cite it far more densely -- 157 cross-references across
# 63 modules -- and a rename leaves them pointing at nothing with no build step
# to notice. Sphinx would report these, but nothing here runs Sphinx.
#
# Resolution is by import rather than by parsing, because a cross-reference
# names a runtime object: `Reduction.EXACT` is an enum member, `TauBand.g_min`
# a dataclass field, and neither is visible to a purely syntactic check.

#: `:func:`x`` and friends. The tilde in ``:class:`~pkg.mod.Name`` is Sphinx's
#: "show only the last component" marker and does not affect the target.
_ROLE = re.compile(r":(?:func|class|meth|attr|data|mod|exc):`~?([^`]+)`")

#: Citations of the addenda from inside a docstring, e.g. ``spec_addenda.md#g13``.
_ADDENDA = re.compile(r"spec_addenda\.md#(g\d+)", re.IGNORECASE)

_SRC = REPO / "src" / "tfidf_stability"


def _module_name(path: Path) -> str:
    relative = path.relative_to(_SRC).with_suffix("")
    parts = [p for p in relative.parts if p != "__init__"]
    return ".".join(["tfidf_stability", *parts])


def _reachable(obj: object, attributes: Sequence[str]) -> bool:
    for attribute in attributes:
        if not hasattr(obj, attribute):
            return False
        obj = getattr(obj, attribute)
    return True


def _defined_in_tests(module: str, name: str) -> bool:
    """Whether ``tests/<module>.py`` defines ``name``.

    A docstring may cite the test that pins its claim. Read rather than
    imported: importing a test module drags in fixtures and plugins for a
    question the source text already answers.
    """
    path = REPO / "tests" / f"{module}.py"
    if not path.exists():
        return False
    return name in {
        node.name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _resolves(
    target: str,
    module: object,
    owner: object | None,
    exported: dict[str, object],
) -> bool:
    """Whether a cross-reference target names something that exists.

    Five places, in the order a reader would look: the class the docstring is
    attached to, the module it lives in, the builtins, an importable dotted path
    (which covers the standard library and fully-qualified package names), and
    finally any public name the package exports anywhere -- docstrings routinely
    name a sibling module's class without importing it.
    """
    head, *rest = target.split(".")

    if head == "tests" and len(rest) >= 2:
        return _defined_in_tests(rest[0], rest[1])

    if owner is not None and hasattr(owner, head) and _reachable(getattr(owner, head), rest):
        return True
    if hasattr(module, head) and _reachable(getattr(module, head), rest):
        return True
    if hasattr(builtins, head) and _reachable(getattr(builtins, head), rest):
        return True

    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        try:
            imported = importlib.import_module(".".join(parts[:i]))
        except Exception:  # an unimportable prefix is simply not a match
            continue
        if _reachable(imported, parts[i:]):
            return True

    return head in exported and _reachable(exported[head], rest)


def _documented(tree: ast.Module) -> Iterator[tuple[ast.AST, str, str | None]]:
    """Every docstring in a module, with its text and its owning class name."""
    owners: dict[ast.AST, str | None] = {tree: None}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                owners[child] = node.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.iter_child_nodes(node):
                owners.setdefault(child, owners.get(node))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            text = ast.get_docstring(node)
            if text:
                name = getattr(node, "name", "<module>")
                yield node, text, (name if isinstance(node, ast.ClassDef) else owners.get(node))


def _import_package(
    paths: Sequence[Path],
) -> tuple[dict[Path, object], dict[str, object], list[str]]:
    """Import every module once, and collect the public names they export.

    The export map is what lets a docstring name a sibling module's class
    without importing it, which they do throughout.
    """
    modules: dict[Path, object] = {}
    exported: dict[str, object] = {}
    failures: list[str] = []

    for path in paths:
        name = _module_name(path)
        try:
            modules[path] = importlib.import_module(name)
        except Exception as exc:  # reported as a finding, never raised
            failures.append(f"{name}: docstrings uncheckable, module does not import ({exc})")
            continue
        for public in dir(modules[path]):
            if not public.startswith("_"):
                exported.setdefault(public, getattr(modules[path], public))

    return modules, exported, failures


def check_docstrings(anchor_cache: dict[Path, set[str]]) -> tuple[list[str], int]:
    """Check cross-references, addenda citations and paths in source docstrings."""
    problems: list[str] = []
    paths = sorted(p for p in _SRC.rglob("*.py") if "_snowball" not in p.parts)

    modules, exported, import_failures = _import_package(paths)
    problems.extend(import_failures)

    addenda = REPO / "docs" / "spec_addenda.md"
    if addenda not in anchor_cache:
        anchor_cache[addenda] = _anchors(addenda.read_text(encoding="utf-8"))

    n_refs = 0
    for path in paths:
        module = modules.get(path)
        if module is None:
            continue
        where = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for _node, text, owner_name in _documented(tree):
            owner = getattr(module, owner_name, None) if owner_name else None

            for target in _ROLE.findall(text):
                n_refs += 1
                if not _resolves(target, module, owner, exported):
                    problems.append(f"{where}: cross-reference names nothing: `{target}`")

            for anchor in _ADDENDA.findall(text):
                n_refs += 1
                if anchor.lower() not in anchor_cache[addenda]:
                    problems.append(f"{where}: spec_addenda has no anchor #{anchor.lower()}")

            for reference in _CODE_PATH.findall(text):
                n_refs += 1
                if not any((base / reference).exists() for base in (REPO, _SRC, REPO / "docs")):
                    problems.append(f"{where}: no such file referenced: `{reference}`")

    return problems, n_refs


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

    docstring_problems, n_refs = check_docstrings(anchor_cache)
    problems.extend(docstring_problems)

    if not problems:
        print(
            f"checked {len(documents)} documents: {n_links} links and "
            f"{n_paths} source references all resolve"
        )
        print(f"checked {n_refs} cross-references in source docstrings")
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
