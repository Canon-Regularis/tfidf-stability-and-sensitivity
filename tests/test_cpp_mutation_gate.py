"""The C++ mutation allowlist, checked without running a campaign.

A campaign takes hours, so the failure it would report -- an argument that no
longer describes the code it points at -- would surface a night later at the
earliest, and only on the header that happened to be scheduled. These checks are
the part that can run in a second: every claim must name a line that exists, a
column that holds the token it claims, and a digest that still matches.

The campaign itself is `scripts/run_cpp_mutation_tests.py`, which owns the other
two failure modes: an unlisted survivor, and an entry matching no survivor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO / "configs" / "equivalent_mutants_cpp.txt"


def _harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_cpp_mutation", REPO / "scripts" / "run_cpp_mutation_tests.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cpp_mutation"] = module
    spec.loader.exec_module(module)
    return module


def _entries() -> list[tuple[str, int, int, str, str, str, str]]:
    """``(path, line, column, before, after, stamp, argument)`` for every claim."""
    out = []
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        body, _, reason = raw.partition("#")
        if not body.strip():
            continue
        fields = body.split()
        assert len(fields) == 5, f"malformed entry: {raw}"
        assert fields[3] == "->", f"malformed entry: {raw}"
        line, _, column = fields[1].partition(":")
        stamp, _, argument = reason.strip().partition(" ")
        out.append((fields[0], int(line), int(column), fields[2], fields[4], stamp, argument))
    return out


def test_the_allowlist_is_not_empty() -> None:
    """Every check below is vacuous over an empty file."""
    assert len(_entries()) >= 24


def test_every_claim_points_at_a_line_that_still_exists() -> None:
    for path, line, _, before, after, _, _ in _entries():
        source = REPO / path
        assert source.exists(), f"{path} is gone, but a claim still names it"
        body = source.read_text(encoding="utf-8").splitlines()
        assert line <= len(body), (
            f"{path}:{line} ({before} -> {after}) is past the end of a {len(body)}-line file"
        )


def test_every_claim_still_digests_to_its_recorded_stamp() -> None:
    """The drift guard.

    Line numbers are part of the key, so an edit that moves a line leaves the
    claim pointing at whatever took its place. The digest is of the line's own
    text, so it catches that without needing a campaign to notice.
    """
    harness = _harness()
    wrong = []
    for path, line, _, before, after, stamp, _ in _entries():
        assert stamp.startswith("src="), (
            f"{path}:{line} ({before} -> {after}) carries no src= stamp"
        )
        body = (REPO / path).read_text(encoding="utf-8").splitlines()
        actual = harness.fingerprint(body[line - 1])
        if stamp != f"src={actual}":
            wrong.append(
                f"{path}:{line}  {before} -> {after}\n"
                f"      recorded {stamp}, actual src={actual}\n"
                f"      the line now reads: {body[line - 1].strip()}"
            )
    assert not wrong, "allowlist claims no longer describe their lines:\n" + "\n".join(wrong)


def test_every_claim_names_a_column_holding_the_token_it_claims() -> None:
    """A line can carry several candidates; the column says which one.

    Without this a claim about one `<` silences the other on the same line, and
    the second could be a real gap.
    """
    harness = _harness()
    for path, line, column, before, after, _, _ in _entries():
        text = (REPO / path).read_text(encoding="utf-8", newline="")
        body = text.splitlines()
        actual = body[line - 1][column - 1 : column - 1 + len(before)]
        assert actual == before, (
            f"{path}:{line}:{column} claims {before!r} -> {after!r} but the source holds {actual!r}"
        )
        # And the scanner must agree it is a candidate there: a token inside a
        # comment or a string is never mutated, so a claim about one is dead.
        candidates = {(off, b, a) for off, b, a in harness.scan(text) if b == before and a == after}
        assert candidates, f"{path}: the scanner produces no {before} -> {after} at all"


def test_every_claim_carries_an_argument_and_not_only_a_stamp() -> None:
    """An entry whose whole reason is its own fingerprint is a suppression."""
    harness = _harness()
    for path, line, _, before, after, stamp, argument in _entries():
        assert argument.strip(), (
            f"{path}:{line} ({before} -> {after}) states no reason; "
            f"an entry without one is a suppression, not a finding"
        )
        assert harness.without_stamp(f"{stamp} {argument}"), "the argument must survive stripping"


def test_an_entry_without_a_column_is_refused(tmp_path: Path) -> None:
    """The Python allowlist keys on an AST node kind; this one has only position,
    so an entry that omits the column cannot say which token it means and must
    not be honoured by default."""
    harness = _harness()
    allowlist = tmp_path / "equivalent_mutants_cpp.txt"
    module = Path("cpp/include/tfidf/core/reduction.hpp")
    harness.EQUIVALENTS = allowlist

    allowlist.write_text(f"{module.as_posix()}  57  < -> <=  # a stated reason\n", encoding="utf-8")
    assert harness.load_equivalents(module) == {}, "no column, so no claim"

    allowlist.write_text(
        f"{module.as_posix()}  57:14  < -> <=  # a stated reason\n", encoding="utf-8"
    )
    assert harness.load_equivalents(module) == {(57, 14, "<", "<="): "a stated reason"}


def test_an_entry_whose_only_reason_is_its_stamp_is_refused(tmp_path: Path) -> None:
    harness = _harness()
    allowlist = tmp_path / "equivalent_mutants_cpp.txt"
    module = Path("cpp/include/tfidf/core/reduction.hpp")
    harness.EQUIVALENTS = allowlist
    entry = f"{module.as_posix()}  57:14  < -> <="

    allowlist.write_text(f"{entry}  # src=2a87a12d\n", encoding="utf-8")
    assert harness.load_equivalents(module) == {}, "a stamp is not an argument"

    allowlist.write_text(f"{entry}  # src=2a87a12d both routes agree\n", encoding="utf-8")
    assert harness.load_equivalents(module) == {(57, 14, "<", "<="): "both routes agree"}


def test_the_scanner_never_mutates_a_comment_or_a_string() -> None:
    """The property the whole approach rests on: a mutation that lands in a
    comment is not a mutant, and one that lands in a string literal changes a
    message rather than a behaviour. Either would be scored as a survivor and
    read as a gap in the tests.

    The declaration below avoids a pointer, deliberately: the scanner cannot
    tell the `*` of `char*` from multiplication and offers it as a candidate.
    That mutation does not compile, so it costs a build and is scored stillborn
    rather than producing a wrong verdict.
    """
    harness = _harness()
    source = (
        "// a < b and a > b in a line comment\n"
        "/* a + b and a - b in a block comment */\n"
        "#define NOT_MUTATED (1 + 1)\n"
        'const char s[] = "a < b and 1 + 1";\n'
        "int keep = 2;\n"
    )
    offsets = harness.scan(source)
    lines = source.split("\n")
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)

    def line_of(offset: int) -> int:
        return max(i for i, s in enumerate(starts) if s <= offset)

    touched = {line_of(off) for off, _, _ in offsets}
    assert touched == {4}, f"only the plain statement is mutable, got lines {sorted(touched)}"


@pytest.mark.parametrize(
    ("before", "after"),
    [("<", "<="), (">", ">="), ("==", "!="), ("&&", "||"), ("+", "-"), ("0", "1")],
)
def test_each_mutation_kind_is_produced_on_a_line_that_uses_it(before: str, after: str) -> None:
    """Guards the operator table against a silent shrink: dropping an entry would
    quietly stop testing that operator everywhere, and the campaign would report
    a better score for it."""
    harness = _harness()
    source = "if (a < b && c > d && e == f) { x = x + 0; }\n"
    produced = {(b, a) for _, b, a in harness.scan(source)}
    assert (before, after) in produced


def test_every_header_is_in_the_nightly_matrix() -> None:
    """A header added later must not go unmutated in silence.

    The campaign is a matrix of explicit paths rather than a glob, because a
    glob would also pick up whatever lands in cpp/include next and there is no
    way to spell "and this one is deliberately excluded" in one. The cost of
    the explicit list is that it can fall behind, which is what this refuses.
    """
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")
    )
    listed = set(workflow["jobs"]["cpp-mutation"]["strategy"]["matrix"]["header"])
    present = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "cpp" / "include").rglob("*.hpp")
        if "third_party" not in path.parts
    }

    assert present, "the scan found no headers; it is not testing anything"
    assert not present - listed, f"headers with no nightly mutation job: {sorted(present - listed)}"
    assert not listed - present, (
        f"the matrix names headers that no longer exist: {sorted(listed - present)}"
    )
