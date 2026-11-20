"""BEIR archive parsing, loading and registration.

The archives are not redistributed, so each test builds a small one in memory
with the layout and quirks of the published zips: directory entries, CRLF
qrels with a header, and a queries file holding more queries than the split judges.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tfidf_stability.datasets import beir
from tfidf_stability.datasets.loaders import DATASET_NAMES, load_dataset
from tfidf_stability.utils.hashing import hash_bytes, hash_text
from tfidf_stability.utils.io import canonical_json
from tfidf_stability.utils.validation import DataIntegrityError

_CORPUS: list[dict[str, Any]] = [
    {"_id": "d2", "title": "Alpha beta", "text": "gamma delta", "metadata": {}},
    {"_id": "d10", "title": "", "text": "beta gamma", "metadata": {}},
    {"_id": "d1", "title": "Epsilon", "text": "alpha", "metadata": {}},
]
_QUERIES: list[dict[str, Any]] = [
    {"_id": "q2", "text": "gamma", "metadata": {}},
    {"_id": "q1", "text": "alpha beta", "metadata": {}},
    {"_id": "q3", "text": "judged only in train", "metadata": {}},
    {"_id": "q4", "text": "nothing relevant", "metadata": {}},
]
_QRELS = "query-id\tcorpus-id\tscore\nq2\td10\t1\nq1\td2\t2\nq1\td1\t0\nq4\td1\t0\n"


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _beir_zip(
    *,
    top: str = "tiny",
    corpus: str | None = None,
    queries: str | None = None,
    qrels: str | None = _QRELS,
    extra: dict[str, str] | None = None,
) -> bytes:
    """A BEIR-shaped zip; ``None`` leaves that member out."""
    members = {
        f"{top}/corpus.jsonl": _jsonl(_CORPUS) if corpus is None else corpus,
        f"{top}/queries.jsonl": _jsonl(_QUERIES) if queries is None else queries,
        f"{top}/qrels/train.tsv": "query-id\tcorpus-id\tscore\r\nq3\td1\t1\r\n",
    }
    if qrels is not None:
        members[f"{top}/qrels/test.tsv"] = qrels.replace("\n", "\r\n")
    members.update(extra or {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{top}/", "")
        archive.writestr(f"{top}/qrels/", "")
        for name, text in members.items():
            archive.writestr(name, text)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_documents_sort_by_utf8_identifier_and_keep_title_and_body_apart() -> None:
    corpus = beir.parse_archive(_beir_zip())

    assert corpus.variant == "tiny"
    assert corpus.split == "test"
    assert corpus.doc_ids == ("d1", "d10", "d2"), "UTF-8 order, so 'd10' precedes 'd2'"
    assert corpus.fields == (("Epsilon", "alpha"), ("beta gamma",), ("Alpha beta", "gamma delta"))
    assert corpus.texts == ("Epsilon alpha", "beta gamma", "Alpha beta gamma delta"), (
        "an empty title leaves no leading space"
    )
    assert corpus.records()[1] == {"doc_id": "d10", "text": "beta gamma", "fields": ["beta gamma"]}


def test_only_the_queries_the_split_judges_are_kept() -> None:
    corpus = beir.parse_archive(_beir_zip())

    assert corpus.queries == (("q1", "alpha beta"), ("q2", "gamma"), ("q4", "nothing relevant"))
    assert corpus.query_texts() == {"q1": "alpha beta", "q2": "gamma", "q4": "nothing relevant"}


def test_judgements_keep_every_grade_including_zero() -> None:
    corpus = beir.parse_archive(_beir_zip())

    assert corpus.judgements() == {"q1": {"d1": 0, "d2": 2}, "q2": {"d10": 1}, "q4": {"d1": 0}}
    assert corpus.qrels == (("q1", "d1", 0), ("q1", "d2", 2), ("q2", "d10", 1), ("q4", "d1", 0))
    assert corpus.n_queries_without_relevant == 1, "q4 is judged but has no grade above zero"


def test_another_split_is_read_by_name() -> None:
    corpus = beir.parse_archive(_beir_zip(), split="train")

    assert corpus.queries == (("q3", "judged only in train"),)
    assert corpus.judgements() == {"q3": {"d1": 1}}


def test_a_document_with_neither_title_nor_body_has_no_fields() -> None:
    rows = [{"_id": "d1", "title": " ", "text": ""}, {"_id": "d10", "text": "x"}]
    rows.append({"_id": "d2", "title": "t", "text": "y"})
    corpus = beir.parse_archive(_beir_zip(corpus=_jsonl(rows)))

    assert corpus.fields[0] == ()
    assert corpus.texts[0] == ""
    assert corpus.fields[1] == ("x",), "a missing title reads as empty"


def test_blank_lines_in_every_member_are_skipped() -> None:
    corpus = beir.parse_archive(
        _beir_zip(
            corpus="\n" + _jsonl(_CORPUS) + "\n",
            queries=_jsonl(_QUERIES) + "  \n",
            qrels=_QRELS + "\n",
        )
    )

    assert corpus.n_documents == 3
    assert len(corpus.qrels) == 4


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_an_archive_with_two_top_level_folders_is_refused() -> None:
    data = _beir_zip(extra={"__MACOSX/tiny/._corpus.jsonl": "resource fork"})

    with pytest.raises(DataIntegrityError, match=r"one top-level folder, found \['__MACOSX'"):
        beir.parse_archive(data)


def test_a_missing_member_is_named() -> None:
    with pytest.raises(DataIntegrityError, match=r"missing \['tiny/qrels/test\.tsv'\]"):
        beir.parse_archive(_beir_zip(qrels=None))


@pytest.mark.parametrize(
    ("corpus", "message"),
    [
        ('{"_id": "d1", "text": "a"}\n{not json\n', r"corpus\.jsonl line 2: malformed JSON"),
        ('["d1", "a"]\n', r"corpus\.jsonl line 1: expected a JSON object"),
        ('{"title": "t", "text": "a"}\n', r"lacks '_id'"),
        ('{"_id": 7, "text": "a"}\n', r"'_id' of 7 is not a string"),
        ('{"_id": "", "text": "a"}\n', r"empty '_id'"),
        ('{"_id": "d1", "text": "a"}\n{"_id": "d1", "text": "b"}\n', r"duplicate '_id' 'd1'"),
        ('{"_id": "d1", "title": "t"}\n', r"lacks 'text': 'd1'"),
        ('{"_id": "d1", "title": 3, "text": "a"}\n', r"'title' of 'd1' is not a string"),
    ],
)
def test_a_malformed_corpus_row_is_refused(corpus: str, message: str) -> None:
    rows = _jsonl([r for r in _CORPUS if r["_id"] != "d1"])
    with pytest.raises(DataIntegrityError, match=message):
        beir.parse_archive(_beir_zip(corpus=corpus + rows))


def test_a_judged_query_without_text_is_refused() -> None:
    queries = _jsonl([q for q in _QUERIES if q["_id"] != "q2"]) + '{"_id": "q2"}\n'

    with pytest.raises(DataIntegrityError, match=r"queries\.jsonl: a row lacks 'text': 'q2'"):
        beir.parse_archive(_beir_zip(queries=queries))


@pytest.mark.parametrize(
    ("qrels", "message"),
    [
        ("", r"expected the header 'query-id\\tcorpus-id\\tscore', found ''"),
        ("q\td\ts\nq1\td1\t1\n", r"found 'q\\td\\ts'"),
        ("query-id\tcorpus-id\tscore\nq1\td1\n", r"line 2: expected 'query doc grade'"),
        ("query-id\tcorpus-id\tscore\nq1\t\t1\n", r"line 2: expected 'query doc grade'"),
        ("query-id\tcorpus-id\tscore\n\tq1\t1\n", r"line 2: expected 'query doc grade'"),
        ("query-id\tcorpus-id\tscore\nq1\td1\t1.0\n", r"line 2: expected 'query doc grade'"),
        ("query-id\tcorpus-id\tscore\nq1\td1\t1_0\n", r"line 2: expected 'query doc grade'"),
        ("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t2\n", r"line 3: 'q1' judges 'd1' twice"),
    ],
)
def test_a_malformed_qrels_file_is_refused(qrels: str, message: str) -> None:
    with pytest.raises(DataIntegrityError, match=message):
        beir.parse_archive(_beir_zip(qrels=qrels))


@pytest.mark.parametrize("separator", [chr(0x2028), chr(0x2029), chr(0x85)])
def test_a_unicode_line_separator_inside_a_json_string_is_text(separator: str) -> None:
    """JSON allows these unescaped in a string; only LF ends a JSON Lines row."""
    rows = [{"_id": "d1", "title": "", "text": f"alpha{separator}beta"}, *_CORPUS[:2]]
    corpus = beir.parse_archive(
        _beir_zip(corpus="".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    )

    assert corpus.texts[0] == f"alpha{separator}beta"


def test_bytes_that_are_not_a_zip_are_refused() -> None:
    with pytest.raises(DataIntegrityError, match="not a zip archive"):
        beir.parse_archive(b"<html>proxy error</html>")


def test_a_member_that_is_not_utf8_is_refused() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tiny/corpus.jsonl", _jsonl(_CORPUS).encode() + b'{"_id": "\xff"}\n')
        archive.writestr("tiny/queries.jsonl", _jsonl(_QUERIES))
        archive.writestr("tiny/qrels/test.tsv", _QRELS)

    with pytest.raises(DataIntegrityError, match=r"tiny/corpus\.jsonl: not UTF-8"):
        beir.parse_archive(buffer.getvalue())


def test_an_escaped_lone_surrogate_is_refused() -> None:
    rows = _jsonl(_CORPUS) + '{"_id": "d\\ud800", "text": "a"}\n'

    with pytest.raises(DataIntegrityError, match="is not valid text"):
        beir.parse_archive(_beir_zip(corpus=rows))


def test_a_negative_grade_is_read_as_given() -> None:
    corpus = beir.parse_archive(_beir_zip(qrels="query-id\tcorpus-id\tscore\nq1\td1\t-1\n"))

    assert corpus.qrels == (("q1", "d1", -1),)
    assert corpus.n_queries_without_relevant == 1


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("q9\td1\t1", r"1 unknown queries \['q9'\] and 0 unknown documents"),
        ("q1\td9\t1", r"0 unknown queries \[\] and 1 unknown documents \['d9'\]"),
    ],
)
def test_a_judgement_naming_an_unknown_identifier_is_refused(row: str, message: str) -> None:
    with pytest.raises(DataIntegrityError, match=message):
        beir.parse_archive(_beir_zip(qrels=f"query-id\tcorpus-id\tscore\n{row}\n"))


# ---------------------------------------------------------------------------
# Loading from disk
# ---------------------------------------------------------------------------
def test_load_verifies_the_digest_and_the_variant(tmp_path: Path) -> None:
    data = _beir_zip()
    path = tmp_path / "tiny.zip"
    path.write_bytes(data)

    assert beir.load(path, variant="tiny", expect_sha256=hash_bytes(data)).n_documents == 3
    assert beir.load(path, variant="tiny", expect_sha256=None).archive_sha256 == hash_bytes(data)
    with pytest.raises(DataIntegrityError, match="does not match the pinned digest"):
        beir.load(path, variant="tiny", expect_sha256="0" * 64)
    with pytest.raises(DataIntegrityError, match=r"holds 'tiny', not 'scifact'"):
        beir.load(path, variant="scifact", expect_sha256=None)


def test_a_missing_archive_says_how_to_get_one(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError, match=r"fetch_data\.py --dataset beir_scifact"):
        beir.load(tmp_path / "absent.zip", variant="scifact", expect_sha256=None)


def test_every_variant_has_a_pin_of_sha256_shape() -> None:
    assert set(beir.PINNED_SHA256) == {"scifact", "nfcorpus"}
    assert all(re.fullmatch("[0-9a-f]{64}", pin) for pin in beir.PINNED_SHA256.values())
    assert beir.BEIR_URL.format(variant="scifact").endswith("/BEIR/datasets/scifact.zip")


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_both_benchmarks_are_registered() -> None:
    assert DATASET_NAMES[-2:] == ("beir_scifact", "beir_nfcorpus")


def test_a_registered_benchmark_without_an_archive_says_how_to_get_one() -> None:
    expected = (
        "beir_nfcorpus needs an archive path. BEIR archives are not in the repository -- "
        "run 'python scripts/fetch_data.py --dataset beir_nfcorpus' first, then pass its path."
    )
    with pytest.raises(DataIntegrityError, match=f"^{re.escape(expected)}$"):
        load_dataset("beir_nfcorpus")


def test_a_registered_benchmark_carries_queries_judgements_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin is read when the loader runs, so patching the mapping reaches it."""
    data = _beir_zip(top="scifact")
    path = tmp_path / "scifact.zip"
    path.write_bytes(data)
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(data))

    loaded = load_dataset("beir_scifact", archive=path)

    assert loaded.name == "beir_scifact"
    assert loaded.doc_ids == ["d1", "d10", "d2"]
    assert loaded.interactions == []
    assert loaded.queries == {"q1": "alpha beta", "q2": "gamma", "q4": "nothing relevant"}
    assert loaded.qrels == {"q1": {"d1": 0, "d2": 2}, "q2": {"d10": 1}, "q4": {"d1": 0}}
    assert loaded.provenance == {
        "kind": "beir",
        "variant": "scifact",
        "split": "test",
        "archive_sha256": hash_bytes(data),
        "n_documents": 3,
        "n_queries": 3,
        "n_judgements": 4,
        "n_queries_without_relevant": 1,
        "queries_sha256": hash_text(canonical_json(loaded.queries, indent=None)),
        "qrels_sha256": hash_text(canonical_json(loaded.qrels, indent=None)),
        "redistributable": False,
    }


def test_a_registered_benchmark_refuses_an_archive_off_its_pin(tmp_path: Path) -> None:
    path = tmp_path / "scifact.zip"
    path.write_bytes(_beir_zip(top="scifact"))

    with pytest.raises(DataIntegrityError, match="does not match the pinned digest"):
        load_dataset("beir_scifact", archive=path)


# ---------------------------------------------------------------------------
# File order, the default FAISS insertion order
# ---------------------------------------------------------------------------
def test_corpus_file_order_keeps_the_order_of_the_member() -> None:
    data = _beir_zip(corpus="\n" + _jsonl(_CORPUS))

    assert beir.corpus_file_order(data) == ("d2", "d10", "d1")
    assert beir.parse_archive(data).doc_ids == ("d1", "d10", "d2"), "parsing sorts it away"
    assert "corpus_file_order" in beir.__all__


def test_corpus_file_order_refuses_what_parsing_refuses() -> None:
    rows = _jsonl(_CORPUS) + '{"_id": "d1", "text": "again"}\n'

    with pytest.raises(DataIntegrityError, match=r"duplicate '_id' 'd1'"):
        beir.corpus_file_order(_beir_zip(corpus=rows))
    with pytest.raises(DataIntegrityError, match=r"missing \['tiny/qrels/test\.tsv'\]"):
        beir.corpus_file_order(_beir_zip(qrels=None))
