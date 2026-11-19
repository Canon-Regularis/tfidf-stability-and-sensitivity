"""Runs, embeddings and sparse vectors read from external files.

Fixtures are written into ``tmp_path`` as bytes. The ``.npy`` cases are written by numpy
itself and by hand, so the reader is checked against the format's own writer.
"""

from __future__ import annotations

import io
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from tfidf_stability.evaluation.external import (
    ExternalDataError,
    aligned_scores,
    grid_from_candidates,
    grid_from_scores,
    parse_npy,
    query_digests,
    read_npy,
    read_run,
    read_sparse_vectors,
    run_digest,
    score_sparse,
    sparse_corpus,
    sparse_queries,
)
from tfidf_stability.evaluation.retrieval import evaluate_retrieval
from tfidf_stability.utils.hashing import hash_bytes, hash_floats
from tfidf_stability.utils.numerics import Reduction, reduce_sum, same_bits

_RECORDS = [{"doc_id": d, "text": ""} for d in ("d1", "d2", "d3")]
_IDS = ["d1", "d2", "d3"]


def _write(tmp_path: Path, text: str, name: str = "data.jsonl") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def _run_text() -> str:
    rows = [
        '{"query_id": "q1", "doc_id": "d2", "score": 0.5}',
        '{"query_id": "q1", "doc_id": "d1", "score": 0.5}',
        '{"query_id": "q1", "doc_id": "d3", "score": 3}',
        "",
        '{"query_id": "q2", "doc_id": "d3", "score": -0.0}',
    ]
    return "\r\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def test_a_run_keeps_its_file_order_and_exact_values(tmp_path: Path) -> None:
    run = read_run(_write(tmp_path, _run_text()))

    assert list(run) == ["q1", "q2"]
    assert list(run["q1"]) == ["d2", "d1", "d3"], "file order carries the tool's tie order"
    assert run["q1"]["d3"] == 3.0
    assert math.copysign(1.0, run["q2"]["d3"]) == -1.0, "-0.0 keeps its sign"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"query_id": "q", "doc_id": "d", "score": NaN}', "line 1: the non-finite number NaN"),
        ('{"query_id": "q", "doc_id": "d", "score": -Infinity}', "non-finite number -Infinity"),
        ('{"query_id": "q", "doc_id": "d", "score": 1e400}', "score inf is not finite"),
        ('{"query_id": "q", "doc_id": "d", "score": true}', "score must be a number, got True"),
        ('{"query_id": "q", "doc_id": "d", "score": "1"}', "score must be a number, got '1'"),
        ('{"query_id": "q", "doc_id": "d", "score": 9007199254740993}', "has no exact float"),
        (f'{{"query_id": "q", "doc_id": "d", "score": {10**400}}}', "has no exact float"),
        ('{"doc_id": "d", "score": 1}', "lacks 'query_id'"),
        ('{"query_id": "q", "score": 1}', "lacks 'doc_id'"),
        ('{"query_id": "q", "doc_id": "d"}', "lacks 'score'"),
        ('{"query_id": "", "doc_id": "d", "score": 1}', "query_id must be a non-empty string"),
        ('{"query_id": "q", "doc_id": 7, "score": 1}', "doc_id must be a non-empty string, got 7"),
        ('["q", "d", 1]', "line 1: expected a JSON object"),
        ('{"query_id": ', "line 1: malformed JSON"),
    ],
)
def test_a_malformed_run_row_is_refused(tmp_path: Path, line: str, message: str) -> None:
    with pytest.raises(ExternalDataError, match=message):
        read_run(_write(tmp_path, line + "\n"))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"query_id": "q", "doc_id": "d", "score": 1, "score": 2}', "repeated key 'score'"),
        ('{"query_id": "q", "doc_id": "d", "score": ' + "1" * 5000 + "}", "unreadable JSON"),
        (
            '{"query_id": "q", "doc_id": "d", "score": 1, "x": '
            + "[" * 100000
            + "]" * 100000
            + "}",
            "unreadable JSON",
        ),
        (
            '{"query_id": "\\ud800", "doc_id": "d", "score": 1}',
            "query_id '\\\\ud800' is not valid UTF-8",
        ),
    ],
)
def test_json_that_no_score_can_come_from_is_refused(
    tmp_path: Path, line: str, message: str
) -> None:
    with pytest.raises(ExternalDataError, match=message):
        read_run(_write(tmp_path, line + "\n"))


def test_a_repeated_pair_an_empty_file_and_bad_bytes_are_refused(tmp_path: Path) -> None:
    twice = '{"query_id": "q", "doc_id": "d", "score": 1}\n' * 2
    with pytest.raises(ExternalDataError, match="line 2: 'q' scores 'd' twice"):
        read_run(_write(tmp_path, twice))
    with pytest.raises(ExternalDataError, match="holds no scores"):
        read_run(_write(tmp_path, "\n\n"))
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b'{"query_id": "\xff"}\n')
    with pytest.raises(ExternalDataError, match="not UTF-8"):
        read_run(bad)


def test_the_run_digest_follows_order_and_every_bit(tmp_path: Path) -> None:
    run = read_run(_write(tmp_path, _run_text()))
    reordered = {"q2": run["q2"], "q1": run["q1"]}
    bumped = {**run, "q1": {**run["q1"], "d2": math.nextafter(0.5, 1.0)}}

    assert run_digest(run) == run_digest(read_run(_write(tmp_path, _run_text(), "again.jsonl")))
    assert run_digest(reordered) != run_digest(run)
    assert run_digest(bumped) != run_digest(run)
    assert query_digests({"q": (0.5, -0.0)}) == {"q": hash_floats((0.5, -0.0))}


def test_aligned_scores_need_every_document_and_no_other(tmp_path: Path) -> None:
    run = read_run(_write(tmp_path, _run_text()))

    assert aligned_scores({"q1": run["q1"]}, _IDS) == {"q1": (0.5, 0.5, 3.0)}
    with pytest.raises(ExternalDataError, match="query 'q2' scores 1 of 3 documents"):
        aligned_scores(run, _IDS)
    with pytest.raises(ExternalDataError, match="outside the corpus: 'd3'"):
        aligned_scores({"q1": run["q1"]}, ["d1", "d2"])


def test_whole_corpus_scores_evaluate_through_e5(tmp_path: Path) -> None:
    run = read_run(_write(tmp_path, _run_text()))
    grid = grid_from_scores(aligned_scores({"q1": run["q1"]}, _IDS), _RECORDS, _IDS)

    report = evaluate_retrieval(grid, {"q1": {"d3": 1}}, _IDS, k=1, depth=3, seeds=1)

    assert report.means["id_asc"]["ndcg@1"].mean == 1.0
    assert grid.queries[0].candidate_ids == tuple(_IDS)
    assert (grid.queries[0].n_excluded, grid.n_degenerate) == (0, 0)
    assert grid.provenance() | {"n_queries": 0} == {
        "query_mode": "text",
        "aggregation": "text_concat",
        "min_interactions": 0,
        "exclude_profile_items": False,
        "n_users": 0,
        "n_queries": 0,
        "n_degenerate_profiles": 0,
        "n_zero_vector_queries": 0,
        "n_candidates_min": 3,
        "n_candidates_max": 3,
    }


def test_a_score_vector_of_the_wrong_length_or_records_out_of_order_are_refused() -> None:
    with pytest.raises(ExternalDataError, match="query 'q' has 2 scores for 3 documents"):
        grid_from_scores({"q": (0.1, 0.2)}, _RECORDS, _IDS)
    with pytest.raises(ExternalDataError, match="the records and doc_ids name different documents"):
        grid_from_scores({"q": (0.1, 0.2, 0.3)}, _RECORDS, ["d3", "d2", "d1"])


def test_candidate_lists_rank_inside_themselves(tmp_path: Path) -> None:
    lines = [
        '{"query_id": "q1", "doc_id": "d3", "score": 0.25}',
        '{"query_id": "q1", "doc_id": "d1", "score": 0.75}',
    ]
    grid = grid_from_candidates(read_run(_write(tmp_path, "\n".join(lines))), _RECORDS)
    (query,) = grid.queries

    assert query.candidate_ids == ("d3", "d1")
    assert query.table.doc_ids == ("d3", "d1")
    assert query.n_excluded == 1
    assert grid.n_degenerate == 0
    report = evaluate_retrieval(
        grid, {"q1": {"d1": 1}}, _IDS, k=1, depth=2, seeds=0, allow_partial=True
    )
    assert report.means["id_asc"]["ndcg@1"].mean == 1.0
    with pytest.raises(ExternalDataError, match="lists a document outside the corpus: 'd9'"):
        grid_from_candidates({"q1": {"d9": 1.0}}, _RECORDS)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _saved(array: np.ndarray, version: tuple[int, int] | None = None) -> bytes:
    buffer = io.BytesIO()
    if version is None:
        np.save(buffer, array)
    else:
        np.lib.format.write_array(buffer, array, version=version)
    return buffer.getvalue()


def _npy(header: str, body: bytes, major: int = 1) -> bytes:
    raw = header.encode("latin1")
    size = struct.pack("<H", len(raw)) if major == 1 else struct.pack("<I", len(raw))
    return b"\x93NUMPY" + bytes([major, 0]) + size + raw + body


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("version", [None, (2, 0), (3, 0)])
def test_numpy_written_embeddings_are_read_bit_for_bit(
    dtype: type, version: tuple[int, int] | None
) -> None:
    array = (np.arange(12, dtype=np.float64).reshape(3, 4) / 7.0).astype(dtype)
    data = _saved(array, version)

    embeddings = parse_npy(data)

    assert (embeddings.n_rows, embeddings.n_cols) == (3, 4)
    assert embeddings.dtype == ("<f4" if dtype is np.float32 else "<f8")
    assert all(
        same_bits(a, float(b))
        for a, b in zip(embeddings.values, array.ravel().tolist(), strict=True)
    )
    assert embeddings.row(2) == tuple(float(v) for v in array[2])
    assert embeddings.sha256 == hash_bytes(data)


def test_an_embedding_file_is_read_from_disk_and_rows_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "e.npy"
    path.write_bytes(_saved(np.zeros((2, 3), dtype=np.float32)))

    embeddings = read_npy(path)

    assert embeddings.row(0) == (0.0, 0.0, 0.0)
    with pytest.raises(IndexError, match=r"row 2 outside 0\.\.1"):
        embeddings.row(2)


_GOOD = "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 2), }"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"NOTNUMPY" + b"\x00" * 8, r"not a \.npy file \(bad magic\)"),
        (b"\x93NUMPY\x01", "truncated header"),
        (_npy(_GOOD, b"", major=1)[:20], "truncated header"),
        (b"\x93NUMPY\x04\x00\x00\x00\x00\x00", r"unsupported \.npy version 4"),
        (_npy("{'descr': ", b""), "unreadable header"),
        (_npy("[1, 2]", b""), "header must name exactly descr, fortran_order and shape"),
        (_npy("{'descr': '<f8', 'shape': (1, 2)}", b""), "header must name exactly descr"),
        (_npy("{'descr': '<i4', 'fortran_order': False, 'shape': (1, 2)}", b""), "dtype '<i4'"),
        (_npy("{'descr': 4, 'fortran_order': False, 'shape': (1, 2)}", b""), "dtype 4"),
        (_npy("{'descr': '<f8', 'fortran_order': True, 'shape': (1, 2)}", b""), "Fortran order"),
        (_npy("{'descr': '<f8', 'fortran_order': False, 'shape': (2,)}", b""), r"shape \(2,\)"),
        (
            _npy("{'descr': '<f8', 'fortran_order': False, 'shape': (1, -2)}", b""),
            r"shape \(1, -2\) is not two non-negative integers",
        ),
        (
            _npy("{'descr': '<f8', 'fortran_order': False, 'shape': (True, 2)}", b""),
            r"shape \(True, 2\) is not two non-negative integers",
        ),
        (
            _npy("{'descr': '<f8', 'fortran_order': False, 'shape': [1, 2]}", b""),
            r"shape \[1, 2\] is not two non-negative integers",
        ),
        (_npy(_GOOD, struct.pack("<d", 1.0)), r"8 data bytes for shape \(1, 2\), expected 16"),
        (_npy(_GOOD, struct.pack("<2d", 1.0, math.nan)), "holds a non-finite value"),
    ],
)
def test_a_malformed_embedding_file_is_refused(data: bytes, message: str) -> None:
    with pytest.raises(ExternalDataError, match=message):
        parse_npy(data, "e.npy")


def test_a_header_of_exactly_numpys_limit_is_read() -> None:
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 1), }"
    padded = header + " " * (10_000 - len(header) - 1) + "\n"
    data = _npy(padded, struct.pack("<d", 2.5), major=2)

    assert parse_npy(data).values == (2.5,)


def test_a_header_over_numpys_limit_or_with_an_unhashable_key_is_refused() -> None:
    long = b"\x93NUMPY\x02\x00" + struct.pack("<I", 10_001) + b" " * 10_001
    with pytest.raises(ExternalDataError, match="header of 10001 bytes exceeds 10000"):
        parse_npy(long)
    with pytest.raises(ExternalDataError, match="unreadable header"):
        parse_npy(_npy("{[1]: 2}", b""))


def test_a_version_three_header_that_is_not_utf8_is_refused() -> None:
    raw = b"{'descr': '\xff'}"
    data = b"\x93NUMPY\x03\x00" + struct.pack("<I", len(raw)) + raw
    with pytest.raises(ExternalDataError, match="unreadable header"):
        parse_npy(data)


def test_an_empty_embedding_matrix_is_read() -> None:
    embeddings = parse_npy(_saved(np.zeros((0, 3), dtype=np.float32)))

    assert (embeddings.n_rows, embeddings.n_cols, embeddings.values) == (0, 3, ())


def test_a_fortran_ordered_numpy_array_is_refused() -> None:
    data = _saved(np.asfortranarray(np.ones((2, 3), dtype=np.float32)))
    with pytest.raises(ExternalDataError, match="Fortran order"):
        parse_npy(data)


# ---------------------------------------------------------------------------
# Sparse vectors
# ---------------------------------------------------------------------------
def _vectors_text() -> str:
    rows = [
        '{"id": "d1", "vector": {"apple": 1.5, "pear": 0.0}}',
        '{"id": "d2", "vector": {"apple": 0.5, "fig": 2}}',
        '{"id": "d3", "vector": {}}',
    ]
    return "\n".join(rows) + "\n"


def test_sparse_vectors_drop_zero_weights(tmp_path: Path) -> None:
    vectors = read_sparse_vectors(_write(tmp_path, _vectors_text()))

    assert vectors == {"d1": {"apple": 1.5}, "d2": {"apple": 0.5, "fig": 2.0}, "d3": {}}


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "d", "vector": {"a": -0.5}}', "weight of 'a' is negative"),
        ('{"id": "d", "vector": {"": 1}}', "a token is empty"),
        ('{"id": "d", "vector": [1]}', "vector must be a JSON object"),
        ('{"id": "d", "vector": {"a": false}}', "weight of 'a' must be a number"),
        ('{"vector": {}}', "lacks 'id'"),
        ('{"id": "d", "vector": {"a": 1, "a": 2}}', "repeated key 'a'"),
        ('{"id": "d", "vector": {"\\udc80": 1}}', "token '\\\\udc80' is not valid UTF-8"),
    ],
)
def test_a_malformed_vector_row_is_refused(tmp_path: Path, line: str, message: str) -> None:
    with pytest.raises(ExternalDataError, match=message):
        read_sparse_vectors(_write(tmp_path, line))


def test_a_repeated_vector_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExternalDataError, match="line 2: duplicate id 'd'"):
        read_sparse_vectors(_write(tmp_path, '{"id": "d", "vector": {}}\n' * 2))


def test_a_sparse_corpus_spans_its_own_vocabulary(tmp_path: Path) -> None:
    corpus = sparse_corpus(read_sparse_vectors(_write(tmp_path, _vectors_text())), _IDS)

    assert corpus.vocabulary.tokens == ("apple", "fig")
    assert corpus.vocabulary.df == (2, 1)
    assert corpus.matrix.row(1).values == (0.5, 2.0)
    assert corpus.matrix.row(2).nnz == 0
    assert corpus.doc_ids == tuple(_IDS)


def test_a_sparse_corpus_must_hold_exactly_the_corpus(tmp_path: Path) -> None:
    vectors = read_sparse_vectors(_write(tmp_path, _vectors_text()))

    with pytest.raises(ExternalDataError, match="no document holds a positive weight"):
        sparse_corpus({doc: {} for doc in _IDS}, _IDS)
    with pytest.raises(ExternalDataError, match=r"miss 1 documents \['d4'\] and add 0"):
        sparse_corpus(vectors, [*_IDS, "d4"])
    with pytest.raises(
        ExternalDataError, match=r"miss 0 documents \[\] and add 1 the corpus lacks \['d3'\]"
    ):
        sparse_corpus(vectors, ["d1", "d2"])


@pytest.mark.parametrize("policy", list(Reduction))
def test_sparse_scores_are_plain_dot_products_under_every_policy(
    tmp_path: Path, policy: Reduction
) -> None:
    corpus = sparse_corpus(read_sparse_vectors(_write(tmp_path, _vectors_text())), _IDS)
    queries = sparse_queries(
        {"q1": {"apple": 2.0, "fig": 0.25, "kiwi": 9.0}, "q2": {"kiwi": 1.0}}, corpus
    )

    grid = score_sparse(queries, corpus, _RECORDS, policy=policy)
    q1, q2 = grid.queries

    by_hand = [reduce_sum([2.0 * 1.5], policy), reduce_sum([2.0 * 0.5, 0.25 * 2.0], policy), 0.0]
    assert all(same_bits(a, b) for a, b in zip(q1.scores, by_hand, strict=True))
    assert queries["q2"].nnz == 0, "a token no document uses is dropped"
    assert q2.is_zero_vector
    assert (grid.n_zero_vector, grid.n_degenerate, q1.n_excluded) == (1, 0, 0)


def test_sparse_scores_evaluate_through_e5(tmp_path: Path) -> None:
    corpus = sparse_corpus(read_sparse_vectors(_write(tmp_path, _vectors_text())), _IDS)
    queries = sparse_queries({"q1": {"fig": 1.0}}, corpus)
    grid = score_sparse(queries, corpus, _RECORDS, policy=Reduction.EXACT)

    report = evaluate_retrieval(grid, {"q1": {"d2": 1}}, _IDS, k=1, depth=3, seeds=0)

    assert report.means["id_desc"]["ndcg@1"].mean == 1.0
