"""The score store: numpy's npy bytes, exact rows and a checked index."""

from __future__ import annotations

import io
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from tfidf_stability.evaluation.external import parse_npy
from tfidf_stability.evaluation.retrieval import evaluate_retrieval
from tfidf_stability.evaluation.score_store import (
    INDEX_NAME,
    ScoreStore,
    ScoreStoreError,
    StoreEntry,
    StoreKind,
    grid_scores,
    matrix_scores,
    npy_bytes,
)
from tfidf_stability.utils.validation import DuplicateIdentifierError

_IDS = ["d1", "d2", "d3"]
_RECORDS = [{"doc_id": d, "text": f"text {d}"} for d in _IDS]
#: One ulp above 0.5, a signed zero and a subnormal: each must keep its bits.
_SCORES = {"q2": [0.5, math.nextafter(0.5, 1.0), -0.0], "q1": [5e-324, 0.25, 0.25]}


def _store(tmp_path: Path) -> ScoreStore:
    return ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS, create=True)


def _bits(values: tuple[float, ...] | list[float]) -> list[bytes]:
    return [struct.pack("<d", v) for v in values]


@pytest.mark.parametrize("shape", [(0, 0), (1, 0), (3, 4), (12, 1), (300, 7)])
def test_npy_bytes_equal_numpy_save(shape: tuple[int, int]) -> None:
    rows = [[0.5 * i - j for j in range(shape[1])] for i in range(shape[0])]
    buffer = io.BytesIO()
    np.save(buffer, np.array(rows, dtype="<f8").reshape(shape))

    assert npy_bytes(rows, shape[1]) == buffer.getvalue()


@pytest.mark.parametrize("n_cols", [10**36, 10**37, 10**38, 10**39])
def test_npy_header_reserves_growth_room_for_the_row_count(n_cols: int) -> None:
    """Only the row count's digits shrink the room numpy leaves for growth."""
    buffer = io.BytesIO()
    shape = (0, n_cols)
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<f8", "fortran_order": False, "shape": shape}
    )

    assert npy_bytes([], n_cols) == buffer.getvalue()


def test_npy_bytes_read_back_through_the_parser() -> None:
    matrix = parse_npy(npy_bytes([[1.0, -0.0], [2.5, 5e-324]], 2))

    assert (matrix.n_rows, matrix.n_cols, matrix.dtype) == (2, 2, "<f8")
    assert _bits(matrix.values) == _bits([1.0, -0.0, 2.5, 5e-324])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[1.0, 2.0], [3.0]], "row 1 holds 1 values, not 2"),
        ([[1.0, math.inf]], "a non-finite score cannot be stored"),
        ([[math.nan, 0.0]], "a non-finite score cannot be stored"),
    ],
)
def test_npy_bytes_refuse_ragged_or_non_finite_rows(rows: list[list[float]], message: str) -> None:
    with pytest.raises(ScoreStoreError, match=message):
        npy_bytes(rows, 2)


def test_a_stored_matrix_reads_back_bit_for_bit(tmp_path: Path) -> None:
    store = _store(tmp_path).add(
        "toy-a.naive", _SCORES, kind=StoreKind.LEXICAL, params={"n_max": 1}, degenerate=["q3"]
    )
    reopened = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS)
    scores = reopened.scores("toy-a.naive")

    assert list(scores) == ["q1", "q2"], "rows follow UTF-8 query order"
    assert all(_bits(scores[q]) == _bits(_SCORES[q]) for q in scores)
    assert reopened.entries == store.entries
    entry = reopened.entry("toy-a.naive")
    assert (entry.condition, entry.variant, entry.kind) == ("toy-a", "naive", StoreKind.LEXICAL)
    assert entry.degenerate_query_ids == ("q3",)
    assert (tmp_path / "toy-a.naive.npy").exists()


def test_a_grid_counts_its_degenerate_queries_and_evaluates(tmp_path: Path) -> None:
    store = _store(tmp_path).add(
        "toy-a.naive", _SCORES, kind=StoreKind.LEXICAL, params={}, degenerate=["q3"]
    )
    grid = store.grid("toy-a.naive")
    qrels = {"q1": {"d2": 1}, "q2": {"d1": 1}, "q3": {"d1": 1}}
    report = evaluate_retrieval(grid, qrels, _IDS, k=1, depth=2, seeds=0)

    assert (grid.n_degenerate, len(grid.queries)) == (1, 2)
    assert [q.degenerate for q in report.queries] == [False, False, True]
    assert grid_scores(grid, _IDS)["q1"] == tuple(_SCORES["q1"])


def test_labels_sort_and_a_reference_must_be_stored(tmp_path: Path) -> None:
    store = _store(tmp_path).add("b.exact", _SCORES, kind=StoreKind.LEXICAL, params={})
    store = store.add(
        "b.naive", _SCORES, kind=StoreKind.LEXICAL, params={}, reference_label="b.exact"
    )

    assert store.labels == ("b.exact", "b.naive")
    assert store.covered_queries == ("q1", "q2")
    assert store.add(
        "c.run", {"q2": [0.0] * 3}, kind=StoreKind.RUN, params={}, degenerate=["q1"]
    ).covered_queries == ("q1", "q2")
    assert store.entry("b.naive").reference_label == "b.exact"
    with pytest.raises(ScoreStoreError, match=r"reference 'b\.pairwise' is not another stored"):
        store.add("b.x", _SCORES, kind=StoreKind.RUN, params={}, reference_label="b.pairwise")
    with pytest.raises(ScoreStoreError, match=r"reference 'b\.x' is not another stored"):
        store.add("b.x", _SCORES, kind=StoreKind.RUN, params={}, reference_label="b.x")


def test_every_entry_covers_the_store_judged_set(tmp_path: Path) -> None:
    """A later entry may mark other queries degenerate but not judge another set."""
    store = _store(tmp_path).add("a.exact", _SCORES, kind=StoreKind.LEXICAL, params={})

    with pytest.raises(ScoreStoreError, match=r"'b\.run' covers 3 judged queries, not .* of 2"):
        store.add("b.run", _SCORES, kind=StoreKind.RUN, params={}, degenerate=["q3"])
    with pytest.raises(ScoreStoreError, match=r"'b\.run' covers 1 judged queries"):
        store.add("b.run", {"q1": [0.0] * 3}, kind=StoreKind.RUN, params={})
    replaced = store.add("a.exact", {"q9": [0.0] * 3}, kind=StoreKind.LEXICAL, params={})
    assert replaced.covered_queries == ("q9",), "a lone entry may be replaced by any set"


def test_the_judged_set_counts_scored_and_degenerate_queries(tmp_path: Path) -> None:
    """The store's judged set counts its degenerate queries with its scored ones."""
    store = _store(tmp_path).add(
        "a.exact", _SCORES, kind=StoreKind.LEXICAL, params={}, degenerate=["q3"]
    )

    with pytest.raises(ScoreStoreError, match=r"covers 1 judged queries, not .* set of 3$"):
        store.add("b.run", {"q1": [0.0] * 3}, kind=StoreKind.RUN, params={})


def test_the_index_records_format_one(tmp_path: Path) -> None:
    """An index names its format as 1, the version a reader checks."""
    _filled(tmp_path)

    document = json.loads((tmp_path / INDEX_NAME).read_bytes())
    assert document["format"] == 1
    assert sorted(document["entries"]) == ["a.exact", "a.naive"]


def test_a_reference_must_score_the_same_queries(tmp_path: Path) -> None:
    store = _store(tmp_path).add("a.exact", _SCORES, kind=StoreKind.LEXICAL, params={})

    with pytest.raises(ScoreStoreError, match=r"'a\.naive' scores other queries than 'a\.exact'"):
        store.add(
            "a.naive",
            {"q1": _SCORES["q1"]},
            kind=StoreKind.LEXICAL,
            params={},
            degenerate=["q2"],
            reference_label="a.exact",
        )


@pytest.mark.parametrize("label", ["nodot", "Upper.naive", "../x.naive", "a.b.c", ".naive"])
def test_a_label_that_is_not_condition_dot_variant_is_refused(tmp_path: Path, label: str) -> None:
    with pytest.raises(ScoreStoreError, match=r"is not <condition>\.<variant>"):
        _store(tmp_path).add(label, _SCORES, kind=StoreKind.RUN, params={})


def test_a_query_both_scored_and_degenerate_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ScoreStoreError, match="query 'q1' is both scored and degenerate"):
        _store(tmp_path).add("a.run", _SCORES, kind=StoreKind.RUN, params={}, degenerate=["q1"])


def test_a_short_row_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ScoreStoreError, match="row 0 holds 2 values, not 3"):
        _store(tmp_path).add("a.run", {"q1": [0.5, 0.25]}, kind=StoreKind.RUN, params={})


def test_an_unknown_label_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ScoreStoreError, match=r"the store holds no 'a.run'; it holds \[\]"):
        _store(tmp_path).scores("a.run")


def test_a_missing_index_is_refused_unless_created(tmp_path: Path) -> None:
    with pytest.raises(ScoreStoreError, match="build the store first"):
        ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS)


@pytest.mark.parametrize(
    ("dataset", "records", "ids", "key"),
    [
        ("other", _RECORDS, _IDS, "dataset"),
        ("toy", [*_RECORDS[:2], {"doc_id": "d3", "text": "changed"}], _IDS, "records_digest"),
        ("toy", _RECORDS, ["d1", "d3", "d2"], "doc_ids_sha256"),
    ],
)
def test_a_store_for_another_corpus_is_refused(
    tmp_path: Path, dataset: str, records: list[dict[str, str]], ids: list[str], key: str
) -> None:
    _store(tmp_path)

    with pytest.raises(ScoreStoreError, match=f"records {key} "):
        ScoreStore.open(tmp_path, dataset=dataset, records=records, doc_ids=ids)


def _rewrite_index(tmp_path: Path, change: object) -> None:
    index = tmp_path / INDEX_NAME
    document = json.loads(index.read_bytes())
    change(document)  # type: ignore[operator]
    index.write_bytes(json.dumps(document).encode("utf-8"))


def _filled(tmp_path: Path) -> ScoreStore:
    store = _store(tmp_path).add("a.exact", _SCORES, kind=StoreKind.LEXICAL, params={})
    return store.add(
        "a.naive", _SCORES, kind=StoreKind.LEXICAL, params={}, reference_label="a.exact"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("format"), "must name exactly"),
        (lambda d: d.update(entries=[]), "entries must be an object"),
        (lambda d: d["entries"].update({"../up.naive": {}}), "is not <condition>.<variant>"),
        (lambda d: d["entries"]["a.exact"].pop("kind"), "index entry 'a.exact' must name"),
        (lambda d: d["entries"]["a.exact"].update(kind="other"), "'a.exact' is malformed"),
        (lambda d: d["entries"]["a.exact"].update(query_ids=3), "'a.exact' is malformed"),
        (lambda d: d["entries"].pop("a.exact"), "refers to 'a.exact', which is not stored"),
    ],
)
def test_a_damaged_index_is_refused(tmp_path: Path, change: object, message: str) -> None:
    _filled(tmp_path)
    _rewrite_index(tmp_path, change)

    with pytest.raises(ScoreStoreError, match=message):
        ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS)


def test_an_index_that_is_not_json_is_refused(tmp_path: Path) -> None:
    (tmp_path / INDEX_NAME).write_bytes(b"{not json")

    with pytest.raises(ScoreStoreError, match="is not readable JSON"):
        ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS)


def test_a_changed_or_missing_matrix_is_refused(tmp_path: Path) -> None:
    store = _filled(tmp_path)
    path = tmp_path / "a.exact.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))

    with pytest.raises(ScoreStoreError, match="does not match the digest its index records"):
        store.scores("a.exact")
    (tmp_path / "a.naive.npy").unlink()
    with pytest.raises(ScoreStoreError, match=r"a\.naive\.npy not found"):
        store.scores("a.naive")


def test_a_matrix_whose_shape_disagrees_with_the_index_is_refused(tmp_path: Path) -> None:
    store = _filled(tmp_path)
    data = npy_bytes([[0.5, 0.25, 0.0]], 3)
    (tmp_path / "a.exact.npy").write_bytes(data)
    _rewrite_index(
        tmp_path, lambda d: d["entries"]["a.exact"].update(npy_sha256=parse_npy(data).sha256)
    )
    store = ScoreStore.open(tmp_path, dataset="toy", records=_RECORDS, doc_ids=_IDS)

    with pytest.raises(ScoreStoreError, match=r"is 1 x 3, the index says 2 x 3"):
        store.scores("a.exact")


def test_an_entry_round_trips_through_its_dict() -> None:
    entry = StoreEntry("a.run", "run", None, {"x": 1}, ("q1",), (), "00")  # type: ignore[arg-type]

    assert entry.kind is StoreKind.RUN
    assert str(entry.kind) == "run"
    assert StoreEntry.from_dict("a.run", entry.as_dict()) == entry


def test_matrix_columns_are_reordered_into_corpus_order() -> None:
    matrix = parse_npy(npy_bytes([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], 3))

    scores = matrix_scores(matrix, ["q1", "q2"], ["d3", "d1", "d2"], _IDS)

    assert scores == {"q1": (1.0, 2.0, 3.0), "q2": (4.0, 5.0, 6.0)}


def test_a_float32_matrix_widens_exactly() -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.array([[0.1, 0.2, 0.3]], dtype="<f4"))

    scores = matrix_scores(parse_npy(buffer.getvalue()), ["q1"], _IDS, _IDS)

    assert scores["q1"] == tuple(float(v) for v in np.array([0.1, 0.2, 0.3], dtype="<f4"))


@pytest.mark.parametrize(
    ("queries", "columns", "error", "message"),
    [
        (["q1"], _IDS[:2], ScoreStoreError, "a 1 x 3 matrix for 1 queries and 2 documents"),
        (["q1", "q2"], _IDS, ScoreStoreError, "a 1 x 3 matrix for 2 queries"),
        (["q1"], ["d1", "d2", "d9"], ScoreStoreError, "name different documents"),
        (["q1"], ["d1", "d1", "d2"], DuplicateIdentifierError, "'d1' appears"),
    ],
)
def test_a_matrix_that_does_not_fit_the_corpus_is_refused(
    queries: list[str], columns: list[str], error: type[Exception], message: str
) -> None:
    matrix = parse_npy(npy_bytes([[1.0, 2.0, 3.0]], 3))

    with pytest.raises(error, match=message):
        matrix_scores(matrix, queries, columns, _IDS)


def test_a_partial_grid_has_no_whole_corpus_scores(tmp_path: Path) -> None:
    grid = _store(tmp_path).add("a.run", _SCORES, kind=StoreKind.RUN, params={}).grid("a.run")

    with pytest.raises(ScoreStoreError, match="query 'q1' does not rank the whole corpus"):
        grid_scores(grid, _IDS[:2])
