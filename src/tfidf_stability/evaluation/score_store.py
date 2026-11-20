"""Score matrices, one per condition, and an index that names each.

Each matrix is a version 1 ``.npy`` file of binary64 scores, one row per query in UTF-8
order and one column per document in corpus order, so every float keeps its bits.
Rationale: docs/engineering_notes.md#score-store
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from tfidf_stability.analysis.query_grid import QueryGrid
from tfidf_stability.evaluation.external import Embeddings, grid_from_scores, parse_npy
from tfidf_stability.utils.hashing import hash_bytes, hash_text
from tfidf_stability.utils.io import atomic_write_bytes, canonical_json, write_json
from tfidf_stability.utils.validation import DataIntegrityError, check_unique_ids

__all__ = [
    "INDEX_NAME",
    "ScoreStore",
    "ScoreStoreError",
    "StoreEntry",
    "StoreKind",
    "grid_scores",
    "matrix_scores",
    "npy_bytes",
]

#: The index file inside a store directory.
INDEX_NAME = "index.json"
_FORMAT = 1
#: ``<condition>.<variant>``: a condition, then a reduction policy or the run's kind.
_LABEL = re.compile(r"[a-z0-9][a-z0-9_-]*\.[a-z0-9_]+")
#: numpy leaves room for the row count to grow to this many digits.
_GROWTH_DIGITS = 21
_ALIGN = 64
_ENTRY_KEYS = frozenset(
    {"kind", "reference_label", "params", "query_ids", "degenerate_query_ids", "npy_sha256"}
)


class ScoreStoreError(DataIntegrityError, ValueError):
    """A store whose index, matrices or corpus disagree."""


class StoreKind(str, Enum):
    """Where a matrix came from."""

    LEXICAL = "lexical"
    SPARSE = "sparse"
    RUN = "run"
    MATRIX = "matrix"

    def __str__(self) -> str:
        return self.value


def npy_bytes(rows: Sequence[Sequence[float]], n_cols: int) -> bytes:
    """The ``.npy`` bytes ``numpy.save`` writes for rows of ``'<f8'``.

    Raises ``ScoreStoreError`` for a row of another length or a non-finite value.
    """
    values: list[float] = []
    for i, row in enumerate(rows):
        if len(row) != n_cols:
            raise ScoreStoreError(f"row {i} holds {len(row)} values, not {n_cols}")
        values.extend(row)
    if not all(map(math.isfinite, values)):
        raise ScoreStoreError("a non-finite score cannot be stored")
    header = f"{{'descr': '<f8', 'fortran_order': False, 'shape': ({len(rows)}, {n_cols}), }}"
    header += " " * (_GROWTH_DIGITS - len(str(len(rows))))
    pad = _ALIGN - (len(header) + 1 + 10) % _ALIGN
    encoded = (header + " " * pad + "\n").encode("latin1")
    prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded))
    return prefix + encoded + struct.pack(f"<{len(values)}d", *values)


def grid_scores(grid: QueryGrid, doc_ids: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """Each query's scores from a grid that ranks the whole of ``doc_ids``."""
    corpus = tuple(doc_ids)
    out: dict[str, tuple[float, ...]] = {}
    for query in grid.queries:
        if query.candidate_ids != corpus:
            raise ScoreStoreError(f"query {query.query_id!r} does not rank the whole corpus")
        out[query.query_id] = tuple(query.scores)
    return out


def matrix_scores(
    matrix: Embeddings,
    query_ids: Sequence[str],
    column_ids: Sequence[str],
    doc_ids: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    """Rows of a score matrix by query, columns in ``doc_ids`` order.

    The columns must name every corpus document once; a float32 matrix widens exactly.
    """
    if (matrix.n_rows, matrix.n_cols) != (len(query_ids), len(column_ids)):
        raise ScoreStoreError(
            f"a {matrix.n_rows} x {matrix.n_cols} matrix for {len(query_ids)} queries "
            f"and {len(column_ids)} documents"
        )
    check_unique_ids(query_ids)
    check_unique_ids(column_ids)
    if set(column_ids) != set(doc_ids):
        raise ScoreStoreError("the matrix columns and the corpus name different documents")
    column = {doc: j for j, doc in enumerate(column_ids)}
    order = [column[doc] for doc in doc_ids]
    out: dict[str, tuple[float, ...]] = {}
    for i, query_id in enumerate(query_ids):
        row = matrix.row(i)
        out[query_id] = tuple(row[j] for j in order)
    return out


def _check_label(label: str) -> None:
    """Refuse a label other than ``<condition>.<variant>``: it names a file."""
    if not _LABEL.fullmatch(label):
        raise ScoreStoreError(f"label {label!r} is not <condition>.<variant> in [a-z0-9_-]")


@dataclass(frozen=True, slots=True)
class StoreEntry:
    """One stored matrix: where it came from and which queries its rows hold.

    ``reference_label`` names the matrix a noise study compares this one against.
    """

    label: str
    kind: StoreKind
    reference_label: str | None
    params: Mapping[str, Any]
    query_ids: tuple[str, ...]
    degenerate_query_ids: tuple[str, ...]
    npy_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", StoreKind(self.kind))

    @property
    def condition(self) -> str:
        return self.label.split(".", 1)[0]

    @property
    def variant(self) -> str:
        return self.label.split(".", 1)[1]

    @property
    def file_name(self) -> str:
        return f"{self.label}.npy"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "reference_label": self.reference_label,
            "params": dict(self.params),
            "query_ids": list(self.query_ids),
            "degenerate_query_ids": list(self.degenerate_query_ids),
            "npy_sha256": self.npy_sha256,
        }

    @classmethod
    def from_dict(cls, label: str, row: Any) -> StoreEntry:
        _check_label(label)
        if not isinstance(row, dict) or set(row) != _ENTRY_KEYS:
            raise ScoreStoreError(f"index entry {label!r} must name exactly {sorted(_ENTRY_KEYS)}")
        try:
            return cls(
                label=label,
                kind=row["kind"],
                reference_label=row["reference_label"],
                params=dict(row["params"]),
                query_ids=tuple(row["query_ids"]),
                degenerate_query_ids=tuple(row["degenerate_query_ids"]),
                npy_sha256=row["npy_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise ScoreStoreError(f"index entry {label!r} is malformed: {exc}") from exc


def _identity(records: Sequence[Mapping[str, Any]], doc_ids: Sequence[str]) -> tuple[str, str]:
    """The records digest, as ``LoadedDataset.digest``, and the ids digest."""
    return (
        hash_text(canonical_json(list(records), indent=None)),
        hash_text(canonical_json(list(doc_ids), indent=None)),
    )


@dataclass(frozen=True, slots=True)
class ScoreStore:
    """The matrices scored over one corpus, loaded one label at a time."""

    directory: Path
    dataset: str
    records: tuple[Mapping[str, Any], ...]
    doc_ids: tuple[str, ...]
    entries: Mapping[str, StoreEntry]

    @classmethod
    def open(
        cls,
        directory: Path | str,
        *,
        dataset: str,
        records: Sequence[Mapping[str, Any]],
        doc_ids: Sequence[str],
        create: bool = False,
    ) -> ScoreStore:
        """Open the store in ``directory``, or start an empty one with ``create``.

        Raises ``ScoreStoreError`` when the index names another dataset or corpus.
        """
        path = Path(directory)
        store = cls(path, dataset, tuple(records), tuple(doc_ids), {})
        index = path / INDEX_NAME
        if not index.exists():
            if not create:
                raise ScoreStoreError(f"{index} not found; build the store first")
            store._write_index()
            return store
        try:
            document = json.loads(index.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ScoreStoreError(f"{index} is not readable JSON") from exc
        records_digest, ids_digest = _identity(records, doc_ids)
        expected = {
            "format": _FORMAT,
            "dataset": dataset,
            "records_digest": records_digest,
            "doc_ids_sha256": ids_digest,
        }
        if not isinstance(document, dict) or set(document) != {*expected, "entries"}:
            raise ScoreStoreError(f"{index} must name exactly {sorted({*expected, 'entries'})}")
        for key, value in expected.items():
            if document[key] != value:
                raise ScoreStoreError(
                    f"{index} records {key} {document[key]!r}; this corpus gives {value!r}"
                )
        if not isinstance(document["entries"], dict):
            raise ScoreStoreError(f"{index} entries must be an object")
        entries = {
            label: StoreEntry.from_dict(label, row) for label, row in document["entries"].items()
        }
        for entry in entries.values():
            if entry.reference_label is not None and entry.reference_label not in entries:
                raise ScoreStoreError(
                    f"{entry.label!r} refers to {entry.reference_label!r}, which is not stored"
                )
        return replace(store, entries=entries)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries, key=lambda label: label.encode("utf-8")))

    @property
    def covered_queries(self) -> tuple[str, ...]:
        """Every query some matrix scores or lists as degenerate, in UTF-8 order."""
        covered = {q for e in self.entries.values() for q in e.query_ids + e.degenerate_query_ids}
        return tuple(sorted(covered, key=lambda q: q.encode("utf-8")))

    def entry(self, label: str) -> StoreEntry:
        if label not in self.entries:
            raise ScoreStoreError(f"the store holds no {label!r}; it holds {list(self.labels)}")
        return self.entries[label]

    def _write_index(self) -> None:
        records_digest, ids_digest = _identity(self.records, self.doc_ids)
        write_json(
            self.directory / INDEX_NAME,
            {
                "format": _FORMAT,
                "dataset": self.dataset,
                "records_digest": records_digest,
                "doc_ids_sha256": ids_digest,
                "entries": {label: self.entries[label].as_dict() for label in self.labels},
            },
        )

    def add(
        self,
        label: str,
        scores_by_qid: Mapping[str, Sequence[float]],
        *,
        kind: StoreKind,
        params: Mapping[str, Any],
        degenerate: Sequence[str] = (),
        reference_label: str | None = None,
    ) -> ScoreStore:
        """Write ``label``'s matrix and the index, replacing any older matrix.

        ``degenerate`` lists judged queries that rank nothing. Every entry covers one judged
        set, and a reference, already stored, scores the same queries as ``label``.
        """
        _check_label(label)
        if reference_label is not None and (
            reference_label == label or reference_label not in self.entries
        ):
            raise ScoreStoreError(f"reference {reference_label!r} is not another stored label")
        query_ids = tuple(sorted(scores_by_qid, key=lambda q: q.encode("utf-8")))
        overlap = set(query_ids) & set(degenerate)
        if overlap:
            raise ScoreStoreError(f"query {sorted(overlap)[0]!r} is both scored and degenerate")
        others = [e for e in self.entries.values() if e.label != label]
        judged = {*query_ids, *degenerate}
        if others and judged != {*others[0].query_ids, *others[0].degenerate_query_ids}:
            raise ScoreStoreError(
                f"{label!r} covers {len(judged)} judged queries, not the store's set of "
                f"{len(others[0].query_ids) + len(others[0].degenerate_query_ids)}"
            )
        if reference_label is not None and self.entries[reference_label].query_ids != query_ids:
            raise ScoreStoreError(f"{label!r} scores other queries than {reference_label!r}")
        data = npy_bytes([scores_by_qid[q] for q in query_ids], len(self.doc_ids))
        entry = StoreEntry(
            label=label,
            kind=kind,
            reference_label=reference_label,
            params=dict(params),
            query_ids=query_ids,
            degenerate_query_ids=tuple(sorted(degenerate, key=lambda q: q.encode("utf-8"))),
            npy_sha256=hash_bytes(data),
        )
        atomic_write_bytes(self.directory / entry.file_name, data)
        store = replace(self, entries={**self.entries, label: entry})
        store._write_index()
        return store

    def scores(self, label: str) -> dict[str, tuple[float, ...]]:
        """``label``'s rows by query, after its file's digest is checked."""
        entry = self.entry(label)
        path = self.directory / entry.file_name
        if not path.exists():
            raise ScoreStoreError(f"{path} not found")
        data = path.read_bytes()
        if hash_bytes(data) != entry.npy_sha256:
            raise ScoreStoreError(f"{path} does not match the digest its index records")
        matrix = parse_npy(data, str(path))
        if (matrix.n_rows, matrix.n_cols) != (len(entry.query_ids), len(self.doc_ids)):
            raise ScoreStoreError(
                f"{path} is {matrix.n_rows} x {matrix.n_cols}, the index says "
                f"{len(entry.query_ids)} x {len(self.doc_ids)}"
            )
        return {query_id: matrix.row(i) for i, query_id in enumerate(entry.query_ids)}

    def grid(self, label: str) -> QueryGrid:
        """``label`` as a text-query grid, its degenerate queries counted."""
        grid = grid_from_scores(self.scores(label), self.records, self.doc_ids)
        return replace(grid, n_degenerate=len(self.entry(label).degenerate_query_ids))
