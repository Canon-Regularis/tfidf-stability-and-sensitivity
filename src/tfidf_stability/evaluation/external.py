"""Read scores, embeddings and sparse vectors made outside the package.

Every reader refuses what cannot be compared exactly: NaN, infinity, a repeated pair or
a malformed header. ``.npy`` files are parsed with ``struct``, so no numpy is imported.
Rationale: docs/engineering_notes.md#external-run-ingestion
"""

from __future__ import annotations

import ast
import json
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.profiles.query_modes import QueryMode
from tfidf_stability.profiles.user_profile import ProfileAggregation
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.similarity.scoring import InvertedIndex, ScoringScratch, taat_scores
from tfidf_stability.utils.hashing import hash_bytes, hash_floats, hash_text
from tfidf_stability.utils.io import canonical_json
from tfidf_stability.utils.numerics import Reduction
from tfidf_stability.utils.validation import DataIntegrityError
from tfidf_stability.vectorisation.sparse import CsrMatrix, SparseVector
from tfidf_stability.vectorisation.vocabulary import Vocabulary, build_vocabulary

__all__ = [
    "Embeddings",
    "ExternalDataError",
    "SparseCorpus",
    "aligned_scores",
    "grid_from_candidates",
    "grid_from_scores",
    "parse_npy",
    "query_digests",
    "read_npy",
    "read_run",
    "read_sparse_vectors",
    "run_digest",
    "score_sparse",
    "sparse_corpus",
    "sparse_queries",
]

_NPY_MAGIC = b"\x93NUMPY"
#: numpy's own cap on a header it will read.
_NPY_MAX_HEADER = 10_000
#: The two dtypes an embedding file may hold, by numpy descriptor: struct code and width.
_NPY_DTYPES = {"<f4": ("f", 4), "<f8": ("d", 8)}


class ExternalDataError(DataIntegrityError, ValueError):
    """A run, embedding file or vector file that cannot be read exactly."""


# ---------------------------------------------------------------------------
# JSON Lines
# ---------------------------------------------------------------------------
def _refuse_constant(token: str) -> NoReturn:
    raise ExternalDataError(f"the non-finite number {token} has no exact score")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ExternalDataError(f"repeated key {key!r}")
        out[key] = value
    return out


def _json_rows(path: Path | str) -> Iterator[tuple[str, dict[str, Any]]]:
    """``(where, row)`` per non-blank line; only LF ends a line."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExternalDataError(f"{path}: not UTF-8 ({exc.reason} at byte {exc.start})") from exc
    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.removesuffix("\r")
        if not line.strip():
            continue
        where = f"{path} line {number}"
        try:
            row = json.loads(line, parse_constant=_refuse_constant, object_pairs_hook=_unique_keys)
        except json.JSONDecodeError as exc:
            raise ExternalDataError(f"{where}: malformed JSON ({exc.msg})") from exc
        except ExternalDataError as exc:
            raise ExternalDataError(f"{where}: {exc}") from exc
        except (ValueError, RecursionError) as exc:
            # A 4300-digit integer or deep nesting: json raises these, not its own error.
            raise ExternalDataError(f"{where}: unreadable JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise ExternalDataError(f"{where}: expected a JSON object")
        yield where, row


def _field(row: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in row:
        raise ExternalDataError(f"{where}: lacks {key!r}")
    return row[key]


def _utf8(value: str, what: str, where: str) -> str:
    """``value`` if it encodes as UTF-8; a JSON escape can name a lone surrogate."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ExternalDataError(f"{where}: {what} {value!r} is not valid UTF-8") from exc
    return value


def _identifier(value: Any, what: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExternalDataError(f"{where}: {what} must be a non-empty string, got {value!r}")
    return _utf8(value, what, where)


def _number(value: Any, what: str, where: str) -> float:
    """A finite JSON number as the float it denotes; an integer must convert exactly."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExternalDataError(f"{where}: {what} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ExternalDataError(f"{where}: {what} {value!r} has no exact float") from exc
    if not math.isfinite(number):
        raise ExternalDataError(f"{where}: {what} {value!r} is not finite")
    if isinstance(value, int) and number != value:
        raise ExternalDataError(f"{where}: {what} {value!r} has no exact float")
    return number


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def read_run(path: Path | str) -> dict[str, dict[str, float]]:
    """Scores by document, per query, from JSON Lines ``{query_id, doc_id, score}``.

    File order is kept: it carries the tool's own order inside a score tie.
    """
    run: dict[str, dict[str, float]] = {}
    for where, row in _json_rows(path):
        query_id = _identifier(_field(row, "query_id", where), "query_id", where)
        doc_id = _identifier(_field(row, "doc_id", where), "doc_id", where)
        score = _number(_field(row, "score", where), "score", where)
        scores = run.setdefault(query_id, {})
        if doc_id in scores:
            raise ExternalDataError(f"{where}: {query_id!r} scores {doc_id!r} twice")
        scores[doc_id] = score
    if not run:
        raise ExternalDataError(f"{path} holds no scores")
    return run


def run_digest(run: Mapping[str, Mapping[str, float]]) -> str:
    """SHA-256 of the run in its own order, every float's bits included."""
    ordered = [
        [query_id, [[doc, score] for doc, score in scores.items()]]
        for query_id, scores in run.items()
    ]
    return hash_text(canonical_json(ordered, indent=None))


def query_digests(scores_by_qid: Mapping[str, Sequence[float]]) -> dict[str, str]:
    """``hash_floats`` of each query's score vector, over its binary64 bit patterns."""
    return {query_id: hash_floats(scores) for query_id, scores in scores_by_qid.items()}


def aligned_scores(
    run: Mapping[str, Mapping[str, float]], doc_ids: Sequence[str]
) -> dict[str, tuple[float, ...]]:
    """Each query's scores in ``doc_ids`` order, every document scored."""
    known = set(doc_ids)
    out: dict[str, tuple[float, ...]] = {}
    for query_id, scores in run.items():
        outside = [doc for doc in scores if doc not in known]
        if outside:
            raise ExternalDataError(
                f"query {query_id!r} scores a document outside the corpus: {outside[0]!r}"
            )
        if len(scores) != len(known):
            raise ExternalDataError(
                f"query {query_id!r} scores {len(scores)} of {len(known)} documents; "
                f"a whole-corpus run scores every one"
            )
        out[query_id] = tuple(scores[doc] for doc in doc_ids)
    return out


def _text_grid(queries: tuple[EvaluatedQuery, ...], n_degenerate: int) -> QueryGrid:
    """A grid under the text-query protocol values of ``profiles/text_queries.py``."""
    return QueryGrid(
        queries=queries,
        mode=QueryMode.TEXT,
        aggregation=ProfileAggregation.TEXT_CONCAT,
        min_interactions=0,
        exclude_profile_items=False,
        n_users=0,
        n_degenerate=n_degenerate,
        n_zero_vector=sum(1 for q in queries if q.is_zero_vector),
    )


def _corpus_table(records: Sequence[Mapping[str, Any]], doc_ids: Sequence[str]) -> AttributeTable:
    table = AttributeTable.from_records(records)
    if table.doc_ids != tuple(doc_ids):
        raise ExternalDataError("the records and doc_ids name different documents")
    return table


def grid_from_scores(
    scores_by_qid: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, Any]],
    doc_ids: Sequence[str],
) -> QueryGrid:
    """A text-query grid from whole-corpus score vectors aligned with ``doc_ids``."""
    table = _corpus_table(records, doc_ids)
    candidates = tuple(doc_ids)
    queries = []
    for query_id in sorted(scores_by_qid, key=lambda q: q.encode("utf-8")):
        scores = tuple(scores_by_qid[query_id])
        if len(scores) != len(candidates):
            raise ExternalDataError(
                f"query {query_id!r} has {len(scores)} scores for {len(candidates)} documents"
            )
        queries.append(
            EvaluatedQuery(
                query_id=query_id,
                mode=QueryMode.TEXT,
                scores=scores,
                table=table,
                candidate_ids=candidates,
                target=None,
                n_excluded=0,
            )
        )
    return _text_grid(tuple(queries), n_degenerate=0)


def grid_from_candidates(
    run: Mapping[str, Mapping[str, float]], records: Sequence[Mapping[str, Any]]
) -> QueryGrid:
    """A text-query grid whose queries each rank only the documents their list names.

    Each list keeps its run order and gets a table over its own documents; evaluate it
    with ``allow_partial=True``. A document absent from ``records`` is refused.
    """
    by_id = {str(record["doc_id"]): record for record in records}
    queries = []
    for query_id in sorted(run, key=lambda q: q.encode("utf-8")):
        listed = run[query_id]
        unknown = [doc for doc in listed if doc not in by_id]
        if unknown:
            raise ExternalDataError(
                f"query {query_id!r} lists a document outside the corpus: {unknown[0]!r}"
            )
        candidates = tuple(listed)
        queries.append(
            EvaluatedQuery(
                query_id=query_id,
                mode=QueryMode.TEXT,
                scores=tuple(listed[doc] for doc in candidates),
                table=AttributeTable.from_records([by_id[doc] for doc in candidates]),
                candidate_ids=candidates,
                target=None,
                n_excluded=len(by_id) - len(candidates),
            )
        )
    return _text_grid(tuple(queries), n_degenerate=0)


# ---------------------------------------------------------------------------
# Embeddings (.npy)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Embeddings:
    """A row-major float32 or float64 matrix, each value widened exactly."""

    n_rows: int
    n_cols: int
    dtype: str
    values: tuple[float, ...]
    sha256: str

    def row(self, i: int) -> tuple[float, ...]:
        if not 0 <= i < self.n_rows:
            raise IndexError(f"row {i} outside 0..{self.n_rows - 1}")
        return self.values[i * self.n_cols : (i + 1) * self.n_cols]


def _npy_error(name: str, problem: str) -> ExternalDataError:
    return ExternalDataError(f"{name}: {problem}")


def _npy_header(data: bytes, name: str) -> tuple[Any, int]:
    """The header dict literal of a ``.npy`` file, and the offset its data starts at."""
    if data[:6] != _NPY_MAGIC:
        raise _npy_error(name, "not a .npy file (bad magic)")
    if len(data) < 12:
        raise _npy_error(name, "truncated header")
    major = data[6]
    if major == 1:
        (header_length,) = struct.unpack_from("<H", data, 8)
        start = 10
    elif major in (2, 3):
        (header_length,) = struct.unpack_from("<I", data, 8)
        start = 12
    else:
        raise _npy_error(name, f"unsupported .npy version {major}")
    if header_length > _NPY_MAX_HEADER:
        raise _npy_error(name, f"header of {header_length} bytes exceeds {_NPY_MAX_HEADER}")
    end = start + header_length
    if len(data) < end:
        raise _npy_error(name, "truncated header")
    try:
        header = data[start:end].decode("latin1" if major < 3 else "utf-8")
        meta = ast.literal_eval(header)
    except (
        UnicodeDecodeError,
        ValueError,
        SyntaxError,
        TypeError,
        MemoryError,
        RecursionError,
    ) as exc:
        raise _npy_error(name, "unreadable header") from exc
    return meta, end


def parse_npy(data: bytes, name: str = "<bytes>") -> Embeddings:
    """Parse a ``.npy`` file holding a 2-D ``'<f4'`` or ``'<f8'`` array.

    Any other dtype, order or shape, a length that disagrees with the shape, or a
    non-finite value raises ``ExternalDataError``. Rationale: docs/engineering_notes.md#npy-reader
    """
    meta, end = _npy_header(data, name)
    if not isinstance(meta, dict) or set(meta) != {"descr", "fortran_order", "shape"}:
        raise _npy_error(name, "header must name exactly descr, fortran_order and shape")
    descr, shape = meta["descr"], meta["shape"]
    if not isinstance(descr, str) or descr not in _NPY_DTYPES:
        raise _npy_error(name, f"dtype {descr!r} is not '<f4' or '<f8'")
    if meta["fortran_order"] is not False:
        raise _npy_error(name, "Fortran order is not supported")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or not all(type(n) is int and n >= 0 for n in shape)
    ):
        raise _npy_error(name, f"shape {shape!r} is not two non-negative integers")
    code, width = _NPY_DTYPES[descr]
    n_rows, n_cols = shape
    count = n_rows * n_cols
    if len(data) - end != count * width:
        raise _npy_error(
            name, f"{len(data) - end} data bytes for shape {shape}, expected {count * width}"
        )
    values = struct.unpack(f"<{count}{code}", data[end:])
    if not all(map(math.isfinite, values)):
        raise _npy_error(name, "holds a non-finite value")
    return Embeddings(n_rows, n_cols, descr, tuple(values), hash_bytes(data))


def read_npy(path: Path | str) -> Embeddings:
    """:func:`parse_npy` of the file at ``path``."""
    return parse_npy(Path(path).read_bytes(), str(path))


# ---------------------------------------------------------------------------
# Sparse vectors (SPLADE)
# ---------------------------------------------------------------------------
def read_sparse_vectors(path: Path | str) -> dict[str, dict[str, float]]:
    """Weights by token, per id, from JSON Lines ``{id, vector: {token: weight}}``.

    A weight must be finite and non-negative; a zero weight is dropped, as a quantised
    impact of 0 adds nothing to any score.
    """
    vectors: dict[str, dict[str, float]] = {}
    for where, row in _json_rows(path):
        identifier = _identifier(_field(row, "id", where), "id", where)
        if identifier in vectors:
            raise ExternalDataError(f"{where}: duplicate id {identifier!r}")
        raw = _field(row, "vector", where)
        if not isinstance(raw, dict):
            raise ExternalDataError(f"{where}: vector must be a JSON object")
        weights: dict[str, float] = {}
        for token, value in raw.items():
            if not token:
                raise ExternalDataError(f"{where}: a token is empty")
            _utf8(token, "token", where)
            weight = _number(value, f"weight of {token!r}", where)
            if weight < 0.0:
                raise ExternalDataError(f"{where}: weight of {token!r} is negative: {weight!r}")
            if weight > 0.0:
                weights[token] = weight
        vectors[identifier] = weights
    return vectors


@dataclass(frozen=True, slots=True)
class SparseCorpus:
    """Documents as sparse vectors over their own vocabulary."""

    vocabulary: Vocabulary
    matrix: CsrMatrix
    doc_ids: tuple[str, ...]


def sparse_corpus(
    vectors: Mapping[str, Mapping[str, float]], doc_ids: Sequence[str]
) -> SparseCorpus:
    """Rows in ``doc_ids`` order over a vocabulary of every token the documents use."""
    missing = [doc for doc in doc_ids if doc not in vectors]
    extra = sorted(set(vectors) - set(doc_ids), key=lambda d: d.encode("utf-8"))
    if missing or extra:
        raise ExternalDataError(
            f"the vectors miss {len(missing)} documents {missing[:3]} and add "
            f"{len(extra)} the corpus lacks {extra[:3]}"
        )
    if not any(vectors[doc] for doc in doc_ids):
        raise ExternalDataError("no document holds a positive weight")
    vocabulary = build_vocabulary([sorted(vectors[doc]) for doc in doc_ids])
    rows = [
        SparseVector.from_mapping(
            {_known_id(vocabulary, token): w for token, w in vectors[doc].items()}, len(vocabulary)
        )
        for doc in doc_ids
    ]
    return SparseCorpus(
        vocabulary=vocabulary,
        matrix=CsrMatrix.from_rows(rows, len(vocabulary)),
        doc_ids=tuple(doc_ids),
    )


def _known_id(vocabulary: Vocabulary, token: str) -> int:
    term = vocabulary.id_of(token)
    if term is None:  # pragma: no cover - the vocabulary is built from these very tokens
        raise ExternalDataError(f"token {token!r} is missing from its own vocabulary")
    return term


def sparse_queries(
    vectors: Mapping[str, Mapping[str, float]], corpus: SparseCorpus
) -> dict[str, SparseVector]:
    """Each query over the corpus vocabulary; a token no document uses is dropped."""
    out: dict[str, SparseVector] = {}
    for query_id, weights in vectors.items():
        mapped = {}
        for token, weight in weights.items():
            term = corpus.vocabulary.id_of(token)
            if term is not None:
                mapped[term] = weight
        out[query_id] = SparseVector.from_mapping(mapped, len(corpus.vocabulary))
    return out


def score_sparse(
    queries: Mapping[str, SparseVector],
    corpus: SparseCorpus,
    records: Sequence[Mapping[str, Any]],
    *,
    policy: Reduction,
) -> QueryGrid:
    """Score each query's dot product with every document, under ``policy``.

    Unit norms leave every dot product unchanged, as for BM25; a query whose tokens no
    document uses scores zero everywhere and is kept as a zero-vector query.
    """
    table = _corpus_table(records, corpus.doc_ids)
    index = InvertedIndex.from_csr(corpus.matrix)
    norms = (1.0,) * corpus.matrix.n_rows
    scratch = ScoringScratch()
    evaluated = tuple(
        EvaluatedQuery(
            query_id=query_id,
            mode=QueryMode.TEXT,
            scores=tuple(
                taat_scores(
                    queries[query_id], index, norms, policy, query_norm=1.0, scratch=scratch
                )
            ),
            table=table,
            candidate_ids=corpus.doc_ids,
            target=None,
            n_excluded=0,
        )
        for query_id in sorted(queries, key=lambda q: q.encode("utf-8"))
    )
    return _text_grid(evaluated, n_degenerate=0)
