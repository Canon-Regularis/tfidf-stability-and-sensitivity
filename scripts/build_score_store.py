#!/usr/bin/env python3
"""Score every judged query of a BEIR split once and store the matrices.

Lexical conditions are TF-IDF to bigrams and unigram BM25 with exact and Lucene byte
lengths, each under every policy; runs, sparse vectors and matrices join under given labels.
Rationale: docs/engineering_notes.md#score-store
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.datasets.loaders import LoadedDataset, load_dataset  # noqa: E402
from tfidf_stability.evaluation.external import (  # noqa: E402
    ExternalDataError,
    aligned_scores,
    read_npy,
    read_run,
    read_sparse_vectors,
    run_digest,
    score_sparse,
    sparse_corpus,
    sparse_queries,
)
from tfidf_stability.evaluation.retrieval import score_text_queries  # noqa: E402
from tfidf_stability.evaluation.score_store import (  # noqa: E402
    ScoreStore,
    ScoreStoreError,
    StoreKind,
    grid_scores,
    matrix_scores,
)
from tfidf_stability.preprocessing.pipeline import (  # noqa: E402
    PreprocessingConfig,
    PreprocessingPipeline,
    preprocess_records,
)
from tfidf_stability.profiles.text_queries import text_queries  # noqa: E402
from tfidf_stability.utils.hashing import hash_file  # noqa: E402
from tfidf_stability.utils.numerics import Reduction  # noqa: E402
from tfidf_stability.vectorisation.bm25 import Bm25Model, Bm25Vectoriser, LengthNorm  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser  # noqa: E402

#: The lexical conditions: label stem, weighting, length normalisation and longest n-gram.
LEXICAL: tuple[tuple[str, str, LengthNorm | None, int], ...] = (
    ("tfidf_n2", "tfidf", None, 2),
    ("bm25-exact_n1", "bm25", LengthNorm.EXACT, 1),
    ("bm25-lucene_byte_n1", "bm25", LengthNorm.LUCENE_BYTE, 1),
)
#: Every policy, exact first: the others name it as their reference, so it is stored first.
_POLICIES: tuple[Reduction, ...] = (
    Reduction.EXACT,
    *(p for p in list(Reduction) if p is not Reduction.EXACT),
)


def _under(model: TfidfModel | Bm25Model, policy: Reduction) -> TfidfModel | Bm25Model:
    """``model`` scored under ``policy``: only TF-IDF norms depend on it."""
    if isinstance(model, Bm25Model):
        return dataclasses.replace(model, reduction=policy)
    return dataclasses.replace(model, norms=model.matrix.row_norms(policy), reduction=policy)


def _split(value: str, n_parts: int, flag: str) -> tuple[str, list[str]]:
    """``LABEL=A,B,...`` as the label and exactly ``n_parts`` paths."""
    label, _, rest = value.partition("=")
    parts = rest.split(",") if rest else []
    if not label or len(parts) != n_parts or not all(parts):
        raise ValueError(f"{flag} expects LABEL={','.join(['PATH'] * n_parts)}, got {value!r}")
    return label, parts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered BEIR dataset")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default data/processed/scores/")
    parser.add_argument("--limit", type=int, default=None, help="first N judged queries")
    parser.add_argument("--no-lexical", action="store_true", help="skip the lexical conditions")
    parser.add_argument("--run", action="append", default=[], help="LABEL=PATH, a whole run")
    parser.add_argument(
        "--sparse", action="append", default=[], help="CONDITION=CORPUS,QUERIES vectors"
    )
    parser.add_argument(
        "--matrix", action="append", default=[], help="LABEL=NPY,QUERY_IDS,DOC_IDS scores"
    )
    parser.add_argument(
        "--reference", action="append", default=[], help="LABEL=REFERENCE for a run or matrix"
    )
    return parser


def _degenerate(scored: Mapping[str, Any], judged: Sequence[str]) -> list[str]:
    return [q for q in judged if q not in scored]


def _add_lexical(store: ScoreStore, data: LoadedDataset, judged: Sequence[str]) -> ScoreStore:
    for stem, weighting, length_norm, n_max in LEXICAL:
        pipeline = PreprocessingPipeline(PreprocessingConfig(n_max=n_max))
        features = preprocess_records(pipeline, data.records)
        fitted: TfidfModel | Bm25Model = (
            Bm25Vectoriser(length_norm=length_norm).fit(features, data.doc_ids)
            if length_norm is not None
            else TfidfVectoriser().fit(features, data.doc_ids)
        )
        queries = text_queries(
            {q: data.queries[q] for q in judged}, pipeline, n_documents=fitted.n_documents
        )
        for policy in _POLICIES:
            model = _under(fitted, policy)
            grid = score_text_queries(queries, model, data.records, data.doc_ids)
            scores = grid_scores(grid, data.doc_ids)
            store = store.add(
                f"{stem}.{policy}",
                scores,
                kind=StoreKind.LEXICAL,
                params={
                    "weighting": weighting,
                    "length_norm": None if length_norm is None else str(length_norm),
                    "n_max": n_max,
                    "policy": str(policy),
                    "preprocessing_digest": pipeline.digest(),
                    "model_digest": model.digest(),
                },
                degenerate=_degenerate(scores, judged),
                reference_label=None if policy is Reduction.EXACT else f"{stem}.exact",
            )
            print(f"stored {stem}.{policy}: {len(scores)} queries")
    return store


def _add_sparse(
    store: ScoreStore, condition: str, paths: list[str], data: LoadedDataset, judged: list[str]
) -> ScoreStore:
    """Score sparse vectors under every policy, each referring to exact."""
    corpus_path, queries_path = paths
    corpus = sparse_corpus(read_sparse_vectors(corpus_path), data.doc_ids)
    vectors = sparse_queries(read_sparse_vectors(queries_path), corpus)
    keep = set(judged)
    judged_vectors = {q: v for q, v in vectors.items() if q in keep}
    for policy in _POLICIES:
        grid = score_sparse(judged_vectors, corpus, data.records, policy=policy)
        scores = grid_scores(grid, data.doc_ids)
        store = store.add(
            f"{condition}.{policy}",
            scores,
            kind=StoreKind.SPARSE,
            params={
                "policy": str(policy),
                "corpus_sha256": hash_file(corpus_path),
                "queries_sha256": hash_file(queries_path),
            },
            degenerate=_degenerate(scores, judged),
            reference_label=None if policy is Reduction.EXACT else f"{condition}.exact",
        )
        print(f"stored {condition}.{policy}")
    return store


def _external_scores(
    kind: StoreKind, paths: list[str], doc_ids: Sequence[str]
) -> tuple[dict[str, tuple[float, ...]], dict[str, str]]:
    """A run's or a matrix's scores in corpus order, with the digests of its files."""
    if kind is StoreKind.RUN:
        run = read_run(paths[0])
        params = {"run_sha256": hash_file(paths[0]), "run_digest": run_digest(run)}
        return aligned_scores(run, doc_ids), params
    npy, query_file, doc_file = paths
    query_ids = Path(query_file).read_text(encoding="utf-8").split()
    column_ids = Path(doc_file).read_text(encoding="utf-8").split()
    params = {
        "npy_sha256": hash_file(npy),
        "query_ids_sha256": hash_file(query_file),
        "doc_ids_sha256": hash_file(doc_file),
    }
    return matrix_scores(read_npy(npy), query_ids, column_ids, doc_ids), params


def _in_dependency_order(labels: Sequence[str], references: Mapping[str, str]) -> list[str]:
    """``labels``, each after the reference it names when that is given here."""
    unknown = sorted(set(references) - set(labels))
    if unknown:
        raise ValueError(f"--reference names {unknown[0]!r}, which no --run or --matrix stores")
    ordered: list[str] = []
    pending = list(labels)
    while pending:
        ready = [
            label
            for label in pending
            if references.get(label) not in pending or references.get(label) == label
        ]
        if not ready:
            raise ValueError(f"--reference labels form a cycle through {pending[0]!r}")
        ordered += ready
        pending = [label for label in pending if label not in ready]
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be at least 1, got {args.limit}")
    sparse: list[tuple[str, list[str]]] = []
    external: dict[str, tuple[StoreKind, list[str]]] = {}
    references: dict[str, str] = {}
    order: list[str] = []
    try:
        sparse = [_split(value, 2, "--sparse") for value in args.sparse]
        for kind, flag, values, n_parts in (
            (StoreKind.RUN, "--run", args.run, 1),
            (StoreKind.MATRIX, "--matrix", args.matrix, 3),
        ):
            for label, parts in (_split(value, n_parts, flag) for value in values):
                if label in external:
                    raise ValueError(f"{label!r} is given twice")
                external[label] = (kind, parts)
        pairs = [_split(value, 1, "--reference") for value in args.reference]
        references = {label: parts[0] for label, parts in pairs}
        order = _in_dependency_order(list(external), references)
    except ValueError as exc:
        parser.error(str(exc))

    data = load_dataset(args.dataset, archive=args.archive)
    if not data.qrels:
        parser.error(f"{args.dataset} carries no relevance judgements")
    judged = sorted(data.qrels, key=lambda q: q.encode("utf-8"))[: args.limit]
    keep = set(judged)
    out = args.out or REPO / "data" / "processed" / "scores" / args.dataset
    store = ScoreStore.open(
        out, dataset=args.dataset, records=data.records, doc_ids=data.doc_ids, create=True
    )
    try:
        if not args.no_lexical:
            store = _add_lexical(store, data, judged)
        for condition, paths in sparse:
            store = _add_sparse(store, condition, paths, data, judged)
        for label in order:
            kind, parts = external[label]
            aligned, params = _external_scores(kind, parts, data.doc_ids)
            scores = {q: s for q, s in aligned.items() if q in keep}
            store = store.add(
                label,
                scores,
                kind=kind,
                params=params,
                degenerate=_degenerate(scores, judged),
                reference_label=references.get(label),
            )
            print(f"stored {label}")
    except (ExternalDataError, ScoreStoreError, OSError) as exc:
        parser.error(str(exc))
    print(f"\nstore {out}: {len(store.labels)} labels over {len(judged)} judged queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
