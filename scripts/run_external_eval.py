#!/usr/bin/env python3
"""Evaluate an external run, or SPLADE-style vectors, on a BEIR split.

A run is scored as given; ``--reference`` adds top-k agreement with another run, and for
a whole-corpus run its noise. Sparse vectors are scored here under every reduction policy.
Rationale: docs/engineering_notes.md#external-run-ingestion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.noise_floor import noise_floor_from_runs  # noqa: E402
from tfidf_stability.analysis.query_grid import QueryGrid  # noqa: E402
from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.loaders import load_dataset  # noqa: E402
from tfidf_stability.evaluation.external import (  # noqa: E402
    ExternalDataError,
    aligned_scores,
    grid_from_candidates,
    grid_from_scores,
    read_run,
    read_sparse_vectors,
    run_digest,
    score_sparse,
    sparse_corpus,
    sparse_queries,
)
from tfidf_stability.evaluation.retrieval import (  # noqa: E402
    evaluate_retrieval,
    policy_noise,
    top_k_agreement,
)
from tfidf_stability.utils.hashing import hash_file  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.utils.numerics import Reduction  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered dataset with qrels")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--label", required=True, help="the condition, e.g. bge_cpu_batch32")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, default=None, help="JSON Lines query_id, doc_id, score")
    source.add_argument("--sparse-corpus", type=Path, default=None, help="JSON Lines id, vector")
    parser.add_argument("--sparse-queries", type=Path, default=None, help="with --sparse-corpus")
    parser.add_argument("--partial", action="store_true", help="the run lists top-N per query")
    parser.add_argument("--reference", type=Path, default=None, help="a run to compare against")
    parser.add_argument("--k", type=int, default=10, help="cutoff for nDCG and MRR")
    parser.add_argument("--depth", type=int, default=100, help="ranking depth and recall cutoff")
    parser.add_argument("--seeds", type=int, default=20, help="number of shuffled tie orders")
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    return parser


def _grid(run: dict[str, dict[str, float]], data: Any, *, partial: bool) -> QueryGrid:
    if partial:
        return grid_from_candidates(run, data.records)
    return grid_from_scores(aligned_scores(run, data.doc_ids), data.records, data.doc_ids)


def _judged(run: dict[str, dict[str, float]], qrels: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {query_id: scores for query_id, scores in run.items() if query_id in qrels}


def _evaluate_run(args: argparse.Namespace, data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    run = _judged(read_run(args.run), data.qrels)
    if not run:
        raise ExternalDataError(f"{args.run} scores no judged query")
    grid = _grid(run, data, partial=args.partial)
    report = evaluate_retrieval(
        grid,
        data.qrels,
        data.doc_ids,
        k=args.k,
        depth=args.depth,
        seeds=args.seeds,
        allow_partial=args.partial,
    )
    payload: dict[str, Any] = {"report": report.as_dict()}
    inputs = {"run_sha256": hash_file(args.run), "run_digest": run_digest(run)}
    if args.reference is not None:
        reference_run = _judged(read_run(args.reference), data.qrels)
        if set(reference_run) != set(run):
            raise ExternalDataError("the run and the reference score different judged queries")
        reference = _grid(reference_run, data, partial=args.partial)
        payload["agreement_with_reference"] = top_k_agreement(reference, grid, k=args.k).as_dict()
        if not args.partial:
            payload["noise_vs_reference"] = policy_noise(
                reference, grid, k=args.k, label="reference"
            ).as_dict()
            payload["noise_floor"] = noise_floor_from_runs(
                reference.score_vectors(), {args.label: grid.score_vectors()}
            ).as_dict()
        inputs |= {"reference_sha256": hash_file(args.reference)}
    return payload, inputs


def _evaluate_sparse(args: argparse.Namespace, data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus = sparse_corpus(read_sparse_vectors(args.sparse_corpus), data.doc_ids)
    vectors = read_sparse_vectors(args.sparse_queries)
    judged = {q: v for q, v in vectors.items() if q in data.qrels}
    if not judged:
        raise ExternalDataError(f"{args.sparse_queries} holds no judged query")
    queries = sparse_queries(judged, corpus)
    grids = {p: score_sparse(queries, corpus, data.records, policy=p) for p in list(Reduction)}
    per_policy = {}
    for policy, grid in grids.items():
        report = evaluate_retrieval(
            grid, data.qrels, data.doc_ids, k=args.k, depth=args.depth, seeds=args.seeds
        )
        noise = policy_noise(grids[Reduction.EXACT], grid, k=args.k, label=str(Reduction.EXACT))
        per_policy[str(policy)] = {**report.as_dict(), "noise_vs_exact": noise.as_dict()}
    inputs = {
        "sparse_corpus_sha256": hash_file(args.sparse_corpus),
        "sparse_queries_sha256": hash_file(args.sparse_queries),
        "vocabulary_digest": corpus.vocabulary.digest(),
    }
    return {"policies": per_policy}, inputs


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.k <= args.depth:
        parser.error(f"need 1 <= --k <= --depth, got {args.k} and {args.depth}")
    if args.seeds < 0:
        parser.error(f"--seeds must be non-negative, got {args.seeds}")
    if (args.sparse_corpus is None) != (args.sparse_queries is None):
        parser.error("--sparse-corpus and --sparse-queries go together")
    if args.run is None and (args.partial or args.reference is not None):
        parser.error("--partial and --reference apply to --run")

    data = load_dataset(args.dataset, archive=args.archive)
    if not data.qrels:
        parser.error(f"{args.dataset} carries no relevance judgements")
    payload: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    try:
        payload, inputs = (
            _evaluate_run(args, data) if args.run is not None else _evaluate_sparse(args, data)
        )
    except ExternalDataError as exc:
        parser.error(str(exc))

    kind = "sparse" if args.run is None else ("partial_run" if args.partial else "run")
    result = ExperimentResult(
        experiment="external_eval",
        parameters={
            "dataset": args.dataset,
            "label": args.label,
            "kind": kind,
            "k": args.k,
            "depth": args.depth,
            "seeds": args.seeds,
            **inputs,
        },
        data_provenance=data.provenance,
        payload=payload,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"external_{args.dataset}_{args.label}.json"
    write_json(destination, result.as_dict())
    print(f"written {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
