#!/usr/bin/env python3
"""E5: retrieval effectiveness on a BEIR test split, per policy and tie order.

Fits once, scores every judged query under each reduction policy, and reports
nDCG@k, Recall@depth and MRR@k under ``id_asc``, ``id_desc`` and seeded shuffles.
Rationale: docs/engineering_notes.md#identifier-tie-orders
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.loaders import load_dataset  # noqa: E402
from tfidf_stability.evaluation.retrieval import (  # noqa: E402
    evaluate_retrieval,
    score_text_queries,
)
from tfidf_stability.preprocessing.pipeline import (  # noqa: E402
    PreprocessingPipeline,
    preprocess_records,
)
from tfidf_stability.profiles.text_queries import text_queries  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.utils.numerics import Reduction  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered dataset with qrels")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    parser.add_argument("--k", type=int, default=10, help="cutoff for nDCG and MRR")
    parser.add_argument("--depth", type=int, default=100, help="ranking depth and recall cutoff")
    parser.add_argument("--seeds", type=int, default=20, help="number of shuffled tie orders")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=[str(p) for p in Reduction],
        default=[str(p) for p in Reduction],
    )
    parser.add_argument("--limit", type=int, default=None, help="first N queries by UTF-8 id")
    args = parser.parse_args(argv)
    if not 1 <= args.k <= args.depth:
        parser.error(f"need 1 <= --k <= --depth, got {args.k} and {args.depth}")
    if args.seeds < 0:
        parser.error(f"--seeds must be non-negative, got {args.seeds}")
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be at least 1, got {args.limit}")
    # Declaration order and no repeats, so the result digest names a set.
    policies = [p for p in Reduction if str(p) in args.policies]

    data = load_dataset(args.dataset, archive=args.archive)
    if not data.qrels:
        parser.error(f"{args.dataset} carries no relevance judgements")
    judged = sorted(data.qrels, key=lambda q: q.encode("utf-8"))[: args.limit]
    qrels = {q: data.qrels[q] for q in judged}
    if not any(grade > 0 for judgements in qrels.values() for grade in judgements.values()):
        parser.error(f"no judged query among the first {len(judged)} has a relevant document")

    pipeline = PreprocessingPipeline()
    features = preprocess_records(pipeline, data.records)
    # Reduction enters only the norms: one fit serves every policy, and a test
    # pins the replaced model to a refit bit for bit.
    fitted = TfidfVectoriser(reduction=Reduction.NAIVE).fit(features, data.doc_ids)
    queries = text_queries(
        {q: data.queries[q] for q in judged}, pipeline, n_documents=fitted.n_documents
    )

    per_policy: dict[str, object] = {}
    digests: dict[str, str] = {}
    for policy in policies:
        model = dataclasses.replace(fitted, norms=fitted.matrix.row_norms(policy), reduction=policy)
        grid = score_text_queries(queries, model, data.records, data.doc_ids)
        report = evaluate_retrieval(
            grid, qrels, data.doc_ids, k=args.k, depth=args.depth, seeds=args.seeds
        )
        per_policy[str(policy)] = report.as_dict()
        digests[str(policy)] = model.digest()
        means = report.means
        print(
            f"{policy!s:>9}  "
            + "  ".join(
                f"{label} ndcg@{args.k}={means[label][f'ndcg@{args.k}'].mean:.4f}"
                for label in ("id_asc", "id_desc")
            )
        )

    result = ExperimentResult(
        experiment="retrieval_eval",
        parameters={
            "dataset": args.dataset,
            "k": args.k,
            "depth": args.depth,
            "seeds": args.seeds,
            "n_queries": len(judged),
            "policies": [str(p) for p in policies],
            "preprocessing_digest": pipeline.digest(),
            "model_digests": digests,
        },
        data_provenance=data.provenance,
        payload={"policies": per_policy},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"retrieval_{args.dataset}.json"
    write_json(destination, result.as_dict())
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
