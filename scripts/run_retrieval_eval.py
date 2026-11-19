#!/usr/bin/env python3
"""E5: BEIR retrieval effectiveness per weighting, policy and tie order.

Fits TF-IDF or BM25 once, scores every judged query under each reduction policy, measures
each policy's noise against ``exact`` and reports nDCG@k, Recall@depth and MRR@k.
Rationale: docs/engineering_notes.md#policy-noise-against-exact
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
    policy_noise,
    score_text_queries,
)
from tfidf_stability.preprocessing.pipeline import (  # noqa: E402
    PreprocessingConfig,
    PreprocessingPipeline,
    preprocess_records,
)
from tfidf_stability.profiles.text_queries import text_queries  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.utils.numerics import Reduction  # noqa: E402
from tfidf_stability.vectorisation.bm25 import (  # noqa: E402
    Bm25Model,
    Bm25Vectoriser,
    LengthNorm,
)
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser  # noqa: E402


def _under(model: TfidfModel | Bm25Model, policy: Reduction) -> TfidfModel | Bm25Model:
    """``model`` scored under ``policy``: only TF-IDF norms depend on it."""
    if isinstance(model, Bm25Model):
        return dataclasses.replace(model, reduction=policy)
    return dataclasses.replace(model, norms=model.matrix.row_norms(policy), reduction=policy)


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--weighting", choices=("tfidf", "bm25"), default="tfidf")
    parser.add_argument("--n-max", type=int, default=2, help="longest n-gram in the features")
    parser.add_argument("--bm25-k1", default="0.9", help="BM25 k1, as a decimal string")
    parser.add_argument("--bm25-b", default="0.4", help="BM25 b, as a decimal string")
    parser.add_argument("--length-norm", choices=[str(n) for n in LengthNorm], default="exact")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.k <= args.depth:
        parser.error(f"need 1 <= --k <= --depth, got {args.k} and {args.depth}")
    if args.seeds < 0:
        parser.error(f"--seeds must be non-negative, got {args.seeds}")
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be at least 1, got {args.limit}")
    if args.n_max < 1:
        parser.error(f"--n-max must be at least 1, got {args.n_max}")
    try:
        bm25 = Bm25Vectoriser(k1=args.bm25_k1, b=args.bm25_b, length_norm=args.length_norm)
    except ValueError as exc:
        parser.error(str(exc))
    # Declaration order and no repeats, so the result digest names a set.
    policies = [p for p in Reduction if str(p) in args.policies]

    data = load_dataset(args.dataset, archive=args.archive)
    if not data.qrels:
        parser.error(f"{args.dataset} carries no relevance judgements")
    judged = sorted(data.qrels, key=lambda q: q.encode("utf-8"))[: args.limit]
    qrels = {q: data.qrels[q] for q in judged}
    if not any(grade > 0 for judgements in qrels.values() for grade in judgements.values()):
        parser.error(f"no judged query among the first {len(judged)} has a relevant document")

    pipeline = PreprocessingPipeline(PreprocessingConfig(n_max=args.n_max))
    features = preprocess_records(pipeline, data.records)
    # Reduction enters no weight: one fit serves every policy, and tests pin the
    # swapped model to a refit bit for bit.
    fitted: TfidfModel | Bm25Model = (
        bm25.fit(features, data.doc_ids)
        if args.weighting == "bm25"
        else TfidfVectoriser().fit(features, data.doc_ids)
    )
    queries = text_queries(
        {q: data.queries[q] for q in judged}, pipeline, n_documents=fitted.n_documents
    )
    # The exact-sum grid is the reference every policy's noise is measured against.
    exact_grid = score_text_queries(
        queries, _under(fitted, Reduction.EXACT), data.records, data.doc_ids
    )

    per_policy: dict[str, object] = {}
    digests: dict[str, str] = {}
    for policy in policies:
        model = _under(fitted, policy)
        grid = (
            exact_grid
            if policy is Reduction.EXACT
            else score_text_queries(queries, model, data.records, data.doc_ids)
        )
        report = evaluate_retrieval(
            grid, qrels, data.doc_ids, k=args.k, depth=args.depth, seeds=args.seeds
        )
        noise = policy_noise(exact_grid, grid, k=args.k, label=str(Reduction.EXACT))
        per_policy[str(policy)] = {**report.as_dict(), "noise_vs_exact": noise.as_dict()}
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
            "n_max": args.n_max,
            "weighting": (
                {
                    "scheme": "bm25",
                    "k1": args.bm25_k1,
                    "b": args.bm25_b,
                    "length_norm": args.length_norm,
                    "idf": "lucene",
                }
                if args.weighting == "bm25"
                else {"scheme": "tfidf"}
            ),
            "preprocessing_digest": pipeline.digest(),
            "model_digests": digests,
        },
        data_provenance=data.provenance,
        payload={"policies": per_policy},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    scheme = f"bm25-{args.length_norm}" if args.weighting == "bm25" else "tfidf"
    destination = args.output / f"retrieval_{args.dataset}_{scheme}_n{args.n_max}.json"
    write_json(destination, result.as_dict())
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
