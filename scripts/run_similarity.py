#!/usr/bin/env python3
"""Score a query set and write the ranked results, with a manifest.

The plainest of the runners: no perturbation, no ablation, just the pipeline of
sections 2 and 3 run end to end. It exists so that "what does this system
actually return" is answerable without reading an experiment, and so the other
runners have something to be compared against.

Reports the margin at every ``k`` beside each ranking, because a top-k list
without its margin is exactly the artefact this study argues is misleading: two
rankings that look identical can be one ulp and one tie-break apart.

Usage::

    python scripts/run_similarity.py --dataset synthetic_tiny -o reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.loaders import load_dataset  # noqa: E402
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from tfidf_stability.ranking.attributes import AttributeTable  # noqa: E402
from tfidf_stability.ranking.margins import margin_profile  # noqa: E402
from tfidf_stability.ranking.ranker import rank_all_operators  # noqa: E402
from tfidf_stability.similarity.cosine import cosine_against_corpus  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402

DEFAULT_KS = (1, 5, 10, 20, 50)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic_tiny")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--query-length", type=int, default=6)
    parser.add_argument("--top", type=int, default=10, help="how many results to record")
    args = parser.parse_args()

    data = load_dataset(args.dataset, archive=args.archive)
    pipeline = PreprocessingPipeline()
    features = [pipeline.preprocess(str(r["text"])) for r in data.records]
    model = TfidfVectoriser().fit(features, data.doc_ids)
    table = AttributeTable.from_records(data.records)
    documents = [model.document(i) for i in range(model.n_documents)]
    ids = data.doc_ids

    stride = max(1, len(features) // max(1, args.queries))
    ks = tuple(k for k in DEFAULT_KS if k < model.n_documents)

    records = []
    for i in range(min(args.queries, len(features))):
        terms = list(features[i * stride])[: args.query_length]
        scores = cosine_against_corpus(
            TfidfVectoriser.transform_query(terms, model), documents, model.norms
        )
        ranking = rank_all_operators(scores, table)["pi"]
        records.append(
            {
                "query_id": f"q{i}",
                "terms": terms,
                "degenerate": ranking.query_degenerate,
                "top": [
                    {"rank": r + 1, "doc_id": ids[d], "score": scores[d]}
                    for r, d in enumerate(ranking.order[: args.top])
                ],
                "margins": {
                    f"k{m.k}": {
                        "value": m.value,
                        "defined": m.defined,
                        "flip_radius": m.flip_radius,
                        "exact_tie": m.is_exact_tie,
                    }
                    for m in margin_profile(ranking.sorted_scores, ks)
                },
            }
        )
        top = records[-1]["top"]
        margin = records[-1]["margins"].get("k1", {})
        if top:
            tie = " EXACT TIE at rank 1" if margin.get("exact_tie") else ""
            print(
                f"q{i:<3} top={top[0]['doc_id']:<10} score={top[0]['score']:.6f}  "
                f"m_1={margin.get('value', float('nan')):.3e}{tie}"
            )
        else:
            print(f"q{i:<3} (no results)")

    result = ExperimentResult(
        experiment="similarity",
        parameters={
            "dataset": args.dataset,
            "n_queries": len(records),
            "query_length": args.query_length,
            "top": args.top,
            "model_digest": model.digest(),
            "reduction": str(model.reduction),
        },
        data_provenance=data.provenance,
        payload={"queries": records},
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "similarity.json"
    write_json(destination, result.as_dict())
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
