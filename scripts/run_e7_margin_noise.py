#!/usr/bin/env python3
"""E7: predict top-k changes from each query's margin and the noise.

Every stored label with a reference is compared with it; the report holds the confusion
table per pair and cutoff, and the run exits 1 when any observed change falsifies the bound.
Rationale: docs/engineering_notes.md#margin-against-noise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.loaders import LoadedDataset, load_dataset  # noqa: E402
from tfidf_stability.evaluation.margin_noise import run_margin_noise  # noqa: E402
from tfidf_stability.evaluation.score_store import ScoreStore, ScoreStoreError  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered BEIR dataset")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=None, help="default data/processed/scores/")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--depth", type=int, default=100, help="ranking depth and recall cutoff")
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    return parser


def _open_store(
    parser: argparse.ArgumentParser, args: argparse.Namespace, data: LoadedDataset
) -> ScoreStore:
    """The store for ``--dataset``; a store for another corpus ends the run."""
    directory = args.store or REPO / "data" / "processed" / "scores" / args.dataset
    try:
        return ScoreStore.open(
            directory, dataset=args.dataset, records=data.records, doc_ids=data.doc_ids
        )
    except ScoreStoreError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if min(args.cutoffs) < 1 or len(set(args.cutoffs)) != len(args.cutoffs):
        parser.error(f"--cutoffs must be distinct and at least 1, got {args.cutoffs}")
    if args.depth < 1:
        parser.error(f"--depth must be at least 1, got {args.depth}")
    data = load_dataset(args.dataset, archive=args.archive)
    store = _open_store(parser, args, data)
    qrels = {q: data.qrels[q] for q in store.covered_queries if q in data.qrels}
    payload = run_margin_noise(store, qrels, cutoffs=args.cutoffs, depth=args.depth)
    result = ExperimentResult(
        experiment="e7_margin_noise",
        parameters={
            "dataset": args.dataset,
            "cutoffs": args.cutoffs,
            "depth": args.depth,
            "n_queries": len(qrels),
            "store": {label: store.entry(label).npy_sha256 for label in store.labels},
        },
        data_provenance=data.provenance,
        payload=payload,
    )
    destination = args.output / f"e7_margin_noise_{args.dataset}.json"
    write_json(destination, result.as_dict())
    for pair in payload["pairs"]:
        for k, row in pair["by_cutoff"].items():
            confusion = row["confusion"]["query"]
            print(
                f"{pair['variant']:>28} k={k:>3}  eta={row['eta']:.3e}  "
                + "  ".join(f"{name}={count}" for name, count in confusion.items())
            )
    print(f"\nfalsifications: {payload['n_falsifications']}")
    print(f"written {destination}\nresult digest {result.digest()}")
    return 1 if payload["n_falsifications"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
