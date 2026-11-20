#!/usr/bin/env python3
"""E6: exact and near ties at the cutoff, and metric ranges over tie orders.

Reads every scoring condition from the score store; a row the store lacks is reported
as missing with the issue that supplies it.
Rationale: docs/engineering_notes.md#tie-census
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.loaders import LoadedDataset, load_dataset  # noqa: E402
from tfidf_stability.evaluation.census import run_census  # noqa: E402
from tfidf_stability.evaluation.score_store import ScoreStore, ScoreStoreError  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered BEIR dataset")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=None, help="default data/processed/scores/")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--depth", type=int, default=100, help="ranking depth and recall cutoff")
    parser.add_argument("--seeds", type=int, default=20, help="number of shuffled tie orders")
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
    if args.depth < 1 or args.seeds < 0:
        parser.error(f"need --depth >= 1 and --seeds >= 0, got {args.depth} and {args.seeds}")
    data = load_dataset(args.dataset, archive=args.archive)
    store = _open_store(parser, args, data)
    qrels = {q: data.qrels[q] for q in store.covered_queries if q in data.qrels}
    payload = run_census(store, qrels, cutoffs=args.cutoffs, depth=args.depth, seeds=args.seeds)
    result = ExperimentResult(
        experiment="e6_census",
        parameters={
            "dataset": args.dataset,
            "cutoffs": args.cutoffs,
            "depth": args.depth,
            "seeds": args.seeds,
            "n_queries": len(qrels),
            "store": {label: store.entry(label).npy_sha256 for label in store.labels},
        },
        data_provenance=data.provenance,
        payload=payload,
    )
    destination = args.output / f"e6_census_{args.dataset}.json"
    write_json(destination, result.as_dict())
    for row in [*payload["rows"], *payload["extra_rows"]]:
        print(f"{row['name']:>22}  {row['status']}")
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
