#!/usr/bin/env python3
"""E8: metrics under the tie rules of trec_eval, Anserini and FAISS.

Each rule is set against the tie-aware expected metric. FAISS ids follow the order an
index adds documents: corpus.jsonl's own order unless ``--insertion-order`` names one.
Rationale: docs/engineering_notes.md#hidden-tie-rules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.beir import corpus_file_order  # noqa: E402
from tfidf_stability.datasets.loaders import LoadedDataset, load_dataset  # noqa: E402
from tfidf_stability.evaluation.hidden_rules import run_hidden_rules  # noqa: E402
from tfidf_stability.evaluation.score_store import ScoreStore, ScoreStoreError  # noqa: E402
from tfidf_stability.utils.hashing import hash_file  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered BEIR dataset")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=None, help="default data/processed/scores/")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--depth", type=int, default=100, help="ranking depth and recall cutoff")
    parser.add_argument(
        "--insertion-order", type=Path, default=None, help="one document id per line"
    )
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


def _insertion(
    parser: argparse.ArgumentParser, args: argparse.Namespace, doc_ids: list[str]
) -> tuple[list[str], str]:
    """The FAISS insertion order, which must list the corpus once, and its source."""
    if args.insertion_order is None:
        return list(corpus_file_order(args.archive.read_bytes())), "corpus.jsonl file order"
    order = args.insertion_order.read_text(encoding="utf-8").split()
    if sorted(order) != sorted(doc_ids):
        parser.error(f"{args.insertion_order} does not list every corpus document once")
    return order, f"{args.insertion_order.name} sha256 {hash_file(args.insertion_order)}"


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if min(args.cutoffs) < 1 or len(set(args.cutoffs)) != len(args.cutoffs):
        parser.error(f"--cutoffs must be distinct and at least 1, got {args.cutoffs}")
    if args.depth < 1:
        parser.error(f"--depth must be at least 1, got {args.depth}")
    data = load_dataset(args.dataset, archive=args.archive)
    store = _open_store(parser, args, data)
    order, source = _insertion(parser, args, data.doc_ids)
    qrels = {q: data.qrels[q] for q in store.covered_queries if q in data.qrels}
    payload = run_hidden_rules(
        store,
        qrels,
        cutoffs=args.cutoffs,
        depth=args.depth,
        insertion_order=order,
        insertion_source=source,
    )
    result = ExperimentResult(
        experiment="e8_hidden_rules",
        parameters={
            "dataset": args.dataset,
            "cutoffs": args.cutoffs,
            "depth": args.depth,
            "n_queries": len(qrels),
            "insertion_source": source,
            "store": {label: store.entry(label).npy_sha256 for label in store.labels},
        },
        data_provenance=data.provenance,
        payload=payload,
    )
    destination = args.output / f"e8_hidden_rules_{args.dataset}.json"
    write_json(destination, result.as_dict())
    print(f"written {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
