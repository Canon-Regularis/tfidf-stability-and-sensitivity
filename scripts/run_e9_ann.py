#!/usr/bin/env python3
"""E9: split each approximate top-k's differences from the exact top-k.

Each ``--ann`` run is read in its own file order and set against ``--reference``, a stored
whole-corpus label. With no run the report states that the HNSW builds are missing.
Rationale: docs/engineering_notes.md#ann-decomposition
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.loaders import LoadedDataset, load_dataset  # noqa: E402
from tfidf_stability.evaluation.ann_decomposition import MISSING_RUNS, run_ann  # noqa: E402
from tfidf_stability.evaluation.external import ExternalDataError, read_run  # noqa: E402
from tfidf_stability.evaluation.score_store import ScoreStore, ScoreStoreError  # noqa: E402
from tfidf_stability.utils.hashing import hash_file  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered BEIR dataset")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=None, help="default data/processed/scores/")
    parser.add_argument("--reference", default=None, help="the stored exact label")
    parser.add_argument("--ann", action="append", default=[], help="LABEL=PATH, a top-N run")
    parser.add_argument("--group", action="append", default=[], help="LABEL=GROUP for a run")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 100])
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    return parser


def _pairs(parser: argparse.ArgumentParser, values: list[str], flag: str) -> dict[str, str]:
    """``LABEL=VALUE`` arguments by label; a malformed or repeated one ends the run."""
    out: dict[str, str] = {}
    for value in values:
        label, _, rest = value.partition("=")
        if not label or not rest or label in out:
            parser.error(f"{flag} expects distinct LABEL=VALUE arguments, got {value!r}")
        out[label] = rest
    return out


def _decompose(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    data: LoadedDataset,
    paths: Mapping[str, str],
    groups: Mapping[str, str] | None,
) -> dict[str, Any]:
    """``run_ann`` over the store; a bad store, label or run file ends the run."""
    directory = args.store or REPO / "data" / "processed" / "scores" / args.dataset
    try:
        store = ScoreStore.open(
            directory, dataset=args.dataset, records=data.records, doc_ids=data.doc_ids
        )
        runs = {label: read_run(path) for label, path in paths.items()}
        return run_ann(store, args.reference, runs, cutoffs=args.cutoffs, groups=groups)
    except (ScoreStoreError, ExternalDataError, ValueError, OSError) as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if min(args.cutoffs) < 1 or len(set(args.cutoffs)) != len(args.cutoffs):
        parser.error(f"--cutoffs must be distinct and at least 1, got {args.cutoffs}")
    paths = _pairs(parser, args.ann, "--ann")
    groups = _pairs(parser, args.group, "--group") or None
    if paths and args.reference is None:
        parser.error("--ann needs --reference, the exact label to compare against")
    data = load_dataset(args.dataset, archive=args.archive)
    payload: dict[str, Any] = {
        "reference": args.reference,
        "cutoffs": args.cutoffs,
        "runs": [],
        "missing": MISSING_RUNS,
        "spreads": [],
    }
    if paths:
        payload = _decompose(parser, args, data, paths, groups)
    result = ExperimentResult(
        experiment="e9_ann",
        parameters={
            "dataset": args.dataset,
            "cutoffs": args.cutoffs,
            "reference": args.reference,
            "runs": {label: hash_file(path) for label, path in paths.items()},
            "groups": groups,
        },
        data_provenance=data.provenance,
        payload=payload,
    )
    destination = args.output / f"e9_ann_{args.dataset}.json"
    write_json(destination, result.as_dict())
    print(f"runs: {len(paths)}{'' if paths else ' (' + MISSING_RUNS + ')'}")
    print(f"written {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
