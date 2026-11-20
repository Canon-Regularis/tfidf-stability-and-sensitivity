#!/usr/bin/env python3
"""E10: the repeatability floor per metric, against published differences.

Reads the E6 to E9 reports of one dataset, takes the largest spread per metric with its
source, and compares each published difference in ``--differences`` with it.
Rationale: docs/engineering_notes.md#repeatability-floor
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.evaluation.repeatability_floor import (  # noqa: E402
    EXPERIMENTS,
    published_differences,
    run_floor,
)
from tfidf_stability.utils.hashing import hash_bytes, hash_file  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="a registered BEIR dataset")
    parser.add_argument("--reports", type=Path, default=REPO / "reports", help="E6 to E9 reports")
    parser.add_argument(
        "--differences", type=Path, default=REPO / "configs" / "published_differences.yaml"
    )
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    return parser


def _floor(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    reports: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """``run_floor`` on the reports read; a malformed report or table ends the run."""
    try:
        document = yaml.safe_load(args.differences.read_text(encoding="utf-8"))
        return run_floor(args.dataset, reports, published_differences(document))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


def _read_reports(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Each report present, parsed and hashed from one read of its bytes."""
    reports: dict[str, Any] = {}
    sources: dict[str, str | None] = {}
    for name in EXPERIMENTS:
        path = args.reports / f"{name}_{args.dataset}.json"
        reports[name] = sources[name] = None
        if path.exists():
            try:
                data = path.read_bytes()
                reports[name] = json.loads(data)
                sources[name] = hash_bytes(data)
            except (OSError, ValueError) as exc:
                parser.error(f"{path}: {exc}")
    return reports, sources


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    reports, sources = _read_reports(parser, args)
    payload = _floor(parser, args, reports)
    result = ExperimentResult(
        experiment="e10_floor",
        parameters={
            "dataset": args.dataset,
            "report_sha256": sources,
            "differences_sha256": hash_file(args.differences),
        },
        data_provenance=payload["data_provenance"],
        payload=payload,
    )
    destination = args.output / f"e10_floor_{args.dataset}.json"
    write_json(destination, result.as_dict())
    for metric, floor in payload["floors"].items():
        print(f"{metric:>12}  floor {floor['value']:.6f}  from {floor['source']['source']}")
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
