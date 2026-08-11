#!/usr/bin/env python3
"""Time the native backend against the normative reference, and prove they agree.

The C++20 core is justified by speed alone -- it is required to be bit-identical,
so it can never be justified by producing a better answer. This runner is what
turns "it is faster" from a claim into a measurement, and it refuses to print a
single ratio until it has checked that the two backends agreed to the last bit.

Without a compiled backend it degrades to reference-only timings, which is the
supported configuration for a contributor with no compiler rather than a
failure. ``--reference-only`` forces that path on a machine that does have one.

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --docs 4000 --queries 40 --repeats 9
    python scripts/benchmark.py --reference-only --json reports/benchmark.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.benchmarks.tfidf_perf import (  # noqa: E402
    DEFAULT_REPEATS,
    BitIdentityError,
    Workload,
    format_report,
    run_benchmarks,
)
from tfidf_stability.utils.io import write_json  # noqa: E402

# The defaults are read off an instance rather than the class: `Workload` uses
# `slots=True`, so the class attributes are slot descriptors, not the defaults.
DEFAULTS = Workload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=DEFAULTS.n_docs)
    parser.add_argument("--vocab", type=int, default=DEFAULTS.vocab_size)
    parser.add_argument("--queries", type=int, default=DEFAULTS.n_queries)
    parser.add_argument("--query-length", type=int, default=DEFAULTS.query_length)
    parser.add_argument("-k", type=int, default=DEFAULTS.k, help="top-k cut for the selection row")
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="timed batches per measurement; the minimum of these is reported",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="ignore the native backend even if it is built",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = parser.parse_args()

    workload = Workload(
        n_docs=args.docs,
        vocab_size=args.vocab,
        n_queries=args.queries,
        query_length=args.query_length,
        k=args.k,
        seed=args.seed,
    )

    try:
        report = run_benchmarks(workload, repeats=args.repeats, use_native=not args.reference_only)
    except BitIdentityError as exc:
        # Deliberately not a traceback: this is a finding about the build, not a
        # crash in the harness, and it must not be mistaken for one.
        print("BIT-IDENTITY FAILURE -- no timings reported", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1

    print(format_report(report))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.json, report.as_dict())
        print(f"\nwritten {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
