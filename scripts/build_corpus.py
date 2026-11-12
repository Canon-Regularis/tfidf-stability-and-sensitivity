#!/usr/bin/env python3
"""Materialise a registered dataset as JSONL, with a provenance sidecar.

Downstream reads only ``{doc_id, text, ...}`` records; the written file, not
the generator behind :func:`load_dataset`, is the reproducibility boundary.
Rationale: docs/engineering_notes.md#materialise-corpus-to-file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.datasets.loaders import DATASET_NAMES, load_dataset  # noqa: E402
from tfidf_stability.utils.io import write_json, write_jsonl  # noqa: E402
from tfidf_stability.utils.validation import DataIntegrityError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help=f"one of {', '.join(DATASET_NAMES)}")
    parser.add_argument("-o", "--output", type=Path, required=True, help="destination .jsonl")
    parser.add_argument("--archive", type=Path, default=None, help="MovieLens zip")
    parser.add_argument(
        "--interactions",
        type=Path,
        default=None,
        help="also write interactions here (default: alongside, as <output>.interactions.jsonl)",
    )
    args = parser.parse_args()

    try:
        data = load_dataset(args.dataset, archive=args.archive)
    except DataIntegrityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, data.records)

    interactions = args.interactions or args.output.with_suffix(".interactions.jsonl")
    write_jsonl(
        interactions,
        [{"user_id": u, "doc_id": d, "weight": w} for u, d, w in data.interactions],
    )

    # Provenance sits in a sidecar so the corpus digest covers the documents
    # alone: two corpora with identical content agree even when produced
    # differently.
    sidecar = args.output.with_suffix(".provenance.json")
    write_json(sidecar, {**data.provenance, "corpus_digest": data.digest()})

    print(f"{data.n_documents} documents -> {args.output}")
    print(f"{len(data.interactions)} interactions -> {interactions}")
    print(f"provenance -> {sidecar}")
    print(f"corpus digest {data.digest()}")
    if data.provenance.get("redistributable") is False:
        print("\nNOTE: this corpus is derived from data that may not be redistributed.")
        print("      Do not commit it. See data/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
