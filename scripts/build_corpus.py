#!/usr/bin/env python3
"""Materialise a registered dataset as JSONL, with a provenance sidecar.

The boundary between "where data comes from" and "what the pipeline consumes":
upstream knows about zip archives, CSV quirks and seeded generators, downstream
sees only ``{doc_id, text, ...}`` records.

Three reasons to materialise instead of calling :func:`load_dataset` directly:

1. Parsing MovieLens takes a few seconds; experiments read the corpus many times.
2. It gives a stable file to point ``build-corpus`` and the notebooks at.
3. It fixes the corpus. A regenerated corpus is only as reproducible as its
   generator, which depends on the PRNG, whose selection functions carry no
   stability promise across CPython versions. Writing the file once moves the
   reproducibility boundary from "same interpreter" to "same bytes". Same rule as
   ``datasets/synthetic.py``: the generator writes files, downstream consumes the
   files.

Usage::

    python scripts/build_corpus.py synthetic_small -o data/interim/synth.jsonl
    python scripts/build_corpus.py movielens_small \\
        --archive data/raw/ml-latest-small.zip -o data/interim/movielens.jsonl
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
    parser = argparse.ArgumentParser(description=__doc__)
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
