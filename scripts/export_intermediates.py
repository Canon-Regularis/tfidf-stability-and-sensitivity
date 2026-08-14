#!/usr/bin/env python3
"""Export every intermediate quantity for one document (section 1.2).

Term counts, term frequencies, document frequencies, IDF, the tf-idf weights and
the norm, for a single document, each with its raw bit pattern beside the decimal.

The hex is what makes this evidence. A decimal rendering of a binary64 is a lossy
summary at whatever precision the formatter chose, so two values one ulp apart,
the difference this study is about, can print identically. ``float.hex``
round-trips.

Usage::

    python scripts/export_intermediates.py --dataset synthetic_tiny --doc d000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.datasets.loaders import load_dataset  # noqa: E402
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic_tiny")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--doc", default=None, help="document id (default: the first)")
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    args = parser.parse_args()

    data = load_dataset(args.dataset, archive=args.archive)
    pipeline = PreprocessingPipeline()
    features = [pipeline.preprocess(str(r["text"])) for r in data.records]
    model = TfidfVectoriser().fit(features, data.doc_ids)

    doc_id = args.doc or data.doc_ids[0]
    if doc_id not in data.doc_ids:
        print(f"no such document: {doc_id}", file=sys.stderr)
        return 1
    index = data.doc_ids.index(doc_id)

    vector = model.document(index)
    terms = []
    for column, weight in zip(vector.indices, vector.values, strict=True):
        term = model.vocabulary.token_of(column)
        idf = model.idf.values[column]
        terms.append(
            {
                "term": term,
                "column": column,
                "df": model.vocabulary.df_of(term),
                "idf": idf,
                "idf_hex": float.hex(idf),
                "weight": weight,
                "weight_hex": float.hex(weight),
            }
        )
    terms.sort(key=lambda t: -t["weight"])

    norm = model.norms[index]
    payload = {
        "doc_id": doc_id,
        "index": index,
        "n_features": len(features[index]),
        "n_terms_in_vocabulary": len(terms),
        "norm": norm,
        "norm_hex": float.hex(norm),
        "is_zero_norm": norm == 0.0,
        "model_digest": model.digest(),
        "reduction": str(model.reduction),
        "log_impl": str(model.idf.log_impl),
        "terms": terms,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"intermediates_{doc_id}.json"
    write_json(destination, payload)

    print(f"{doc_id}: {len(terms)} vocabulary terms, norm {norm!r} ({float.hex(norm)})")
    for t in terms[:10]:
        print(f"  {t['term']:<24} df={t['df']:<5} idf={t['idf']:.9f} w={t['weight']:.9f}")
    print(f"\nwritten {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
