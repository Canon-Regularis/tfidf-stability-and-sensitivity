#!/usr/bin/env python3
"""Export every intermediate quantity for one document (section 1.2).

Term counts, term frequencies, document frequencies, IDF, tf-idf weights and
the norm, each with its ``float.hex`` bit pattern beside the decimal, because
a decimal rendering can print two values one ulp apart identically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.summarise import ExperimentResult  # noqa: E402
from tfidf_stability.datasets.loaders import load_dataset  # noqa: E402
from tfidf_stability.preprocessing.pipeline import (  # noqa: E402
    PreprocessingPipeline,
    preprocess_records,
)
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="synthetic_tiny")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--doc", default=None, help="document id (default: the first)")
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    args = parser.parse_args()

    data = load_dataset(args.dataset, archive=args.archive)
    pipeline = PreprocessingPipeline()
    features = preprocess_records(pipeline, data.records)
    model = TfidfVectoriser().fit(features, data.doc_ids)

    doc_id = args.doc or data.doc_ids[0]
    if doc_id not in data.doc_ids:
        print(f"no such document: {doc_id}", file=sys.stderr)
        return 1
    index = data.doc_ids.index(doc_id)

    # Read through `TfidfModel.intermediates`, not the CSR row: the row holds only
    # the weight, and `w / idf` does not recover `tf` exactly. The output keys
    # (`term`, `column`) differ from the method's (`token`, `term_id`).
    # Rationale: docs/engineering_notes.md#intermediates-export-source
    detail = model.intermediates(index)
    length = int(detail["in_vocabulary_length"])
    terms = []
    for entry in detail["terms"]:
        term = str(entry["token"])
        idf = float(entry["idf"])
        weight = float(entry["weight"])
        tf = float(entry["tf"])
        terms.append(
            {
                "term": term,
                "column": int(entry["term_id"]),
                # Integer numerator of `tf = count / L`, recovered as `_exact_tf`
                # does rather than by dividing the weight by the IDF.
                "count": round(tf * length) if length else 0,
                "tf": tf,
                "tf_hex": float.hex(tf),
                "df": model.vocabulary.df_of(term),
                "idf": idf,
                "idf_hex": float.hex(idf),
                "weight": weight,
                "weight_hex": float.hex(weight),
            }
        )
    terms.sort(key=lambda t: -t["weight"])

    norm = model.norms[index]
    # The same envelope the other reports carry: a result digest to check a
    # republished number against and a record of the build that produced it.
    result = ExperimentResult(
        experiment="intermediates",
        parameters={
            "dataset": args.dataset,
            "doc_id": doc_id,
            "model_digest": model.digest(),
            "reduction": str(model.reduction),
            "log_impl": str(model.idf.log_impl),
        },
        data_provenance=data.provenance,
        payload={
            "doc_id": doc_id,
            "index": index,
            "n_features": len(features[index]),
            "n_terms_in_vocabulary": len(terms),
            # `L`, the denominator of every `tf` above. Distinct from
            # `n_features`, which counts the document's feature stream before the
            # vocabulary is applied. `L` lets a reader check a count against its `tf`.
            "in_vocabulary_length": length,
            "norm": norm,
            "norm_hex": float.hex(norm),
            "is_zero_norm": norm == 0.0,
            "terms": terms,
        },
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"intermediates_{doc_id}.json"
    write_json(destination, result.as_dict())

    print(f"{doc_id}: {len(terms)} vocabulary terms, norm {norm!r} ({float.hex(norm)})")
    for t in terms[:10]:
        print(f"  {t['term']:<24} df={t['df']:<5} idf={t['idf']:.9f} w={t['weight']:.9f}")
    print(f"\nwritten {destination}")
    print(f"result digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
