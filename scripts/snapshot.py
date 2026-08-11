#!/usr/bin/env python3
"""Emit the pipeline reproducibility digest.

Prints one SHA-256 covering every number the pipeline produces on a fixed
corpus: vocabulary, IDF, weights, norms, scores, rankings and margins, all
hashed from raw binary64 bit patterns.

This exists so CI can compare the digest **across jobs** -- different operating
systems, compilers and optimisation levels -- which no in-process test can do.
That comparison is the acid test of ``spec_addenda.md#g13``: the platform
logarithm differs from the correctly-rounded value in about 15% of IDF entries,
so before that fix this digest would have differed between Linux and Windows and
the reproducibility claim would have been false.

Usage::

    python scripts/snapshot.py                 # print the digest
    python scripts/snapshot.py --verbose       # print each stage's digest too
    python scripts/snapshot.py --json          # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.persistence.manifest import environment_block  # noqa: E402
from tfidf_stability.persistence.save_load import model_bytes  # noqa: E402
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from tfidf_stability.ranking.attributes import AttributeTable  # noqa: E402
from tfidf_stability.ranking.margins import margin_profile  # noqa: E402
from tfidf_stability.ranking.ranker import rank_all_operators  # noqa: E402
from tfidf_stability.similarity.cosine import cosine_against_corpus  # noqa: E402
from tfidf_stability.utils.hashing import hash_bytes, hash_floats, hash_text  # noqa: E402
from tfidf_stability.utils.io import read_jsonl  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402

CORPUS = REPO / "tests" / "fixtures" / "mini_corpus.jsonl"
QUERIES = (
    ("q1", ("quick", "brown", "fox")),
    ("q2", ("numerical", "stability", "sparse")),
    ("q3", ("cosine", "similarity", "vectors")),
    ("q4", ("the", "of", "and")),  # all stopwords: the zero-vector query
)


def compute() -> dict[str, str]:
    """Digest every stage, keyed by stage name.

    Per-stage rather than one opaque value, so a CI failure says *where* the
    divergence is. A mismatch in ``idf`` alone points at the logarithm; one that
    starts at ``weights`` points at the vectoriser; one confined to ``scores``
    points at the reduction policy.
    """
    records = list(read_jsonl(CORPUS))
    pipeline = PreprocessingPipeline()
    features = [pipeline.preprocess(str(r["text"])) for r in records]
    ids = [str(r["doc_id"]) for r in records]

    model = TfidfVectoriser().fit(features, ids)
    table = AttributeTable.from_records(records)

    stages: dict[str, str] = {
        "preprocessing": pipeline.digest(),
        "vocabulary": model.vocabulary.digest(),
        "idf": hash_floats(model.idf.values),
        "weights": hash_floats(model.matrix.values),
        "norms": hash_floats(model.norms),
        "structure": hash_text(repr((model.matrix.indptr, model.matrix.indices))),
        "container": hash_bytes(model_bytes(model)),
        "attributes": table.digest(),
    }

    docs = [model.document(i) for i in range(model.n_documents)]
    score_parts: list[str] = []
    order_parts: list[str] = []
    margin_parts: list[str] = []
    for _, query_features in QUERIES:
        query = TfidfVectoriser.transform_query(list(query_features), model)
        scores = cosine_against_corpus(query, docs, model.norms)
        score_parts.append(hash_floats(scores))

        rankings = rank_all_operators(scores, table)
        order_parts.extend(hash_text(repr(rankings[n].order)) for n in sorted(rankings))
        margin_parts.append(
            hash_floats(m.value for m in margin_profile(rankings["pi"].sorted_scores, (1, 2, 3, 5)))
        )

    stages["scores"] = hash_text("".join(score_parts))
    stages["rankings"] = hash_text("".join(order_parts))
    stages["margins"] = hash_text("".join(margin_parts))
    stages["overall"] = hash_text("\n".join(f"{k}={v}" for k, v in sorted(stages.items())))
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print every stage")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    stages = compute()
    if args.json:
        print(
            json.dumps(
                {"stages": stages, "environment": environment_block()},
                sort_keys=True,
                indent=2,
                default=str,
            )
        )
    elif args.verbose:
        for name, digest in sorted(stages.items()):
            marker = "  <-- compared across CI jobs" if name == "overall" else ""
            print(f"{name:15} {digest}{marker}")
    else:
        print(stages["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
