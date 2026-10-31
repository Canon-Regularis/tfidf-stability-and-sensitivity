#!/usr/bin/env python3
"""Emit the pipeline reproducibility digest.

One SHA-256 over every number the pipeline produces on a fixed corpus, hashed
from raw binary64 bit patterns, so CI can compare it across operating systems
and compilers. ``--check`` compares it against ``configs/pipeline_digest.txt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.persistence.manifest import environment_block  # noqa: E402
from tfidf_stability.persistence.save_load import model_bytes  # noqa: E402
from tfidf_stability.preprocessing.pipeline import (  # noqa: E402
    PreprocessingPipeline,
    preprocess_records,
)
from tfidf_stability.ranking.attributes import AttributeTable  # noqa: E402
from tfidf_stability.ranking.margins import margin_profile  # noqa: E402
from tfidf_stability.ranking.ranker import rank_all_operators  # noqa: E402
from tfidf_stability.similarity.cosine import cosine_against_corpus  # noqa: E402
from tfidf_stability.utils.hashing import hash_bytes, hash_floats, hash_text  # noqa: E402
from tfidf_stability.utils.io import canonical_json, read_jsonl  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402

CORPUS = REPO / "tests" / "fixtures" / "mini_corpus.jsonl"
#: Digest queries as raw text, put through the same preprocessing as the corpus
#: so each term is looked up in the form the vocabulary holds. Three queries hit
#: the vocabulary; ``q4`` is all stopwords and embeds to the zero vector.
#: Rationale: docs/engineering_notes.md#digest-query-coverage
QUERIES = (
    ("q1", "quick brown fox"),
    ("q2", "numerical stability sparse"),
    ("q3", "cosine similarity vectors"),
    ("q4", "the of and"),  # all stopwords: the zero-vector query, on purpose
)


def compute() -> dict[str, str]:
    """Digest every stage, keyed by stage name.

    Per-stage so a CI failure says where the divergence is: ``idf`` alone points
    at the logarithm, one starting at ``weights`` at the vectoriser, one confined
    to ``scores`` at the reduction policy.
    """
    records = list(read_jsonl(CORPUS))
    pipeline = PreprocessingPipeline()
    features = preprocess_records(pipeline, records)
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
    for _, query_text in QUERIES:
        # Preprocess the query with the corpus pipeline so each term is looked up
        # in the form the vocabulary holds.
        query = TfidfVectoriser.transform_query(pipeline.preprocess(query_text), model)
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


#: The digests this repository is expected to produce, one stage per line.
#: See `--check`.
PINNED = Path(__file__).resolve().parents[1] / "configs" / "pipeline_digest.txt"


def _pinned() -> dict[str, str]:
    """Read the recorded digests from ``PINNED`` as a stage-to-digest mapping."""
    recorded: dict[str, str] = {}
    for line in PINNED.read_text(encoding="utf-8").splitlines():
        statement, _, _comment = line.partition("#")
        fields = statement.split()
        if len(fields) == 2:
            recorded[fields[0]] = fields[1]
    return recorded


def check() -> list[str]:
    """Compare the computed digests against the recorded ones, stage by stage.

    ``determinism.yml`` proves the digest is the same on every platform; this
    check proves it is the recorded value, and the stage that moved says what
    changed. Rationale: docs/engineering_notes.md#pipeline-digest-check
    """
    recorded = _pinned()
    if not recorded:
        return [f"{PINNED} records no digests; the check would pass vacuously"]

    stages = compute()
    problems = [
        f"{name}: expected {recorded[name]}, computed {stages[name]}"
        for name in sorted(recorded)
        if name in stages and stages[name] != recorded[name]
    ]
    # A stage present on one side only is also a problem: an unrecorded stage
    # is unchecked, and a recorded stage that is not computed checks nothing.
    problems += [
        f"{name}: recorded but no longer computed" for name in sorted(recorded - stages.keys())
    ]
    problems += [f"{name}: computed but not recorded" for name in sorted(stages.keys() - recorded)]
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true", help="print every stage")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--check",
        action="store_true",
        help=f"compare against {PINNED.name} and exit non-zero on any difference",
    )
    args = parser.parse_args()

    if args.check:
        problems = check()
        if problems:
            print("::error::the pipeline digest has moved", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print(
                "\nIf this change was intended, update configs/pipeline_digest.txt "
                "in the same commit that caused it -- that file is the record of "
                "which code produced the published numbers.",
                file=sys.stderr,
            )
            return 1
        print(f"every stage matches {PINNED.name}")
        return 0

    stages = compute()
    if args.json:
        # `canonical_json` rather than a local `json.dumps`: one canonical
        # form for every file this project writes, and it carries the trailing
        # newline, so `print` is told not to add a second.
        print(
            canonical_json({"stages": stages, "environment": environment_block()}),
            end="",
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
