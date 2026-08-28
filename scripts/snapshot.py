#!/usr/bin/env python3
"""Emit the pipeline reproducibility digest.

One SHA-256 over every number the pipeline produces on a fixed corpus:
vocabulary, IDF, weights, norms, scores, rankings and margins, all hashed from
raw binary64 bit patterns.

Exists so CI can compare the digest across jobs (operating systems, compilers,
optimisation levels), which no in-process test can do. That comparison is the
acid test of ``spec_addenda.md#g13``: the platform logarithm differs from the
correctly-rounded value in about 15% of IDF entries, and before that fix this
digest differed between Linux and Windows.

Usage::

    python scripts/snapshot.py                 # print the digest
    python scripts/snapshot.py --verbose       # print each stage's digest too
    python scripts/snapshot.py --json          # machine-readable
    python scripts/snapshot.py --check         # compare against the recorded value

``--check`` answers the question the cross-job comparison cannot. That comparison
proves every platform agrees with every other; it says nothing about whether they
agree with what was published. A change that moves all of them together -- an
edited stopword list alters every df, idf and score identically everywhere --
passes it untouched, and the suite compares runs against each other rather than
against a recorded value, so it passes there too.
``configs/pipeline_digest.txt`` is that recorded value.
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
#: Digest queries as raw text, put through the same preprocessing as the corpus.
#:
#: They were tuples of raw English handed straight to `transform_query`, which
#: takes *features* -- the stemmed, n-grammed form the vocabulary is built from.
#: So "cosine" was looked up in a vocabulary holding "cosin", and missed. Three
#: of the four embedded to the zero vector when only `q4` was meant to:
#:
#:     q1 ("quick", "brown", "fox")            nnz=3, 2 non-zero scores
#:     q2 ("numerical", "stability", "sparse") nnz=0   <- vocabulary has numer, stabil, spars
#:     q3 ("cosine", "similarity", "vectors")  nnz=0   <- cosin, similar, vector
#:     q4 ("the", "of", "and")                 nnz=0   <- intended: all stopwords
#:
#: The `scores`, `rankings` and `margins` stages therefore covered one real
#: query, and `rankings` was mostly ordering an all-zero vector by attributes
#: alone. The digests were stable across platforms, so nothing failed; they were
#: just checking far less than they appeared to.
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
    for _, query_text in QUERIES:
        # Through the same pipeline the corpus went through, so a query term is
        # looked up in the form the vocabulary actually holds.
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
    """The recorded digests, in the two-column form ``--check`` writes."""
    recorded: dict[str, str] = {}
    for line in PINNED.read_text(encoding="utf-8").splitlines():
        statement, _, _comment = line.partition("#")
        fields = statement.split()
        if len(fields) == 2:
            recorded[fields[0]] = fields[1]
    return recorded


def check() -> list[str]:
    """Compare the computed digests against the recorded ones, stage by stage.

    `determinism.yml` proves the digest is the *same everywhere*; nothing proved
    it was the *expected value*. Those are different claims, and only the second
    catches a change that moves every platform together -- editing the stopword
    list, say, which alters every df, idf and score identically on all of them.
    The suite could not catch it either: it compares runs against each other
    rather than against a pinned value.

    Reported per stage rather than on the overall digest alone, because which
    stage moved says what changed. `idf` alone points at the logarithm (G13),
    `preprocessing` at the tokeniser or the word list, `norms` onwards at the
    reduction policy.
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
    # A stage appearing or disappearing matters as much as one changing value:
    # either the pipeline grew a step nobody recorded, or a recorded step is no
    # longer computed and its digest has been standing unchecked.
    problems += [
        f"{name}: recorded but no longer computed" for name in sorted(recorded - stages.keys())
    ]
    problems += [f"{name}: computed but not recorded" for name in sorted(stages.keys() - recorded)]
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
