"""The reproducibility snapshot: one digest over the whole pipeline.

README section 3 promises that "all stages of the pipeline are deterministic
given a fixed corpus, configuration, and software environment". This file is
what holds that promise to account.

The snapshot is a single SHA-256 taken over every intermediate the pipeline
produces -- vocabulary, document frequencies, IDF, weights, norms, scores,
rankings, margins -- computed from the raw binary64 bit patterns rather than any
decimal rendering. **Any** change to any number, anywhere, by a single ulp,
changes it.

That makes the digest deliberately brittle, and the brittleness is the feature.
A change here means a published number has moved, and the commit that moves it
should say why. What the digest must *not* be sensitive to is anything that is
not a number: the working directory, the time, the order documents arrive in, or
the interpreter's hash seed. Those are tested separately below.

The value is not hard-coded. Pinning a literal digest would make this file fail
on any platform whose ``log`` rounds differently -- which, before G13, was every
platform. Instead the digest is recomputed twice under deliberately varied
conditions and required to agree, which tests the property the paper actually
claims.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tfidf_stability.persistence.manifest import RunManifest
from tfidf_stability.persistence.save_load import model_bytes
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.margins import margin_profile
from tfidf_stability.ranking.ranker import rank_all_operators
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.hashing import hash_floats, hash_json, hash_text
from tfidf_stability.utils.io import strip_volatile
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

REPO = Path(__file__).resolve().parents[1]

QUERIES = (
    ("q1", ("quick", "brown", "fox")),
    ("q2", ("numerical", "stability", "sparse")),
    ("q3", ("cosine", "similarity", "vectors")),
)


def pipeline_digest(model: TfidfModel, table: AttributeTable) -> str:
    """One digest over every stage of the pipeline.

    Ordered deliberately: vocabulary first, then the quantities derived from it,
    so a diff in the digest can be localised by recomputing the prefixes.
    """
    parts: list[str] = [
        model.vocabulary.digest(),
        hash_floats(model.idf.values),
        hash_floats(model.matrix.values),
        hash_floats(model.norms),
        hash_text(repr((model.matrix.indptr, model.matrix.indices, model.lengths))),
    ]

    docs = [model.document(i) for i in range(model.n_documents)]
    for _, features in QUERIES:
        query = TfidfVectoriser.transform_query(list(features), model)
        scores = cosine_against_corpus(query, docs, model.norms)
        parts.append(hash_floats(scores))

        rankings = rank_all_operators(scores, table)
        for name in sorted(rankings):
            parts.append(hash_text(repr(rankings[name].order)))
        margins = margin_profile(rankings["pi"].sorted_scores, (1, 2, 3, 5))
        parts.append(hash_floats(m.value for m in margins))

    return hash_text("\n".join(parts))


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------
def test_the_pipeline_digest_is_stable_within_a_process(mini_model, mini_attributes) -> None:  # type: ignore[no-untyped-def]
    assert pipeline_digest(mini_model, mini_attributes) == pipeline_digest(
        mini_model, mini_attributes
    )


def test_the_digest_is_independent_of_document_order(
    mini_corpus, pipeline, mini_attributes
) -> None:  # type: ignore[no-untyped-def]
    """Refitting on a shuffled corpus must give the identical digest.

    The vocabulary is frozen in byte order and every sum runs over ascending
    term identifiers, so presentation order cannot reach any number. This is the
    determinism guarantee of section 3, end to end rather than per module.
    """
    ids = [str(d["doc_id"]) for d in mini_corpus]
    features = [pipeline.preprocess(str(d["text"])) for d in mini_corpus]

    forward = TfidfVectoriser().fit(features, ids)
    order = list(reversed(range(len(ids))))
    shuffled = TfidfVectoriser().fit([features[i] for i in order], [ids[i] for i in order])

    # The document *matrix* is permuted, so compare the order-independent parts.
    assert forward.vocabulary.digest() == shuffled.vocabulary.digest()
    assert hash_floats(forward.idf.values) == hash_floats(shuffled.idf.values)
    by_id = {d: i for i, d in enumerate(shuffled.doc_ids)}
    for i, doc_id in enumerate(forward.doc_ids):
        j = by_id[doc_id]
        assert forward.matrix.row(i).values == shuffled.matrix.row(j).values
        assert hash_floats([forward.norms[i]]) == hash_floats([shuffled.norms[j]])


def test_the_digest_survives_a_save_load_round_trip(mini_model, mini_attributes, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A model read back from disk must reproduce the same pipeline exactly."""
    from tfidf_stability.persistence.save_load import load_model, save_model

    path = tmp_path / "snapshot.tfsx"
    save_model(mini_model, path)
    assert pipeline_digest(load_model(path), mini_attributes) == pipeline_digest(
        mini_model, mini_attributes
    )


def test_the_digest_is_stable_across_processes_and_hash_seeds() -> None:
    """The check nothing in-process can make.

    ``PYTHONHASHSEED`` is fixed at interpreter start-up, so a dictionary
    ordering leaking into any stage would be invisible to a single-process test.
    The working directory is varied at the same time, which catches a path
    reaching a digest.
    """
    snippet = (
        "import json,sys;"
        f"sys.path.insert(0, r'{REPO / 'src'}');"
        f"sys.path.insert(0, r'{REPO / 'tests'}');"
        "from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline;"
        "from tfidf_stability.vectorisation.tfidf import TfidfVectoriser;"
        "from tfidf_stability.ranking.attributes import AttributeTable;"
        "from test_reproducibility_snapshot import pipeline_digest;"
        f"recs=[json.loads(l) for l in open(r'{REPO / 'tests' / 'fixtures' / 'mini_corpus.jsonl'}',"
        "encoding='utf-8') if l.strip()];"
        "p=PreprocessingPipeline();"
        "m=TfidfVectoriser().fit([p.preprocess(r['text']) for r in recs],"
        "[r['doc_id'] for r in recs]);"
        "print(pipeline_digest(m, AttributeTable.from_records(recs)))"
    )

    digests = set()
    for seed, cwd in (("0", REPO), ("1", REPO / "src"), ("random", REPO / "tests")):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            cwd=cwd,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"the digest varied across processes: {digests}"


def test_the_serialised_model_is_byte_stable_across_processes() -> None:
    """The container, not just the numbers."""
    snippet = (
        "import json,sys,hashlib;"
        f"sys.path.insert(0, r'{REPO / 'src'}');"
        "from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline;"
        "from tfidf_stability.vectorisation.tfidf import TfidfVectoriser;"
        "from tfidf_stability.persistence.save_load import model_bytes;"
        f"recs=[json.loads(l) for l in open(r'{REPO / 'tests' / 'fixtures' / 'mini_corpus.jsonl'}',"
        "encoding='utf-8') if l.strip()];"
        "p=PreprocessingPipeline();"
        "m=TfidfVectoriser().fit([p.preprocess(r['text']) for r in recs],"
        "[r['doc_id'] for r in recs]);"
        "print(hashlib.sha256(model_bytes(m)).hexdigest())"
    )
    digests = set()
    for seed in ("0", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------
def test_the_manifest_digest_ignores_volatile_fields() -> None:
    """Two identical runs at different times on different machines must agree,
    or the digest could only record a run rather than verify one."""
    base = {
        "run_kind": "stability_profile",
        "parameters": {"tau": 1e-9, "ks": [5, 10]},
    }
    a = RunManifest(**base)  # type: ignore[arg-type]
    b = RunManifest(**base)  # type: ignore[arg-type]
    a.results = {"digest": "abc", "timestamp": "2026-01-01", "duration_seconds": 12.5}
    b.results = {"digest": "abc", "timestamp": "2031-07-09", "duration_seconds": 0.1}
    assert a.digest() == b.digest()


def test_the_manifest_digest_notices_a_parameter_change() -> None:
    """Everything that could move a number must move the digest."""
    a = RunManifest("stability_profile", parameters={"tau": 1e-9})
    b = RunManifest("stability_profile", parameters={"tau": 1e-8})
    assert a.digest() != b.digest()


def test_notes_do_not_affect_the_digest() -> None:
    a = RunManifest("x", notes="first attempt")
    b = RunManifest("x", notes="second attempt, after lunch")
    assert a.digest() == b.digest()


def test_the_manifest_records_the_build_and_its_reproducibility() -> None:
    manifest = RunManifest("stability_profile")
    env = manifest.environment
    assert "python" in env
    assert "float" in env
    assert manifest.is_reproducible_build is True
    manifest.require_reproducible()


def test_a_non_reproducible_build_is_refused() -> None:
    """A fast-math or arch-tuned build must not silently produce results."""
    import pytest

    manifest = RunManifest("stability_profile")
    manifest.environment = {
        **manifest.environment,
        "native": {"reproducible": False, "fast_math": True, "arch_tune": False},
    }
    assert manifest.is_reproducible_build is False
    with pytest.raises(RuntimeError, match="not reproducible"):
        manifest.require_reproducible()


def test_the_manifest_writes_canonical_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = RunManifest("stability_profile", parameters={"tau": 1e-9, "ks": [5, 10]})
    path = tmp_path / "manifest.json"
    manifest.write(path)
    first = path.read_bytes()
    manifest.write(path)
    assert path.read_bytes() == first, "writing twice must give identical bytes"
    assert b"manifest_digest" in first


def test_strip_volatile_is_recursive() -> None:
    """A timestamp two levels down breaks a snapshot as effectively as one at
    the top, so the stripping recurses."""
    payload = {"a": {"b": {"timestamp": 1, "keep": 2}}, "list": [{"cwd": "/x", "keep": 3}]}
    assert strip_volatile(payload) == {"a": {"b": {"keep": 2}}, "list": [{"keep": 3}]}


def test_canonical_json_is_key_order_independent() -> None:
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})


def test_model_bytes_are_the_same_object_every_call(mini_model) -> None:  # type: ignore[no-untyped-def]
    assert model_bytes(mini_model) == model_bytes(mini_model)
