"""scripts/build_score_store.py on a fake BEIR archive, for each kind of input."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tfidf_stability.datasets import beir
from tfidf_stability.datasets.loaders import load_dataset
from tfidf_stability.evaluation.score_store import ScoreStore, StoreKind
from tfidf_stability.utils.hashing import hash_bytes

REPO = Path(__file__).resolve().parents[1]
_DOCS = {"d1": "apple banana", "d2": "banana cherry", "d3": "cherry apple", "d4": "durian"}
#: q3's only word is a stopword, so its query is degenerate.
_QUERIES = {"q1": "banana", "q2": "apple cherry", "q3": "the"}
_QRELS = {"q1": {"d2": 1}, "q2": {"d1": 1}, "q3": {"d4": 1}}


def _builder() -> ModuleType:
    """scripts/build_score_store.py as a module. Local by house convention."""
    path = REPO / "scripts" / "build_score_store.py"
    spec = importlib.util.spec_from_file_location("build_score_store_under_test", path)
    assert spec is not None, "the script must be loadable as a module"
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    corpus = "".join(
        json.dumps({"_id": d, "title": "", "text": t}) + "\n" for d, t in _DOCS.items()
    )
    queries = "".join(json.dumps({"_id": q, "text": t}) + "\n" for q, t in _QUERIES.items())
    qrels = "query-id\tcorpus-id\tscore\n" + "".join(
        f"{q}\t{d}\t{g}\n" for q, judged in _QRELS.items() for d, g in judged.items()
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("scifact/corpus.jsonl", corpus)
        zf.writestr("scifact/queries.jsonl", queries)
        zf.writestr("scifact/qrels/test.tsv", qrels)
    path = tmp_path / "scifact.zip"
    path.write_bytes(buffer.getvalue())
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(buffer.getvalue()))
    return path


def _open(archive: Path, out: Path) -> ScoreStore:
    data = load_dataset("beir_scifact", archive=archive)
    return ScoreStore.open(out, dataset="beir_scifact", records=data.records, doc_ids=data.doc_ids)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
    return path


def test_every_lexical_condition_is_stored_under_every_policy(
    tmp_path: Path, archive: Path
) -> None:
    out = tmp_path / "store"

    code = _builder().main(
        ["--dataset", "beir_scifact", "--archive", str(archive), "--out", str(out)]
    )

    store = _open(archive, out)
    assert code == 0
    assert len(store.labels) == 12
    naive = store.entry("bm25-lucene_byte_n1.naive")
    assert naive.reference_label == "bm25-lucene_byte_n1.exact"
    assert store.entry("tfidf_n2.exact").reference_label is None
    assert (naive.kind, naive.params["length_norm"], naive.params["n_max"]) == (
        StoreKind.LEXICAL,
        "lucene_byte",
        1,
    )
    assert store.entry("tfidf_n2.naive").params["length_norm"] is None
    assert (naive.query_ids, naive.degenerate_query_ids) == (("q1", "q2"), ("q3",))
    assert store.grid("tfidf_n2.exact").n_degenerate == 1
    digests = {label: store.entry(label).params["model_digest"] for label in store.labels}
    assert len(set(digests.values())) == 3, "the length rule reaches the model; policy does not"
    assert {store.entry(label).params["policy"] for label in store.labels} == {
        "naive",
        "neumaier",
        "pairwise",
        "exact",
    }


def test_each_policy_reaches_the_scorer_with_its_own_norms(
    tmp_path: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _builder()
    seen: list[tuple[str, str, bool]] = []
    real = builder.score_text_queries

    def spy(queries: object, model: object, records: object, doc_ids: object) -> object:
        norms = getattr(model, "norms", ())
        own = norms == tuple(model.matrix.row_norms(model.reduction))  # type: ignore[attr-defined]
        kind = str(getattr(model, "length_norm", "tfidf"))
        seen.append((kind, str(model.reduction), own))  # type: ignore[attr-defined]
        return real(queries, model, records, doc_ids)

    monkeypatch.setattr(builder, "score_text_queries", spy)
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "--out", str(tmp_path / "s")]

    assert builder.main(argv) == 0
    policies = ["exact", "naive", "neumaier", "pairwise"]
    assert sorted(seen) == sorted(
        (kind, policy, kind == "tfidf")
        for kind in ("tfidf", "exact", "lucene_byte")
        for policy in policies
    ), "TF-IDF norms follow the policy; BM25 norms are all 1.0"


def test_a_limit_keeps_the_first_judged_queries(tmp_path: Path, archive: Path) -> None:
    out = tmp_path / "store"
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "--out", str(out)]

    assert _builder().main([*argv, "--limit", "1"]) == 0
    assert _open(archive, out).entry("tfidf_n2.naive").query_ids == ("q1",)


def test_runs_matrices_and_vectors_join_the_store(tmp_path: Path, archive: Path) -> None:
    whole = {q: {d: 0.25 * i for i, d in enumerate(_DOCS)} for q in ("q1", "q2", "q9")}
    run = _jsonl(
        tmp_path / "run.jsonl",
        [
            {"query_id": q, "doc_id": d, "score": s}
            for q, row in whole.items()
            for d, s in row.items()
        ],
    )
    npy = tmp_path / "scores.npy"
    np.save(npy, np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype="<f4"))
    (tmp_path / "q.txt").write_bytes(b"q2\nq1\n")
    (tmp_path / "d.txt").write_bytes(b"d4\nd3\nd2\nd1\n")
    corpus = _jsonl(
        tmp_path / "c.jsonl",
        [{"id": d, "vector": dict.fromkeys(t.split(), 1.0)} for d, t in _DOCS.items()],
    )
    queries = _jsonl(tmp_path / "q.jsonl", [{"id": "q1", "vector": {"banana": 0.5}}])
    out = tmp_path / "store"

    code = _builder().main(
        [
            *("--dataset", "beir_scifact", "--archive", str(archive), "--out", str(out)),
            "--no-lexical",
            *("--run", f"b.run={run}", "--run", f"a.run={run}", "--reference", "b.run=a.run"),
            *("--reference", "a.run=dense-fp32.run"),
            *("--matrix", f"dense-fp32.run={npy},{tmp_path / 'q.txt'},{tmp_path / 'd.txt'}"),
            *("--sparse", f"splade-float={corpus},{queries}"),
        ]
    )

    store = _open(archive, out)
    assert code == 0
    assert store.labels == (
        "a.run",
        "b.run",
        "dense-fp32.run",
        "splade-float.exact",
        "splade-float.naive",
        "splade-float.neumaier",
        "splade-float.pairwise",
    )
    assert store.entry("b.run").reference_label == "a.run", "b.run is stored after a.run"
    assert store.entry("a.run").reference_label == "dense-fp32.run", "a run may name a matrix"
    assert store.entry("a.run").query_ids == ("q1", "q2"), "q9 is not judged"
    assert store.entry("a.run").degenerate_query_ids == ("q3",)
    assert store.scores("dense-fp32.run")["q2"] == (4.0, 3.0, 2.0, 1.0)
    assert store.entry("dense-fp32.run").kind is StoreKind.MATRIX
    sparse = store.entry("splade-float.pairwise")
    assert (sparse.kind, sparse.reference_label) == (StoreKind.SPARSE, "splade-float.exact")
    assert sparse.degenerate_query_ids == ("q2", "q3")
    assert store.scores("splade-float.naive")["q1"] == (0.5, 0.5, 0.0, 0.0)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--limit", "0"], "--limit must be at least 1, got 0"),
        (["--run", "nolabel"], "--run expects LABEL=PATH, got 'nolabel'"),
        (["--sparse", "a=onlyone"], r"--sparse expects LABEL=PATH,PATH, got 'a=onlyone'"),
        (["--matrix", "a.run=x,,y"], r"--matrix expects LABEL=PATH,PATH,PATH"),
        (["--reference", "=a.run"], "--reference expects LABEL=PATH"),
        (["--no-lexical", "--run", "a.run=missing.jsonl"], "missing.jsonl"),
        (
            ["--no-lexical", "--run", "a.run={run}", "--reference", "a.run=z.run"],
            r"reference 'z\.run' is not another stored label",
        ),
        (
            ["--run", "a.run={run}", "--reference", "b.run=a.run"],
            r"--reference names 'b\.run', which no --run or --matrix stores",
        ),
        (
            [
                *("--run", "a.run={run}", "--run", "b.run={run}"),
                *("--reference", "a.run=b.run", "--reference", "b.run=a.run"),
            ],
            r"--reference labels form a cycle through 'a\.run'",
        ),
        (["--run", "a.run={run}", "--matrix", "a.run=x,y,z"], r"'a\.run' is given twice"),
    ],
)
def test_bad_arguments_and_inputs_are_refused(
    tmp_path: Path,
    archive: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    run = _jsonl(
        tmp_path / "run.jsonl",
        [{"query_id": "q1", "doc_id": d, "score": 0.5} for d in _DOCS],
    )
    extra = [part.format(run=run) for part in extra]
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "--out", str(tmp_path / "s")]

    with pytest.raises(SystemExit) as exit_info:
        _builder().main([*argv, *extra])

    assert exit_info.value.code == 2
    assert re.search(message, capsys.readouterr().err), message


def test_a_dataset_without_judgements_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _builder()
    real = builder.load_dataset("synthetic_tiny")
    monkeypatch.setattr(builder, "load_dataset", lambda name, archive: real)

    with pytest.raises(SystemExit, match="2"):
        builder.main(["--dataset", "synthetic_tiny", "--archive", str(tmp_path / "x.zip")])

    assert "synthetic_tiny carries no relevance judgements" in capsys.readouterr().err
