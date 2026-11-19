"""scripts/run_external_eval.py on a fake BEIR archive, for each kind of input.

The archive's documents and judgements are fixed; each test writes the external run
or vectors a model would have produced, then checks the report the runner writes.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfidf_stability.datasets import beir
from tfidf_stability.utils.hashing import hash_bytes

REPO = Path(__file__).resolve().parents[1]
_DOCS = ["d1", "d2", "d3", "d4"]
_QRELS = {"q1": {"d2": 1}, "q2": {"d1": 1}}


def _runner() -> ModuleType:
    """scripts/run_external_eval.py as a module. Local by house convention."""
    path = REPO / "scripts" / "run_external_eval.py"
    spec = importlib.util.spec_from_file_location("run_external_eval_under_test", path)
    assert spec is not None, "the script must be loadable as a module"
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    corpus = "".join(json.dumps({"_id": d, "title": "", "text": d}) + "\n" for d in _DOCS)
    queries = "".join(json.dumps({"_id": q, "text": q}) + "\n" for q in ("q1", "q2", "q3"))
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


def _run_file(tmp_path: Path, name: str, scores: dict[str, dict[str, float]]) -> Path:
    path = tmp_path / name
    rows = [
        json.dumps({"query_id": q, "doc_id": d, "score": s})
        for q, by_doc in scores.items()
        for d, s in by_doc.items()
    ]
    path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    return path


def _report(tmp_path: Path, label: str) -> dict[str, Any]:
    path = tmp_path / f"external_beir_scifact_{label}.json"
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


#: d1 and d2 tie for q1, so only the identifier order decides which leads.
_WHOLE = {
    "q1": {"d1": 0.75, "d2": 0.75, "d3": 0.25, "d4": 0.0},
    "q2": {"d1": 0.5, "d2": 0.25, "d3": 0.0, "d4": 0.0},
    "q3": {"d1": 0.1, "d2": 0.2, "d3": 0.3, "d4": 0.4},
}


def test_a_whole_corpus_run_is_compared_with_its_reference(tmp_path: Path, archive: Path) -> None:
    bumped = {**_WHOLE, "q1": {**_WHOLE["q1"], "d2": 0.7500000000000001}}
    run, reference = (
        _run_file(tmp_path, "run.jsonl", bumped),
        _run_file(tmp_path, "ref.jsonl", _WHOLE),
    )
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]

    code = _runner().main(
        [
            *argv,
            "--label",
            "gpu",
            "--run",
            str(run),
            "--reference",
            str(reference),
            "--k",
            "1",
            "--depth",
            "4",
            "--seeds",
            "2",
        ]
    )

    record = _report(tmp_path, "gpu")
    assert code == 0
    assert record["parameters"]["kind"] == "run"
    assert set(record["parameters"]) >= {"run_sha256", "run_digest", "reference_sha256"}
    assert record["payload"]["report"]["n_queries"] == 2, "q3 has no judgements and is dropped"
    noise = record["payload"]["noise_vs_reference"]
    assert (noise["n_scores_differ"], noise["n_top_k_order_differs"]) == (1, 1), (
        "one ulp on d2 breaks q1's tie the other way"
    )
    assert record["payload"]["noise_floor"]["per_policy"][0]["policy"] == "gpu"
    assert record["payload"]["agreement_with_reference"]["mean_recall"] == 0.5


def test_a_partial_run_is_ranked_inside_its_lists(tmp_path: Path, archive: Path) -> None:
    lists = {"q1": {"d2": 0.75, "d3": 0.5}, "q2": {"d1": 0.5}}
    exact = {"q1": {"d2": 0.75, "d1": 0.7}, "q2": {"d1": 0.5}}
    run, reference = (
        _run_file(tmp_path, "ann.jsonl", lists),
        _run_file(tmp_path, "flat.jsonl", exact),
    )
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]

    assert (
        _runner().main(
            [
                *argv,
                "--label",
                "hnsw",
                "--run",
                str(run),
                "--partial",
                "--reference",
                str(reference),
                "--k",
                "2",
                "--depth",
                "2",
                "--seeds",
                "0",
            ]
        )
        == 0
    )

    record = _report(tmp_path, "hnsw")
    assert record["parameters"]["kind"] == "partial_run"
    assert record["payload"]["report"]["partial"] is True
    assert record["payload"]["agreement_with_reference"]["recall"] == {"q1": 0.5, "q2": 1.0}
    assert "noise_vs_reference" not in record["payload"], (
        "lists that differ have no per-score noise"
    )


def test_sparse_vectors_are_scored_under_every_policy(tmp_path: Path, archive: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(
        "".join(
            json.dumps({"id": d, "vector": {d: 1.0, "shared": 0.5}}) + "\n" for d in _DOCS
        ).encode()
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_bytes(
        b'{"id": "q1", "vector": {"d2": 2.0}}\n{"id": "q2", "vector": {"shared": 1.0}}\n'
        b'{"id": "q9", "vector": {"d1": 1.0}}\n'
    )
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]

    assert (
        _runner().main(
            [
                *argv,
                "--label",
                "splade",
                "--sparse-corpus",
                str(corpus),
                "--sparse-queries",
                str(queries),
                "--k",
                "1",
                "--depth",
                "4",
                "--seeds",
                "1",
            ]
        )
        == 0
    )

    record = _report(tmp_path, "splade")
    assert record["parameters"]["kind"] == "sparse"
    assert set(record["payload"]["policies"]) == {"naive", "neumaier", "pairwise", "exact"}
    exact = record["payload"]["policies"]["exact"]
    assert exact["means"]["id_asc"]["ndcg@1"]["mean"] == 1.0, "q1 finds d2; q2 ties, d1 leads"
    assert exact["noise_vs_exact"]["n_scores_differ"] == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--sparse-corpus", "c.jsonl"], "--sparse-corpus and --sparse-queries go together"),
        (
            ["--sparse-corpus", "c.jsonl", "--sparse-queries", "q.jsonl", "--partial"],
            "apply to --run",
        ),
        (["--run", "r.jsonl", "--k", "5", "--depth", "4"], "need 1 <= --k <= --depth"),
        (["--run", "r.jsonl", "--seeds", "-1"], "--seeds must be non-negative"),
    ],
)
def test_bad_arguments_are_refused(
    extra: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _runner().main(["--dataset", "synthetic_tiny", "--label", "x", *extra])

    assert message in capsys.readouterr().err


def test_a_dataset_without_judgements_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _run_file(tmp_path, "run.jsonl", {"q": {"d": 1.0}})

    with pytest.raises(SystemExit):
        _runner().main(["--dataset", "synthetic_tiny", "--label", "x", "--run", str(run)])

    assert "synthetic_tiny carries no relevance judgements" in capsys.readouterr().err


@pytest.mark.parametrize(
    "case",
    [
        ({"q1": {"d1": 1.0}}, None, "query 'q1' scores 1 of 4 documents"),
        ({"q3": _WHOLE["q3"]}, None, "scores no judged query"),
        (_WHOLE, {"q1": _WHOLE["q1"]}, "the run and the reference score different judged queries"),
    ],
)
def test_unreadable_or_mismatched_runs_are_refused(
    tmp_path: Path,
    archive: Path,
    capsys: pytest.CaptureFixture[str],
    case: tuple[dict[str, dict[str, float]], dict[str, dict[str, float]] | None, str],
) -> None:
    run, reference, message = case
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "--label", "x"]
    argv += ["--run", str(_run_file(tmp_path, "run.jsonl", run))]
    if reference is not None:
        argv += ["--reference", str(_run_file(tmp_path, "ref.jsonl", reference))]

    with pytest.raises(SystemExit):
        _runner().main(argv)

    assert message in capsys.readouterr().err


def test_sparse_queries_with_no_judged_query_are_refused(
    tmp_path: Path, archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(
        "".join(json.dumps({"id": d, "vector": {d: 1.0}}) + "\n" for d in _DOCS).encode()
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_bytes(b'{"id": "q9", "vector": {"d1": 1.0}}\n')
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "--label", "x"]

    with pytest.raises(SystemExit):
        _runner().main([*argv, "--sparse-corpus", str(corpus), "--sparse-queries", str(queries)])

    assert "holds no judged query" in capsys.readouterr().err
