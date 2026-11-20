"""The E6 to E10 runners on a fake BEIR archive and a store built from it.

The store is built once per test by scripts/build_score_store.py; each runner then
writes its report, and E10 reads the reports the others wrote.
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
_DOCS = {"d1": "apple banana", "d2": "banana cherry", "d3": "cherry apple", "d4": "durian"}
_QUERIES = {"q1": "banana", "q2": "apple cherry", "q3": "durian apple"}
_QRELS = {"q1": {"d2": 1}, "q2": {"d1": 1}, "q3": {"d4": 2}}


def _script(name: str) -> ModuleType:
    """scripts/<name>.py as a module. Local by house convention."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO / "scripts" / f"{name}.py"
    )
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
        # d3 before d1: the file order differs from the corpus's UTF-8 order.
        zf.writestr("scifact/corpus.jsonl", "".join(sorted(corpus.splitlines(True), reverse=True)))
        zf.writestr("scifact/queries.jsonl", queries)
        zf.writestr("scifact/qrels/test.tsv", qrels)
    path = tmp_path / "scifact.zip"
    path.write_bytes(buffer.getvalue())
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(buffer.getvalue()))
    return path


@pytest.fixture
def common(tmp_path: Path, archive: Path) -> list[str]:
    """Arguments naming the dataset, archive, built store and report folder."""
    store = tmp_path / "store"
    argv = ["--dataset", "beir_scifact", "--archive", str(archive)]
    assert _script("build_score_store").main([*argv, "--out", str(store)]) == 0
    return [*argv, "--store", str(store), "-o", str(tmp_path / "reports"), "--cutoffs", "1", "2"]


def _report(tmp_path: Path, name: str) -> dict[str, Any]:
    path = tmp_path / "reports" / f"{name}_beir_scifact.json"
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_e6_measures_the_lexical_rows_and_names_the_missing_ones(
    tmp_path: Path, common: list[str]
) -> None:
    assert _script("run_e6_census").main([*common, "--depth", "3", "--seeds", "2"]) == 0

    record = _report(tmp_path, "e6_census")
    status = {row["name"]: row["status"] for row in record["payload"]["rows"]}
    assert record["experiment"] == "e6_census"
    assert record["parameters"]["dataset"] == "beir_scifact"
    assert status["TF-IDF"] == "measured"
    assert status["dense fp32"] == "missing"
    assert set(record["payload"]["rows"][0]["by_cutoff"]) == {"1", "2"}
    assert record["payload"]["spreads"], "every measured row adds spreads"


def test_e7_compares_every_policy_with_exact_and_finds_no_falsification(
    tmp_path: Path, common: list[str]
) -> None:
    assert _script("run_e7_margin_noise").main([*common, "--depth", "3"]) == 0

    payload = _report(tmp_path, "e7_margin_noise")["payload"]
    assert payload["n_falsifications"] == 0
    assert len(payload["pairs"]) == 9, "three conditions, three policies against exact each"


def test_e7_exits_1_on_a_falsification(
    tmp_path: Path, common: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _script("run_e7_margin_noise")
    real = runner.run_margin_noise
    monkeypatch.setattr(
        runner, "run_margin_noise", lambda *a, **kw: {**real(*a, **kw), "n_falsifications": 1}
    )

    assert runner.main(common) == 1
    assert _report(tmp_path, "e7_margin_noise")["payload"]["n_falsifications"] == 1


def test_e8_reads_the_insertion_order_from_the_archive_or_a_file(
    tmp_path: Path, common: list[str]
) -> None:
    """The archive lists d4 to d1, so FAISS ids reverse q1's d1-d2 tie."""
    runner = _script("run_e8_hidden_rules")
    assert runner.main(common) == 0
    from_file = _report(tmp_path, "e8_hidden_rules")
    order = tmp_path / "order.txt"
    order.write_bytes(b"d1\nd2\nd3\nd4\n")

    assert runner.main([*common, "--insertion-order", str(order)]) == 0
    from_list = _report(tmp_path, "e8_hidden_rules")
    assert from_file["parameters"]["insertion_source"] == "corpus.jsonl file order"
    assert from_list["parameters"]["insertion_source"].startswith("order.txt sha256 ")

    def faiss_rr(record: dict[str, Any]) -> float:
        rules = record["payload"]["labels"]["tfidf_n2.naive"]["1"]["rules"]
        return rules["faiss_insertion"]["rr@1"]["mean"]  # type: ignore[no-any-return]

    assert faiss_rr(from_file) != faiss_rr(from_list)


def test_e9_without_runs_reports_them_missing(tmp_path: Path, common: list[str]) -> None:
    assert _script("run_e9_ann").main(common) == 0

    payload = _report(tmp_path, "e9_ann")["payload"]
    assert (payload["runs"], payload["missing"]) == ([], "HNSW builds arrive with #16")


def test_e9_decomposes_each_run_against_the_reference(tmp_path: Path, common: list[str]) -> None:
    rows = [
        {"query_id": q, "doc_id": d, "score": s}
        for q in _QRELS
        for d, s in (("d4", 0.5), ("d1", 0.25))
    ]
    run = tmp_path / "ann.jsonl"
    run.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
    argv = [*common, "--reference", "bm25-exact_n1.exact"]

    code = _script("run_e9_ann").main(
        [*argv, "--ann", f"a={run}", "--ann", f"b={run}", "--group", "a=g", "--group", "b=g"]
    )

    payload = _report(tmp_path, "e9_ann")["payload"]
    assert code == 0
    assert [r["label"] for r in payload["runs"]] == ["a", "b"]
    assert payload["missing"] is None


def test_e10_sets_the_floor_from_the_reports_it_finds(tmp_path: Path, common: list[str]) -> None:
    assert _script("run_e6_census").main([*common, "--seeds", "1"]) == 0
    reports = ["--reports", str(tmp_path / "reports"), "-o", str(tmp_path / "reports")]

    assert _script("run_e10_floor").main(["--dataset", "beir_scifact", *reports]) == 0

    record = _report(tmp_path, "e10_floor")
    assert record["payload"]["inputs"]["e6_census"]["status"] == "read"
    assert record["payload"]["inputs"]["e9_ann"]["status"] == "missing"
    assert record["parameters"]["report_sha256"]["e7_margin_noise"] is None
    assert record["data_provenance"]["kind"] == "beir"


def test_e10_reads_every_experiment_end_to_end(tmp_path: Path, common: list[str]) -> None:
    """Real E6 to E9 payloads; each floor names the experiment it comes from."""
    run = tmp_path / "ann.jsonl"
    rows = [{"query_id": q, "doc_id": "d4", "score": 0.5} for q in _QRELS]
    run.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
    ann = ["--reference", "tfidf_n2.exact", "--ann", f"a={run}", "--ann", f"b={run}"]
    assert _script("run_e6_census").main([*common, "--seeds", "1"]) == 0
    assert _script("run_e7_margin_noise").main(common) == 0
    assert _script("run_e8_hidden_rules").main(common) == 0
    assert _script("run_e9_ann").main([*common, *ann]) == 0
    reports = ["--reports", str(tmp_path / "reports"), "-o", str(tmp_path / "reports")]

    assert _script("run_e10_floor").main(["--dataset", "beir_scifact", *reports]) == 0

    payload = _report(tmp_path, "e10_floor")["payload"]
    assert {slot["status"] for slot in payload["inputs"].values()} == {"read"}
    sources = {metric: floor["source"]["experiment"] for metric, floor in payload["floors"].items()}
    assert {sources[m] for m in sources if not m.startswith("overlap@")} <= {"e6", "e7", "e8"}
    assert {sources[m] for m in sources if m.startswith("overlap@")} <= {"e9"}


def test_e10_refuses_a_report_that_is_not_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "e6_census_beir_scifact.json").write_bytes(b"{not json")

    with pytest.raises(SystemExit, match="2"):
        _script("run_e10_floor").main(["--dataset", "beir_scifact", "--reports", str(tmp_path)])

    assert "e6_census_beir_scifact.json" in capsys.readouterr().err


def test_e10_with_no_report_has_no_floor(tmp_path: Path) -> None:
    out = ["--reports", str(tmp_path), "-o", str(tmp_path / "reports")]

    assert _script("run_e10_floor").main(["--dataset", "beir_nfcorpus", *out]) == 0
    record = json.loads((tmp_path / "reports" / "e10_floor_beir_nfcorpus.json").read_bytes())
    assert (record["payload"]["floors"], record["data_provenance"]) == ({}, {})


@pytest.mark.parametrize(
    "case",
    [
        ("run_e6_census", ["--cutoffs", "0"], "--cutoffs must be distinct and at least 1"),
        ("run_e6_census", ["--seeds", "-1"], "need --depth >= 1 and --seeds >= 0"),
        ("run_e7_margin_noise", ["--cutoffs", "2", "2"], "--cutoffs must be distinct"),
        ("run_e7_margin_noise", ["--depth", "0"], "--depth must be at least 1, got 0"),
        ("run_e8_hidden_rules", ["--cutoffs", "-1"], "--cutoffs must be distinct"),
        ("run_e8_hidden_rules", ["--depth", "0"], "--depth must be at least 1, got 0"),
        ("run_e8_hidden_rules", ["--insertion-order", "ORDER"], "does not list every corpus"),
        ("run_e9_ann", ["--cutoffs", "0"], "--cutoffs must be distinct"),
        ("run_e9_ann", ["--ann", "a=x.jsonl"], "--ann needs --reference"),
        ("run_e9_ann", ["--ann", "nolabel"], "--ann expects distinct LABEL=VALUE"),
        (
            "run_e9_ann",
            ["--reference", "z.exact", "--ann", "a=RUN"],
            "the store holds no 'z.exact'",
        ),
    ],
)
def test_bad_arguments_are_refused(
    tmp_path: Path,
    common: list[str],
    capsys: pytest.CaptureFixture[str],
    case: tuple[str, list[str], str],
) -> None:
    order = tmp_path / "order.txt"
    order.write_bytes(b"d1\nd2\n")
    run = tmp_path / "run.jsonl"
    run.write_bytes(json.dumps({"query_id": "q1", "doc_id": "d1", "score": 0.5}).encode() + b"\n")
    script, raw, message = case
    extra = [part.replace("ORDER", str(order)).replace("RUN", str(run)) for part in raw]

    with pytest.raises(SystemExit) as exit_info:
        _script(script).main([*common, *extra])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("script", ["run_e6_census", "run_e7_margin_noise", "run_e8_hidden_rules"])
def test_a_missing_store_is_refused(
    tmp_path: Path, archive: Path, capsys: pytest.CaptureFixture[str], script: str
) -> None:
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "--store", str(tmp_path / "no")]

    with pytest.raises(SystemExit, match="2"):
        _script(script).main(argv)

    assert "build the store first" in capsys.readouterr().err


def test_e10_refuses_a_malformed_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    table = tmp_path / "table.yaml"
    table.write_bytes(b"differences: [{dataset: x}]\n")

    with pytest.raises(SystemExit, match="2"):
        _script("run_e10_floor").main(
            ["--dataset", "beir_scifact", "--reports", str(tmp_path), "--differences", str(table)]
        )

    assert "difference 0" in capsys.readouterr().err
