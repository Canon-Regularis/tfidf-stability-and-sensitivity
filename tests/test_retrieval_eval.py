"""Retrieval evaluation of text queries, and E5 on a fake BEIR archive.

The corpus holds an exact duplicate pair, so one query's top result is a tie that
only the identifier order decides; that is the effect E5 measures.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from tfidf_stability.analysis.query_grid import QueryGrid, evaluate
from tfidf_stability.datasets import beir
from tfidf_stability.evaluation.retrieval import (
    RetrievalReport,
    evaluate_retrieval,
    score_text_queries,
)
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline, preprocess_records
from tfidf_stability.profiles import QueryMode, text_queries
from tfidf_stability.profiles.query_modes import QuerySet
from tfidf_stability.utils.hashing import hash_bytes
from tfidf_stability.utils.numerics import Reduction, same_bits
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

REPO = Path(__file__).resolve().parents[1]

_RECORDS = [
    {"doc_id": "d1", "text": "apple banana"},
    {"doc_id": "d2", "text": "apple banana"},
    {"doc_id": "d3", "text": "apple cherry"},
    {"doc_id": "d4", "text": "durian"},
]
_IDS = [r["doc_id"] for r in _RECORDS]
#: q1 ties d1 with d2 at the top; q2 is all stopwords; q3 names no corpus term.
_QUERIES = {"q1": "apple banana", "q2": "the of", "q3": "zebra"}
_QRELS = {"q1": {"d2": 1, "d4": 0}, "q2": {"d1": 1}, "q3": {"d3": 1}}


def _fit(policy: Reduction = Reduction.NAIVE) -> tuple[TfidfModel, QuerySet]:
    pipeline = PreprocessingPipeline()
    model = TfidfVectoriser(reduction=policy).fit(preprocess_records(pipeline, _RECORDS), _IDS)
    return model, text_queries(_QUERIES, pipeline, n_documents=model.n_documents)


def _grid() -> QueryGrid:
    model, queries = _fit()
    return score_text_queries(queries, model, _RECORDS, _IDS)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(Reduction))
def test_term_at_a_time_scoring_matches_the_grid_evaluator_bit_for_bit(policy: Reduction) -> None:
    model, queries = _fit(policy)

    fast = score_text_queries(queries, model, _RECORDS, _IDS)
    reference = evaluate(queries, model, _RECORDS, _IDS)

    assert fast.n_degenerate == reference.n_degenerate == 1
    assert fast.n_zero_vector == reference.n_zero_vector == 1
    assert fast.provenance() == reference.provenance()
    for got, want in zip(fast.queries, reference.queries, strict=True):
        assert (got.query_id, got.candidate_ids, got.table) == (
            want.query_id,
            want.candidate_ids,
            want.table,
        )
        assert all(same_bits(a, b) for a, b in zip(got.scores, want.scores, strict=True))


@pytest.mark.parametrize("policy", list(Reduction))
def test_replacing_the_norms_of_one_fit_equals_a_refit(policy: Reduction) -> None:
    """The runner fits once and swaps the norms per policy; nothing else may differ."""
    naive, _ = _fit(Reduction.NAIVE)
    refit, _ = _fit(policy)

    swapped = dataclasses.replace(naive, norms=naive.matrix.row_norms(policy), reduction=policy)

    assert swapped.digest() == refit.digest()
    assert swapped.reduction is refit.reduction
    assert all(same_bits(a, b) for a, b in zip(swapped.norms, refit.norms, strict=True))


def test_scoring_refuses_a_set_that_is_not_text() -> None:
    model, queries = _fit()

    with pytest.raises(ValueError, match="expected text queries, got leave_one_out"):
        score_text_queries(
            dataclasses.replace(queries, mode=QueryMode.LEAVE_ONE_OUT), model, _RECORDS, _IDS
        )


def test_scoring_refuses_a_text_query_that_excludes_documents() -> None:
    model, queries = _fit()
    narrowed = dataclasses.replace(queries.queries[0], excluded=frozenset({"d1"}))

    with pytest.raises(ValueError, match="text query 'q1' excludes documents"):
        score_text_queries(dataclasses.replace(queries, queries=(narrowed,)), model, _RECORDS, _IDS)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def test_the_tie_order_alone_decides_the_tied_top_result() -> None:
    report = evaluate_retrieval(_grid(), _QRELS, _IDS, k=1, depth=3, seeds=3)
    q1 = report.queries[0]

    assert q1.query_id == "q1"
    assert q1.metrics["id_asc"]["ndcg@1"] == 0.0, "d1 ranks first and is not relevant"
    assert q1.metrics["id_desc"]["ndcg@1"] == 1.0, "d2 ranks first and is"
    assert q1.margin == 0.0
    assert q1.tie_group == 2
    assert q1.top_k_set_differs
    assert q1.top_k_order_differs
    assert q1.n_nonzero == 3, "d4 shares no term with the query"


def test_a_degenerate_query_ranks_nothing_and_scores_zero() -> None:
    report = evaluate_retrieval(_grid(), _QRELS, _IDS, k=1, depth=3, seeds=0)
    q2 = report.queries[1]

    assert (q2.query_id, q2.degenerate, q2.zero_vector) == ("q2", True, False)
    assert (q2.margin, q2.tie_group, q2.n_nonzero) == (None, 0, 0)
    assert q2.metrics["id_asc"] == {"ndcg@1": 0.0, "recall@3": 0.0, "rr@1": 0.0}
    assert not q2.top_k_set_differs


def test_a_zero_vector_query_is_ordered_by_identifier_alone() -> None:
    report = evaluate_retrieval(_grid(), _QRELS, _IDS, k=1, depth=4, seeds=0)
    q3 = report.queries[2]

    assert (q3.query_id, q3.zero_vector, q3.n_nonzero, q3.tie_group) == ("q3", True, 0, 4)
    assert q3.metrics["id_asc"]["rr@1"] == 0.0, "d1 leads in ascending order"
    assert q3.metrics["id_desc"]["recall@4"] == 1.0, "every document is ranked"


def test_means_count_every_judged_query() -> None:
    report = evaluate_retrieval(_grid(), _QRELS, _IDS, k=1, depth=3, seeds=2)

    asc, desc = report.means["id_asc"]["ndcg@1"], report.means["id_desc"]["ndcg@1"]
    assert (asc.mean, asc.n_queries, asc.n_undefined) == (0.0, 3, 0)
    assert desc.mean == pytest.approx(1 / 3, rel=1e-15)
    assert list(report.means) == ["id_asc", "id_desc", "shuffle_0", "shuffle_1"]


def test_the_report_serialises_counts_means_and_the_shuffle_spread() -> None:
    record = evaluate_retrieval(_grid(), _QRELS, _IDS, k=1, depth=3, seeds=2).as_dict()

    assert (record["k"], record["depth"], record["n_queries"]) == (1, 3, 3)
    assert (record["n_degenerate"], record["n_zero_vector"]) == (1, 1)
    assert record["n_tie_at_cutoff"] == 2, "q1's duplicate pair and q3's all-zero list"
    assert record["n_top_k_set_differs"] == 2
    assert record["means"]["id_desc"]["rr@1"]["n_queries"] == 3
    spread = record["shuffle_spread"]["ndcg@1"]
    assert spread["range"] == spread["max"] - spread["min"] >= 0.0
    assert json.loads(json.dumps(record)) == record, "plain JSON values only"


def test_without_shuffles_there_is_no_spread() -> None:
    assert (
        evaluate_retrieval(_grid(), _QRELS, _IDS, k=1, depth=3, seeds=0).as_dict()["shuffle_spread"]
        == {}
    )


def test_a_grid_with_no_scored_query_is_still_reported() -> None:
    model, queries = _fit()
    only_degenerate = dataclasses.replace(queries, queries=(queries.queries[1],))
    grid = score_text_queries(only_degenerate, model, _RECORDS, _IDS)

    report = evaluate_retrieval(grid, {"q2": {"d1": 1}}, _IDS, k=1, depth=2, seeds=1)

    assert isinstance(report, RetrievalReport)
    assert report.means["shuffle_0"]["ndcg@1"].mean == 0.0


@pytest.mark.parametrize(
    ("k", "depth", "seeds", "message"),
    [
        (0, 3, 0, "need 1 <= k <= depth, got k=0, depth=3"),
        (4, 3, 0, "need 1 <= k <= depth, got k=4, depth=3"),
        (1, 3, -1, "seeds must be non-negative, got -1"),
    ],
)
def test_bad_cutoffs_are_refused(k: int, depth: int, seeds: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(_grid(), _QRELS, _IDS, k=k, depth=depth, seeds=seeds)


def test_a_grid_of_another_mode_is_refused() -> None:
    grid = dataclasses.replace(_grid(), mode=QueryMode.USER_PROFILE)

    with pytest.raises(ValueError, match="expected a grid of text queries, got user_profile"):
        evaluate_retrieval(grid, _QRELS, _IDS, k=1, depth=3, seeds=0)


def test_a_grid_over_another_corpus_order_is_refused() -> None:
    with pytest.raises(ValueError, match="query 'q1' does not rank the whole corpus"):
        evaluate_retrieval(_grid(), _QRELS, list(reversed(_IDS)), k=1, depth=3, seeds=0)


def test_a_scored_query_without_judgements_is_refused() -> None:
    qrels = {q: j for q, j in _QRELS.items() if q != "q3"}

    with pytest.raises(ValueError, match="query 'q3' has no judgements"):
        evaluate_retrieval(_grid(), qrels, _IDS, k=1, depth=3, seeds=0)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
def _runner() -> ModuleType:
    """scripts/run_retrieval_eval.py as a module. Local by house convention."""
    path = REPO / "scripts" / "run_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_retrieval_eval_under_test", path)
    assert spec is not None, "the script must be loadable as a module"
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scifact_zip(qrels: dict[str, dict[str, int]] = _QRELS) -> bytes:
    corpus = "".join(
        json.dumps({"_id": r["doc_id"], "title": "", "text": r["text"]}) + "\n" for r in _RECORDS
    )
    queries = "".join(json.dumps({"_id": q, "text": t}) + "\n" for q, t in _QUERIES.items())
    qrels = "query-id\tcorpus-id\tscore\r\n" + "".join(
        f"{q}\t{d}\t{g}\r\n" for q, judged in qrels.items() for d, g in judged.items()
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scifact/corpus.jsonl", corpus)
        archive.writestr("scifact/queries.jsonl", queries)
        archive.writestr("scifact/qrels/test.tsv", qrels)
    return buffer.getvalue()


def test_the_runner_writes_one_report_per_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _scifact_zip()
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(data)
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(data))
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]

    assert _runner().main([*argv, "--k", "1", "--depth", "3", "--seeds", "2"]) == 0

    record = json.loads(
        (tmp_path / "retrieval_beir_scifact_tfidf_n2.json").read_text(encoding="utf-8")
    )
    assert record["experiment"] == "retrieval_eval"
    assert record["parameters"]["policies"] == [str(p) for p in Reduction]
    assert set(record["parameters"]["model_digests"]) == {str(p) for p in Reduction}
    assert record["data_provenance"]["kind"] == "beir"
    assert record["parameters"]["weighting"] == {"scheme": "tfidf"}
    assert record["parameters"]["n_max"] == 2
    naive = record["payload"]["policies"]["naive"]
    assert naive["means"]["id_desc"]["ndcg@1"]["mean"] == pytest.approx(1 / 3, rel=1e-15)
    for policy in list(Reduction):
        noise = record["payload"]["policies"][str(policy)]["noise_vs_exact"]
        assert noise["reference"] == "exact"
        assert noise["n_queries"] == 2, "the degenerate query is not scored"
    assert record["payload"]["policies"]["exact"]["noise_vs_exact"]["n_scores_differ"] == 0


def test_the_runner_limits_queries_and_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _scifact_zip()
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(data)
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(data))
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]

    policies = ["--policies", "exact", "naive", "exact"]
    assert _runner().main([*argv, "--limit", "1", *policies, "--k", "1"]) == 0

    record = json.loads(
        (tmp_path / "retrieval_beir_scifact_tfidf_n2.json").read_text(encoding="utf-8")
    )
    assert record["parameters"]["n_queries"] == 1
    assert record["parameters"]["policies"] == ["naive", "exact"], "declaration order, once each"
    assert set(record["payload"]["policies"]) == {"naive", "exact"}


def test_the_runner_refuses_a_subset_with_nothing_relevant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _scifact_zip({"q1": {"d1": 0}, "q3": {"d3": 1}})
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(data)
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(data))
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]

    with pytest.raises(SystemExit):
        _runner().main([*argv, "--limit", "1", "--k", "1"])

    assert "no judged query among the first 1 has a relevant document" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--dataset", "synthetic_tiny"], "synthetic_tiny carries no relevance judgements"),
        (["--dataset", "synthetic_tiny", "--k", "5", "--depth", "4"], "need 1 <= --k <= --depth"),
        (["--dataset", "synthetic_tiny", "--seeds", "-1"], "--seeds must be non-negative"),
        (["--dataset", "synthetic_tiny", "--limit", "0"], "--limit must be at least 1, got 0"),
        (["--dataset", "synthetic_tiny", "--n-max", "0"], "--n-max must be at least 1, got 0"),
        (["--dataset", "synthetic_tiny", "--bm25-k1", "x"], "k1 is not a decimal number: 'x'"),
        (["--dataset", "synthetic_tiny", "--bm25-b", "2"], "b must lie in [0, 1], got '2'"),
    ],
)
def test_the_runner_refuses_bad_arguments(
    extra: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _runner().main(extra)

    assert message in capsys.readouterr().err


def test_the_runner_weights_with_bm25_and_names_the_report_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _scifact_zip()
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(data)
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(data))
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]
    bm25 = ["--weighting", "bm25", "--length-norm", "lucene_byte", "--bm25-k1", "1.2"]

    assert _runner().main([*argv, *bm25, "--n-max", "1", "--k", "1", "--depth", "3"]) == 0

    path = tmp_path / "retrieval_beir_scifact_bm25-lucene_byte_n1.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["parameters"]["weighting"] == {
        "scheme": "bm25",
        "k1": "1.2",
        "b": "0.4",
        "length_norm": "lucene_byte",
        "idf": "lucene",
    }
    assert record["parameters"]["n_max"] == 1
    naive = record["payload"]["policies"]["naive"]
    assert naive["means"]["id_desc"]["ndcg@1"]["mean"] == pytest.approx(1 / 3, rel=1e-15), (
        "q1's duplicate pair ties under BM25 too, so only id_desc ranks the relevant d2 first"
    )
    assert "near_tie_census" in naive


def test_the_n_gram_order_reaches_the_preprocessing_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _scifact_zip()
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(data)
    monkeypatch.setitem(beir.PINNED_SHA256, "scifact", hash_bytes(data))
    argv = ["--dataset", "beir_scifact", "--archive", str(archive), "-o", str(tmp_path)]
    fast = ["--k", "1", "--depth", "2", "--seeds", "0", "--policies", "naive"]

    assert _runner().main([*argv, *fast]) == 0
    assert _runner().main([*argv, *fast, "--n-max", "1"]) == 0

    digests = {
        name: json.loads((tmp_path / name).read_text(encoding="utf-8"))["parameters"][
            "preprocessing_digest"
        ]
        for name in ("retrieval_beir_scifact_tfidf_n2.json", "retrieval_beir_scifact_tfidf_n1.json")
    }
    assert len(set(digests.values())) == 2
