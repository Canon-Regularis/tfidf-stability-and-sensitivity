"""E10: contract spreads, floors per metric and published differences."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from tfidf_stability.evaluation.ann_decomposition import run_ann
from tfidf_stability.evaluation.repeatability_floor import (
    EXPERIMENTS,
    Floor,
    PublishedDifference,
    Spread,
    compare,
    floors,
    published_differences,
    run_floor,
    spreads_from_report,
)
from tfidf_stability.evaluation.score_store import ScoreStore, StoreKind

REPO = Path(__file__).resolve().parents[1]
_URL = "https://example.org/table.html"


def _spread(
    experiment: str = "e6", metric: str = "ndcg@10", spread: Any = 0.01, **source: Any
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "metric": metric,
        "spread": spread,
        "source": source or {"label": "tfidf_n2.naive"},
    }


def _report(
    name: str = "e6_census",
    spreads: list[dict[str, Any]] | None = None,
    dataset: str = "beir_scifact",
    provenance: Any = None,
) -> dict[str, Any]:
    """A report shaped as ``ExperimentResult.as_dict`` writes it."""
    return {
        "experiment": name,
        "result_digest": f"digest-{name}",
        "parameters": {"dataset": dataset},
        "data_provenance": {"archive_sha256": "AAA"} if provenance is None else provenance,
        "payload": {"spreads": [_spread()] if spreads is None else spreads},
        "environment": {},
    }


def _difference(**changes: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "dataset": "beir_scifact",
        "metric": "ndcg@10",
        "system_a": "bm25-flat",
        "value_a": 0.679,
        "system_b": "bm25-multifield",
        "value_b": 0.665,
        "source_url": _URL,
        "retrieved": "2026-11-19",
    }
    entry.update(changes)
    return entry


def _published(*entries: dict[str, Any]) -> tuple[PublishedDifference, ...]:
    return published_differences({"differences": list(entries)})


# --- spreads_from_report ------------------------------------------------------


def test_spreads_read_in_order_with_their_sources() -> None:
    """Each entry becomes a Spread that writes back the same dict."""
    entries = [
        _spread(metric="ndcg@10", spread=0.02, label="bm25-exact_n1.naive"),
        _spread(metric="recall@100", spread=0, rule="id_desc"),
    ]
    spreads = spreads_from_report(_report(spreads=entries))
    assert [s.metric for s in spreads] == ["ndcg@10", "recall@100"]
    assert spreads[1].spread == 0.0
    assert isinstance(spreads[1].spread, float)
    assert [s.as_dict() for s in spreads] == [{**e, "spread": float(e["spread"])} for e in entries]


def test_a_spread_may_name_the_overlap_with_exact_search() -> None:
    """E9's overlap@k is a spread metric of its own."""
    spreads = spreads_from_report(
        _report(name="e9_ann", spreads=[_spread(experiment="e9", metric="overlap@100")])
    )
    assert [s.metric for s in spreads] == ["overlap@100"]


def test_spread_source_is_copied() -> None:
    """Changing the source dict after reading leaves the spread unchanged."""
    source = {"label": "tfidf_n2.pairwise"}
    spread = Spread("e7", "rr@10", 0.5, source)
    source["label"] = "changed"
    assert spread.source == {"label": "tfidf_n2.pairwise"}


def test_spreads_from_an_empty_list() -> None:
    """A report that found no spread yields none."""
    assert spreads_from_report(_report(spreads=[])) == []


@pytest.mark.parametrize(
    ("report", "match"),
    [
        (_report(name="e5_eval"), r"report experiment must be one of"),
        ({"experiment": "e6_census"}, r"e6_census: payload carries no 'spreads' list"),
        ({**_report(), "payload": {"spreads": {}}}, r"payload carries no 'spreads' list"),
        (_report(spreads=[["e6"]]), r"e6_census spread 0 must be a mapping, got list"),
        (
            _report(spreads=[{"experiment": "e6", "metric": "ndcg@10", "spread": 0.1}]),
            r"spread 0: missing keys \['source'\], unknown keys \[\]",
        ),
        (
            _report(spreads=[{**_spread(), "note": 1}]),
            r"missing keys \[\], unknown keys \['note'\]",
        ),
        (_report(spreads=[_spread(), _spread(spread=True)]), r"spread 1: spread must be a number"),
        (_report(spreads=[_spread(spread="0.1")]), r"spread must be a number, got '0\.1'"),
        (_report(spreads=[_spread(spread=math.nan)]), r"spread must be finite, got nan"),
        (_report(spreads=[_spread(spread=math.inf)]), r"spread must be finite, got inf"),
        (_report(spreads=[_spread(spread=-0.5)]), r"spread must be non-negative, got -0\.5"),
        (_report(spreads=[_spread(experiment="e5")]), r"experiment must be one of e6 to e9"),
        (
            _report(spreads=[_spread(metric="map@10")]),
            r"metric must look like ndcg@10, recall@100, rr@10 or overlap@10, got 'map@10'",
        ),
        (_report(spreads=[_spread(metric="ndcg@")]), r"got 'ndcg@'"),
        (_report(spreads=[_spread(metric=10)]), r"metric must look like .*got 10"),
        (
            _report(spreads=[{**_spread(), "source": ["label"]}]),
            r"source must be a dict with string keys",
        ),
        (
            _report(spreads=[{**_spread(), "source": {1: "label"}}]),
            r"source must be a dict with string keys",
        ),
        (
            _report(spreads=[_spread(experiment="e7")]),
            r"e6_census spread 0: experiment 'e7' in a e6_census report",
        ),
    ],
)
def test_spreads_from_report_refuses(report: dict[str, Any], match: str) -> None:
    """A report outside the contract is refused, naming the entry."""
    with pytest.raises(ValueError, match=match):
        spreads_from_report(report)


# --- floors -------------------------------------------------------------------


def test_floor_is_the_largest_spread_per_metric() -> None:
    """Each metric takes its maximum; a later equal spread keeps the first."""
    first = Spread("e6", "ndcg@10", 0.03, {"pair": "first"})
    spreads = [
        Spread("e6", "ndcg@10", 0.01, {"pair": "small"}),
        first,
        Spread("e8", "ndcg@10", 0.03, {"pair": "equal"}),
        Spread("e9", "rr@10", 0.0, {"pair": "zero"}),
    ]
    found = floors(spreads)
    assert found == {
        "ndcg@10": Floor("ndcg@10", 0.03, first),
        "rr@10": Floor("rr@10", 0.0, spreads[3]),
    }


def test_no_spreads_no_floors() -> None:
    """An empty input yields an empty mapping."""
    assert floors([]) == {}


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=12))
def test_floor_equals_the_maximum(values: list[float]) -> None:
    """The floor is the maximum, its source the first spread to reach it."""
    spreads = [Spread("e7", "recall@100", v, {"i": i}) for i, v in enumerate(values)]
    floor = floors(spreads)["recall@100"]
    assert floor.value == max(values)
    assert floor.source.source == {"i": values.index(max(values))}


# --- published_differences ----------------------------------------------------


def test_published_differences_parsed() -> None:
    """Entries come back in order, an integer value coerced to float."""
    parsed = _published(_difference(), _difference(value_a=1, metric="recall@100"))
    assert parsed[0] == PublishedDifference(
        "beir_scifact", "ndcg@10", "bm25-flat", 0.679, "bm25-multifield", 0.665, _URL, "2026-11-19"
    )
    assert parsed[1].value_a == 1.0
    assert isinstance(parsed[1].value_a, float)


def test_published_values_may_lie_on_either_bound() -> None:
    """Both ends of [0, 1] are values a table may publish."""
    parsed = _published(_difference(value_a=0.0, value_b=1.0))
    assert (parsed[0].value_a, parsed[0].value_b) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ([], r"published differences document must be a mapping, got list"),
        ({}, r"missing keys \['differences'\]"),
        ({"differences": [], "notes": 1}, r"unknown keys \['notes'\]"),
        ({"differences": {}}, r"'differences' must be a list"),
        ({"differences": ["entry"]}, r"difference 0 must be a mapping, got str"),
    ],
)
def test_published_document_refused(document: Any, match: str) -> None:
    """A document not shaped as one list of differences is refused."""
    with pytest.raises(ValueError, match=match):
        published_differences(document)


_MISSING = {k: v for k, v in _difference().items() if k != "retrieved"}


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (_MISSING, r"difference 1: missing keys \['retrieved'\], unknown keys \[\]"),
        (_difference(year=2026), r"difference 1: missing keys \[\], unknown keys \['year'\]"),
        (_difference(dataset=""), r"difference 1: dataset must be a non-empty string"),
        (_difference(system_b=7), r"system_b must be a non-empty string, got 7"),
        (_difference(metric="ndcg10"), r"difference 1: metric must look like"),
        (
            _difference(metric="overlap@10"),
            r"metric must look like ndcg@10, recall@100 or rr@10, got 'overlap@10'",
        ),
        (_difference(value_a=True), r"value_a must be a number, got True"),
        (_difference(value_b="0.5"), r"value_b must be a number"),
        (_difference(value_a=math.nan), r"value_a must be finite"),
        (_difference(value_b=1.5), r"value_b must lie in \[0, 1\], got 1\.5"),
        (_difference(value_a=-0.1), r"value_a must lie in \[0, 1\]"),
        (_difference(source_url="http://example.org"), r"source_url must start with https://"),
        (_difference(source_url=None), r"source_url must start with https://, got None"),
        (
            _difference(retrieved=datetime.date(2026, 11, 19)),
            r"retrieved must be a quoted YYYY-MM-DD string",
        ),
        (_difference(retrieved="2026-1-19"), r"retrieved must be a quoted YYYY-MM-DD string"),
        (_difference(retrieved="2026-02-30"), r"retrieved is not a calendar date: '2026-02-30'"),
    ],
)
def test_published_entry_refused(entry: dict[str, Any], match: str) -> None:
    """A bad entry is refused with its index, after a good one."""
    with pytest.raises(ValueError, match=match):
        _published(_difference(), entry)


def test_published_file_is_valid() -> None:
    """The committed table loads and passes validation for both datasets."""
    path = REPO / "configs" / "published_differences.yaml"
    parsed = published_differences(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert {d.dataset for d in parsed} == {"beir_scifact", "beir_nfcorpus"}
    assert {d.metric for d in parsed} == {"ndcg@10", "recall@100"}
    assert {d.source_url for d in parsed} == {"https://castorini.github.io/pyserini/2cr/beir.html"}


# --- compare ------------------------------------------------------------------


def test_compare_sets_each_difference_against_its_floor() -> None:
    """Above, equal to and without a floor; other datasets are left out."""
    floor = Spread("e8", "ndcg@10", 0.25, {"rule": "id_desc"})
    found = floors([floor])
    differences = _published(
        _difference(value_a=0.75, value_b=0.25),
        _difference(value_a=0.5, value_b=0.75),
        _difference(dataset="beir_nfcorpus"),
        _difference(metric="rr@10", value_a=0.5, value_b=0.5),
    )
    rows = compare(found, differences, "beir_scifact")
    assert rows == [
        {
            "systems": ["bm25-flat", "bm25-multifield"],
            "metric": "ndcg@10",
            "values": [0.75, 0.25],
            "difference": 0.5,
            "floor": 0.25,
            "floor_source": floor.as_dict(),
            "exceeds_floor": True,
        },
        {
            "systems": ["bm25-flat", "bm25-multifield"],
            "metric": "ndcg@10",
            "values": [0.5, 0.75],
            "difference": 0.25,
            "floor": 0.25,
            "floor_source": floor.as_dict(),
            "exceeds_floor": False,
        },
        {
            "systems": ["bm25-flat", "bm25-multifield"],
            "metric": "rr@10",
            "values": [0.5, 0.5],
            "difference": 0.0,
            "floor": None,
            "floor_source": None,
            "exceeds_floor": None,
        },
    ]


# --- run_floor ----------------------------------------------------------------


def test_run_floor_payload() -> None:
    """Floors span every read report; absent and None slots are missing."""
    e6 = _report(spreads=[_spread(spread=0.125, label="tfidf_n2.naive")])
    e8 = _report(
        name="e8_hidden_rules",
        spreads=[
            _spread(experiment="e8", spread=0.25, rule="id_desc"),
            _spread(experiment="e8", metric="rr@10", spread=0.5, rule="shuffle"),
        ],
    )
    reports = {"e6_census": e6, "e8_hidden_rules": e8, "e9_ann": None}
    payload = run_floor("beir_scifact", reports, _published(_difference(value_b=0.379)))
    assert payload["dataset"] == "beir_scifact"
    assert payload["data_provenance"] == {"archive_sha256": "AAA"}
    assert payload["inputs"] == {
        "e6_census": {"status": "read", "result_digest": "digest-e6_census"},
        "e7_margin_noise": {"status": "missing", "result_digest": None},
        "e8_hidden_rules": {"status": "read", "result_digest": "digest-e8_hidden_rules"},
        "e9_ann": {"status": "missing", "result_digest": None},
    }
    best = {"experiment": "e8", "metric": "ndcg@10", "spread": 0.25, "source": {"rule": "id_desc"}}
    assert payload["floors"] == {
        "ndcg@10": {"value": 0.25, "source": best},
        "rr@10": {
            "value": 0.5,
            "source": {
                "experiment": "e8",
                "metric": "rr@10",
                "spread": 0.5,
                "source": {"rule": "shuffle"},
            },
        },
    }
    assert len(payload["published"]) == 1
    assert payload["published"][0]["floor_source"] == best
    assert payload["published"][0]["exceeds_floor"] is True


def test_run_floor_with_no_reports() -> None:
    """Every slot missing gives no floor, so no comparison has a verdict."""
    payload = run_floor("beir_scifact", {}, _published(_difference()))
    assert [v["status"] for v in payload["inputs"].values()] == ["missing"] * len(EXPERIMENTS)
    assert payload["data_provenance"] == {}
    assert payload["floors"] == {}
    assert payload["published"][0]["exceeds_floor"] is None


@pytest.mark.parametrize(
    ("reports", "match"),
    [
        ({"e5_eval": _report()}, r"unknown report slots \['e5_eval'\]"),
        ({"e7_margin_noise": _report()}, r"report under e7_margin_noise is for 'e6_census'"),
        (
            {"e6_census": _report(dataset="beir_nfcorpus")},
            r"e6_census report is for dataset 'beir_nfcorpus', not 'beir_scifact'",
        ),
        (
            {"e6_census": {**_report(), "parameters": None}},
            r"e6_census report is for dataset None",
        ),
        (
            {"e6_census": {**_report(), "result_digest": None}},
            r"e6_census result_digest must be a non-empty string",
        ),
        (
            {"e6_census": _report(spreads=[_spread(experiment="e9")])},
            r"experiment 'e9' in a e6_census report",
        ),
        ({"e8_hidden_rules": []}, r"e8_hidden_rules report must be a mapping, got list"),
        (
            {"e6_census": {**_report(), "data_provenance": None}},
            r"e6_census data_provenance must be a mapping, got None",
        ),
        (
            {
                "e6_census": _report(),
                "e9_ann": _report(name="e9_ann", spreads=[], provenance={"archive_sha256": "BBB"}),
            },
            r"e9_ann report's data_provenance differs from e6_census's",
        ),
        (
            {
                "e7_margin_noise": _report(name="e7_margin_noise", spreads=[], provenance={}),
                "e8_hidden_rules": _report(name="e8_hidden_rules", spreads=[]),
            },
            r"e8_hidden_rules report's data_provenance differs from e7_margin_noise's",
        ),
    ],
)
def test_run_floor_refuses(reports: dict[str, Any], match: str) -> None:
    """A misplaced, foreign or malformed report is refused."""
    with pytest.raises(ValueError, match=match):
        run_floor("beir_scifact", reports, ())


def test_an_e9_overlap_spread_keeps_its_own_floor(tmp_path: Path) -> None:
    """A run_ann payload sets overlap@2, not the judged recall@2 floor of E6."""
    ids = ["d1", "d2", "d3", "d4"]
    records = [{"doc_id": d} for d in ids]
    store = ScoreStore.open(tmp_path, dataset="toy", records=records, doc_ids=ids, create=True)
    store = store.add(
        "dense-fp64.exact", {"q": [0.9, 0.8, 0.7, 0.6]}, kind=StoreKind.LEXICAL, params={}
    )
    runs = {"hnsw-a": {"q": {"d1": 0.9, "d2": 0.8}}, "hnsw-b": {"q": {"d1": 0.9, "d4": 0.6}}}
    e9 = _report(name="e9_ann", spreads=[])
    e9["payload"] = run_ann(store, "dense-fp64.exact", runs, cutoffs=[2])
    e6 = _report(spreads=[_spread(metric="recall@2", spread=0.01)])
    published = _published(_difference(metric="recall@2", value_a=0.925, value_b=0.908))

    assert [s.metric for s in spreads_from_report(e9)] == ["overlap@2"]
    payload = run_floor("beir_scifact", {"e6_census": e6, "e9_ann": e9}, published)
    assert {m: (f["value"], f["source"]["experiment"]) for m, f in payload["floors"].items()} == {
        "recall@2": (0.01, "e6"),
        "overlap@2": (0.5, "e9"),
    }
    assert (payload["published"][0]["floor"], payload["published"][0]["exceeds_floor"]) == (
        0.01,
        True,
    )
