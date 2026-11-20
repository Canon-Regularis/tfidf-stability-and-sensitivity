"""E6: the exact and near ties at the cutoff, per scoring condition.

Each row counts the scored queries tied or within each tau at the cutoff, the metric range
over every tie order, and the tie-aware value no tie-break enters.
Rationale: docs/engineering_notes.md#tie-census
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tfidf_stability.evaluation.metrics import MeanMetric
from tfidf_stability.evaluation.retrieval import (
    NEAR_TIE_TAUS,
    QueryOutcome,
    RetrievalReport,
    evaluate_retrieval,
)
from tfidf_stability.evaluation.score_store import ScoreStore, StoreKind
from tfidf_stability.evaluation.tie_aware import expected_means

__all__ = ["E6_ROWS", "CensusRow", "census_row", "run_census"]

_STORE_FIRST = "build the score store first"


@dataclass(frozen=True, slots=True)
class CensusRow:
    """One census row: its display name, its store label and why it may be absent."""

    name: str
    label: str
    missing_reason: str


#: Every row ranks the naive policy, the pinned configuration value.
E6_ROWS: tuple[CensusRow, ...] = (
    CensusRow("TF-IDF", "tfidf_n2.naive", _STORE_FIRST),
    CensusRow("BM25 exact", "bm25-exact_n1.naive", _STORE_FIRST),
    CensusRow("BM25 lucene_byte", "bm25-lucene_byte_n1.naive", _STORE_FIRST),
    CensusRow("SPLADE float", "splade-float.naive", "SPLADE scores arrive with #18"),
    CensusRow("SPLADE x100", "splade-x100.naive", "SPLADE scores arrive with #18"),
    CensusRow("dense fp32", "dense-fp32.run", "dense scores arrive with #16"),
)


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _exact_tie(scored: Sequence[QueryOutcome]) -> dict[str, Any]:
    """Scored queries whose cutoff margin is zero, with and without zero vectors."""
    tied = [q for q in scored if q.margin == 0.0]
    n_non_zero = sum(1 for q in scored if not q.zero_vector)
    n_tied_non_zero = sum(1 for q in tied if not q.zero_vector)
    return {
        "n": len(tied),
        "share": _share(len(tied), len(scored)),
        "share_excluding_zero_vector": _share(n_tied_non_zero, n_non_zero),
    }


def _near_ties(scored: Sequence[QueryOutcome]) -> dict[str, dict[str, float]]:
    """Per tau, the share of scored queries with a contender past the cutoff."""
    out: dict[str, dict[str, float]] = {}
    for tau in (repr(t) for t in NEAR_TIE_TAUS):
        absolute = sum(1 for q in scored if q.contenders[tau] > 0)
        relative = sum(1 for q in scored if q.relative_contenders[tau] > 0)
        out[tau] = {
            "absolute": _share(absolute, len(scored)),
            "relative": _share(relative, len(scored)),
        }
    return out


def _metric_range(report: RetrievalReport) -> dict[str, dict[str, Any]]:
    """Per metric, the lowest and highest mean over every tie-order variant."""
    out: dict[str, dict[str, Any]] = {}
    for name in report.metric_names:
        means = {label: by_metric[name].mean for label, by_metric in report.means.items()}
        low = min(means, key=means.__getitem__)
        high = max(means, key=means.__getitem__)
        out[name] = {
            "min": means[low],
            "max": means[high],
            "range": means[high] - means[low],
            "min_variant": low,
            "max_variant": high,
        }
    return out


def census_row(report: RetrievalReport, neutral: Mapping[str, MeanMetric]) -> dict[str, Any]:
    """One condition's census at one cutoff, JSON-ready.

    Degenerate queries are counted but left out of every share; a share of none is 0.0.
    """
    scored = [q for q in report.queries if not q.degenerate]
    return {
        "n_queries": len(report.queries),
        "n_degenerate": len(report.queries) - len(scored),
        "n_zero_vector": sum(1 for q in scored if q.zero_vector),
        "n_scored": len(scored),
        "exact_tie": _exact_tie(scored),
        "near_ties": _near_ties(scored),
        "metric_range": _metric_range(report),
        "tie_aware": {
            name: {"mean": m.mean, "n_queries": m.n_queries} for name, m in neutral.items()
        },
    }


def _measure(
    store: ScoreStore,
    name: str,
    label: str,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    cutoffs: Sequence[int],
    depth: int,
    seeds: int,
) -> dict[str, Any]:
    """Census one stored label at every cutoff, its grid loaded once."""
    entry = store.entry(label)
    grid = store.grid(label)
    by_cutoff: dict[str, Any] = {}
    for k in cutoffs:
        deep = max(k, depth)
        report = evaluate_retrieval(grid, qrels, store.doc_ids, k=k, depth=deep, seeds=seeds)
        neutral = expected_means(grid, qrels, k=k, depth=deep)
        by_cutoff[str(k)] = census_row(report, neutral)
    return {
        "name": name,
        "label": label,
        "status": "measured",
        "variant": entry.variant,
        "kind": str(entry.kind),
        "by_cutoff": by_cutoff,
    }


def _spreads(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One E6 spread per cutoff and metric of a measured row."""
    return [
        {
            "experiment": "e6",
            "metric": metric,
            "spread": span["range"],
            "source": {
                "row": row["name"],
                "label": row["label"],
                "min_variant": span["min_variant"],
                "max_variant": span["max_variant"],
            },
        }
        for census in row["by_cutoff"].values()
        for metric, span in census["metric_range"].items()
    ]


def _missing(row: CensusRow) -> dict[str, Any]:
    return {"name": row.name, "label": row.label, "status": "missing", "reason": row.missing_reason}


def _is_extra(store: ScoreStore, label: str, named: set[str]) -> bool:
    """A stored label no row names, at the naive policy or from a run or matrix."""
    entry = store.entry(label)
    return label not in named and (
        entry.variant == "naive" or entry.kind in (StoreKind.RUN, StoreKind.MATRIX)
    )


def run_census(
    store: ScoreStore,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    cutoffs: Sequence[int],
    depth: int,
    seeds: int,
    rows: Sequence[CensusRow] = E6_ROWS,
) -> dict[str, Any]:
    """The E6 payload: every row, measured or missing, extra labels and spreads.

    A cutoff past ``depth`` is evaluated to its own depth.
    """

    def measure(name: str, label: str) -> dict[str, Any]:
        return _measure(store, name, label, qrels, cutoffs=cutoffs, depth=depth, seeds=seeds)

    out_rows = [
        measure(row.name, row.label) if row.label in store.entries else _missing(row)
        for row in rows
    ]
    named = {row.label for row in rows}
    extra = [measure(label, label) for label in store.labels if _is_extra(store, label, named)]
    measured = [row for row in out_rows if row["status"] == "measured"] + extra
    return {
        "cutoffs": list(cutoffs),
        "depth": depth,
        "seeds": seeds,
        "rows": out_rows,
        "extra_rows": extra,
        "spreads": [spread for row in measured for spread in _spreads(row)],
    }
