"""E10: the repeatability floor beside published system differences.

The floor of a metric is the largest spread E6 to E9 report for it on one dataset, named
by its run; a published difference below it is within that noise. No table lists overlap@k.
Rationale: docs/engineering_notes.md#repeatability-floor
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = [
    "EXPERIMENTS",
    "Floor",
    "PublishedDifference",
    "Spread",
    "compare",
    "floors",
    "published_differences",
    "run_floor",
    "spreads_from_report",
]

#: Report experiment names, each with the short name its spreads carry.
EXPERIMENTS: dict[str, str] = {
    "e6_census": "e6",
    "e7_margin_noise": "e7",
    "e8_hidden_rules": "e8",
    "e9_ann": "e9",
}

#: Judged metrics, which published tables report, and E9's overlap with exact search.
_JUDGED = re.compile(r"(ndcg|recall|rr)@[0-9]+")
_SPREAD_METRIC = re.compile(r"(ndcg|recall|rr|overlap)@[0-9]+")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_SPREAD_KEYS = frozenset({"experiment", "metric", "spread", "source"})
_DIFFERENCE_KEYS = (
    "dataset",
    "metric",
    "system_a",
    "value_a",
    "system_b",
    "value_b",
    "source_url",
    "retrieved",
)


def _number(value: Any, what: str) -> float:
    """``value`` as a finite float; a bool or a non-number is refused."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return float(value)


def _text(value: Any, what: str) -> str:
    """``value`` as a non-empty string, or a ``ValueError``."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _check_metric(metric: Any) -> str:
    """A judged metric name in E5's form, such as ``ndcg@10``."""
    if not isinstance(metric, str) or _JUDGED.fullmatch(metric) is None:
        raise ValueError(f"metric must look like ndcg@10, recall@100 or rr@10, got {metric!r}")
    return metric


def _check_spread_metric(metric: Any) -> str:
    """A judged metric name, or E9's ``overlap@k`` with the exact top ``k``."""
    if not isinstance(metric, str) or _SPREAD_METRIC.fullmatch(metric) is None:
        raise ValueError(
            f"metric must look like ndcg@10, recall@100, rr@10 or overlap@10, got {metric!r}"
        )
    return metric


def _check_keys(entry: Any, expected: Iterable[str], what: str) -> Mapping[str, Any]:
    """``entry`` as a dict with exactly the ``expected`` keys."""
    if not isinstance(entry, dict):
        raise ValueError(f"{what} must be a mapping, got {type(entry).__name__}")
    wanted = set(expected)
    if set(entry) != wanted:
        missing = sorted(wanted - set(entry))
        unknown = sorted(map(str, set(entry) - wanted))
        raise ValueError(f"{what}: missing keys {missing}, unknown keys {unknown}")
    return entry


@dataclass(frozen=True, slots=True)
class Spread:
    """One spread an experiment reports, with the run that gives it."""

    experiment: str
    metric: str
    spread: float
    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.experiment not in EXPERIMENTS.values():
            raise ValueError(f"experiment must be one of e6 to e9, got {self.experiment!r}")
        _check_spread_metric(self.metric)
        spread = _number(self.spread, "spread")
        if spread < 0.0:
            raise ValueError(f"spread must be non-negative, got {spread!r}")
        if not isinstance(self.source, dict) or not all(isinstance(k, str) for k in self.source):
            raise ValueError(f"source must be a dict with string keys, got {self.source!r}")
        object.__setattr__(self, "spread", spread)
        object.__setattr__(self, "source", dict(self.source))

    def as_dict(self) -> dict[str, Any]:
        """The spread under the report contract's keys."""
        return {
            "experiment": self.experiment,
            "metric": self.metric,
            "spread": self.spread,
            "source": dict(self.source),
        }


def spreads_from_report(report: Mapping[str, Any]) -> list[Spread]:
    """The spreads of one E6 to E9 report, each checked against the contract.

    Raises ``ValueError`` for an unknown experiment, a payload without a ``spreads``
    list, or an entry with other keys, values or experiment than the report's.
    """
    name = report.get("experiment")
    if name not in EXPERIMENTS:
        raise ValueError(f"report experiment must be one of {sorted(EXPERIMENTS)}, got {name!r}")
    payload = report.get("payload")
    entries = payload.get("spreads") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{name}: payload carries no 'spreads' list")
    spreads: list[Spread] = []
    for index, entry in enumerate(entries):
        what = f"{name} spread {index}"
        fields = _check_keys(entry, _SPREAD_KEYS, what)
        try:
            spread = Spread(**fields)
        except ValueError as error:
            raise ValueError(f"{what}: {error}") from None
        if spread.experiment != EXPERIMENTS[name]:
            raise ValueError(f"{what}: experiment {spread.experiment!r} in a {name} report")
        spreads.append(spread)
    return spreads


@dataclass(frozen=True, slots=True)
class Floor:
    """The largest spread of one metric and the spread that gives it."""

    metric: str
    value: float
    source: Spread


def floors(spreads: Iterable[Spread]) -> dict[str, Floor]:
    """The largest spread per metric; the first in input order wins a tie."""
    found: dict[str, Floor] = {}
    for spread in spreads:
        held = found.get(spread.metric)
        if held is None or spread.spread > held.value:
            found[spread.metric] = Floor(spread.metric, spread.spread, spread)
    return found


@dataclass(frozen=True, slots=True)
class PublishedDifference:
    """Two systems' published values of one metric on one dataset."""

    dataset: str
    metric: str
    system_a: str
    value_a: float
    system_b: str
    value_b: float
    source_url: str
    retrieved: str

    def __post_init__(self) -> None:
        for name in ("dataset", "system_a", "system_b"):
            _text(getattr(self, name), name)
        _check_metric(self.metric)
        for name in ("value_a", "value_b"):
            value = _number(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
            object.__setattr__(self, name, value)
        if not isinstance(self.source_url, str) or not self.source_url.startswith("https://"):
            raise ValueError(f"source_url must start with https://, got {self.source_url!r}")
        _check_date(self.retrieved)


def _check_date(retrieved: Any) -> None:
    """A calendar date written as a ``YYYY-MM-DD`` string."""
    if not isinstance(retrieved, str) or _ISO_DATE.fullmatch(retrieved) is None:
        raise ValueError(f"retrieved must be a quoted YYYY-MM-DD string, got {retrieved!r}")
    try:
        date.fromisoformat(retrieved)
    except ValueError:
        raise ValueError(f"retrieved is not a calendar date: {retrieved!r}") from None


def published_differences(document: Any) -> tuple[PublishedDifference, ...]:
    """The entries of a ``{"differences": [...]}`` document, each checked.

    Raises ``ValueError`` naming the entry index for a missing or unknown key or a bad value.
    """
    _check_keys(document, ("differences",), "published differences document")
    entries = document["differences"]
    if not isinstance(entries, list):
        raise ValueError("'differences' must be a list")
    parsed: list[PublishedDifference] = []
    for index, entry in enumerate(entries):
        fields = _check_keys(entry, _DIFFERENCE_KEYS, f"difference {index}")
        try:
            parsed.append(PublishedDifference(**fields))
        except ValueError as error:
            raise ValueError(f"difference {index}: {error}") from None
    return tuple(parsed)


def compare(
    floors: Mapping[str, Floor],
    differences: Iterable[PublishedDifference],
    dataset: str,
) -> list[dict[str, Any]]:
    """Each published difference on ``dataset`` beside its metric's floor.

    ``exceeds_floor`` is ``None`` when no experiment reported a spread for the metric.
    """
    rows: list[dict[str, Any]] = []
    for item in differences:
        if item.dataset != dataset:
            continue
        floor = floors.get(item.metric)
        difference = abs(item.value_a - item.value_b)
        rows.append(
            {
                "systems": [item.system_a, item.system_b],
                "metric": item.metric,
                "values": [item.value_a, item.value_b],
                "difference": difference,
                "floor": None if floor is None else floor.value,
                "floor_source": None if floor is None else floor.source.as_dict(),
                "exceeds_floor": None if floor is None else difference > floor.value,
            }
        )
    return rows


def _checked_report(slot: str, report: Any, dataset: str) -> tuple[str, dict[str, Any]]:
    """The report's digest and provenance, once it fits its slot and dataset."""
    if not isinstance(report, dict):
        raise ValueError(f"{slot} report must be a mapping, got {type(report).__name__}")
    if report.get("experiment") != slot:
        raise ValueError(f"report under {slot} is for {report.get('experiment')!r}")
    parameters = report.get("parameters")
    found = parameters.get("dataset") if isinstance(parameters, dict) else None
    if found != dataset:
        raise ValueError(f"{slot} report is for dataset {found!r}, not {dataset!r}")
    provenance = report.get("data_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{slot} data_provenance must be a mapping, got {provenance!r}")
    return _text(report.get("result_digest"), f"{slot} result_digest"), provenance


def run_floor(
    dataset: str,
    reports: Mapping[str, Mapping[str, Any] | None],
    differences: Sequence[PublishedDifference],
) -> dict[str, Any]:
    """E10's payload: reports read, their shared provenance, floors and comparison.

    A slot absent from ``reports`` or holding ``None`` is missing. Raises ``ValueError``
    for an unknown slot, a non-dict report or provenance, an experiment or dataset other
    than its slot's, or a ``data_provenance`` other than the first report read.
    """
    unknown = sorted(set(reports) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"unknown report slots {unknown}; expected {sorted(EXPERIMENTS)}")
    inputs: dict[str, dict[str, Any]] = {}
    spreads: list[Spread] = []
    shared: tuple[str, dict[str, Any]] | None = None
    for slot in EXPERIMENTS:
        report = reports.get(slot)
        if report is None:
            inputs[slot] = {"status": "missing", "result_digest": None}
            continue
        digest, provenance = _checked_report(slot, report, dataset)
        if shared is None:
            shared = (slot, provenance)
        elif provenance != shared[1]:
            raise ValueError(f"{slot} report's data_provenance differs from {shared[0]}'s")
        spreads.extend(spreads_from_report(report))
        inputs[slot] = {"status": "read", "result_digest": digest}
    found = floors(spreads)
    return {
        "dataset": dataset,
        "data_provenance": {} if shared is None else shared[1],
        "inputs": inputs,
        "floors": {
            metric: {"value": floor.value, "source": floor.source.as_dict()}
            for metric, floor in found.items()
        },
        "published": compare(found, differences, dataset),
    }
