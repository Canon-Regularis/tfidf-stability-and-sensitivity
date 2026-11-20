"""Whether a top-k set moves only where its margin lets noise move it.

``eta`` is the largest exact score change between a variant and its reference; a top-k
set is predicted to change only when its reference margin at ``k`` is at most ``2 * eta``.
Rationale: docs/engineering_notes.md#margin-against-noise
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.retrieval import evaluate_retrieval
from tfidf_stability.evaluation.score_store import ScoreStore
from tfidf_stability.evaluation.tie_orders import TieOrder, tie_order_label
from tfidf_stability.perturbation.score_bounds import is_top_k_stable
from tfidf_stability.ranking.ranker import rank_top_k
from tfidf_stability.ranking.sort_keys import PI_SCORE
from tfidf_stability.utils.validation import StrictMode

__all__ = [
    "Confusion",
    "Falsification",
    "MarginNoise",
    "Outcome",
    "Scope",
    "classify",
    "falsifies",
    "margin_noise",
    "noise_pairs",
    "pair_means",
    "run_margin_noise",
]


class Scope(str, Enum):
    """Which eta a prediction uses: the pair's largest, or the query's own."""

    GLOBAL = "global"
    QUERY = "query"

    def __str__(self) -> str:
        return self.value


class Outcome(str, Enum):
    """One query's cell in the confusion table."""

    TP = "tp"
    FP = "fp"
    TN = "tn"
    FN = "fn"
    UNDEFINED = "undefined"

    def __str__(self) -> str:
        return self.value


def classify(margin: Fraction | None, eta: Fraction, observed: bool) -> Outcome:
    """The cell for an exact margin, an eta and an observed change.

    A change is predicted when ``margin <= 2 * eta``; an undefined margin predicts none.
    """
    if margin is None:
        return Outcome.UNDEFINED
    if margin <= 2 * eta:
        return Outcome.TP if observed else Outcome.FP
    return Outcome.FN if observed else Outcome.TN


def falsifies(outcome: Outcome, observed: bool) -> bool:
    """Whether a cell holds a change that the bound did not predict."""
    return observed and outcome in (Outcome.FN, Outcome.UNDEFINED)


def _number(name: str, value: Fraction | None) -> dict[str, Any]:
    """``value`` rounded to a float, that float's hex and the exact ``p/q``."""
    if value is None:
        return {name: None, f"{name}_hex": None, f"{name}_exact": None}
    rounded = float(value)
    return {
        name: rounded,
        f"{name}_hex": rounded.hex(),
        f"{name}_exact": f"{value.numerator}/{value.denominator}",
    }


@dataclass(frozen=True, slots=True)
class Confusion:
    """Queries per cell; an undefined margin predicts no change."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    undefined: int = 0

    @classmethod
    def of(cls, outcomes: Iterable[Outcome]) -> Confusion:
        counts = Counter(outcomes)
        return cls(**{str(outcome): counts[outcome] for outcome in list(Outcome)})

    def as_dict(self) -> dict[str, int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "undefined": self.undefined,
        }


@dataclass(frozen=True, slots=True)
class Falsification:
    """A top-k change above ``2 * eta`` or at an undefined margin."""

    query_id: str
    k: int
    scope: Scope
    margin: Fraction | None
    eta: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", Scope(self.scope))

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "k": self.k,
            "scope": str(self.scope),
            **_number("margin", self.margin),
            **_number("eta", self.eta),
        }


@dataclass(frozen=True, slots=True)
class MarginNoise:
    """One variant against its reference at one cutoff, in both scopes.

    ``eta`` is the largest exact score change over every query; ``n_bound_disagreements``
    counts defined margins where ``is_top_k_stable`` contradicts the global prediction.
    """

    reference: str
    variant: str
    k: int
    n_queries: int
    eta: Fraction
    confusion: dict[str, Confusion]
    falsifications: tuple[Falsification, ...]
    n_bound_disagreements: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "variant": self.variant,
            "k": self.k,
            "n_queries": self.n_queries,
            **_number("eta", self.eta),
            "confusion": {scope: cell.as_dict() for scope, cell in self.confusion.items()},
            "falsifications": [f.as_dict() for f in self.falsifications],
            "n_bound_disagreements": self.n_bound_disagreements,
        }


@dataclass(frozen=True, slots=True)
class _Row:
    """One query's exact eta, reference head and change per cutoff."""

    query_id: str
    eta: Fraction
    head: tuple[float, ...]
    changed: dict[int, bool]


def _eta(a: Sequence[float], b: Sequence[float]) -> Fraction:
    """The largest exact ``|a_i - b_i|``; 0 when every pair compares equal."""
    return max(
        (abs(Fraction(x) - Fraction(y)) for x, y in zip(a, b, strict=True) if x != y),
        default=Fraction(0),
    )


def _margin(sorted_scores: Sequence[float], k: int) -> Fraction | None:
    """The exact gap ``s[k-1] - s[k]``, or ``None`` when ``k >= N``."""
    if k >= len(sorted_scores):
        return None
    return Fraction(sorted_scores[k - 1]) - Fraction(sorted_scores[k])


def _ranked(query: EvaluatedQuery, depth: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The top ``depth`` indices under the query's table, and sorted scores."""
    ranking = rank_top_k(query.scores, query.table, PI_SCORE, k=depth, mode=StrictMode.LENIENT)
    return ranking.order[:depth], ranking.sorted_scores


def _observe(reference: QueryGrid, other: QueryGrid, cutoffs: Sequence[int]) -> list[_Row]:
    """Each query's eta and top-k changes, from one ranking per grid."""
    if not cutoffs or min(cutoffs) < 1:
        raise ValueError(f"need at least one cutoff, each at least 1, got {list(cutoffs)}")
    if [q.query_id for q in reference.queries] != [q.query_id for q in other.queries]:
        raise ValueError("the two grids score different queries")
    deepest = max(cutoffs)
    rows: list[_Row] = []
    for ref, cur in zip(reference.queries, other.queries, strict=True):
        if ref.candidate_ids != cur.candidate_ids:
            raise ValueError(f"query {ref.query_id!r} ranks different candidates in the two grids")
        top_ref, ordered = _ranked(ref, deepest)
        top_cur, _ = _ranked(cur, deepest)
        changed = {k: set(top_ref[:k]) != set(top_cur[:k]) for k in cutoffs}
        rows.append(
            _Row(ref.query_id, _eta(ref.scores, cur.scores), ordered[: deepest + 1], changed)
        )
    return rows


def _summarise(rows: Sequence[_Row], k: int, reference_label: str, label: str) -> MarginNoise:
    """Classify every row at ``k`` under the global and the per-query eta."""
    eta_global = max((row.eta for row in rows), default=Fraction(0))
    outcomes: dict[Scope, list[Outcome]] = {scope: [] for scope in list(Scope)}
    found: list[Falsification] = []
    disagreements = 0
    for row in rows:
        margin, observed = _margin(row.head, k), row.changed[k]
        for scope, eta in ((Scope.GLOBAL, eta_global), (Scope.QUERY, row.eta)):
            outcome = classify(margin, eta, observed)
            outcomes[scope].append(outcome)
            if falsifies(outcome, observed):
                found.append(Falsification(row.query_id, k, scope, margin, eta))
        predicted = outcomes[Scope.GLOBAL][-1] in (Outcome.TP, Outcome.FP)
        stable = is_top_k_stable(row.head, k, float(eta_global))
        disagreements += margin is not None and stable == predicted
    return MarginNoise(
        reference=reference_label,
        variant=label,
        k=k,
        n_queries=len(rows),
        eta=eta_global,
        confusion={str(scope): Confusion.of(cells) for scope, cells in outcomes.items()},
        falsifications=tuple(found),
        n_bound_disagreements=disagreements,
    )


def margin_noise(
    reference: QueryGrid, other: QueryGrid, *, k: int, reference_label: str, label: str
) -> MarginNoise:
    """Test the ``2 * eta`` bound for ``other`` against ``reference``.

    Both grids must score the same query ids over the same candidates (``ValueError``
    otherwise); each is ranked under its own table with ``PI_SCORE``.
    """
    return _summarise(_observe(reference, other, (k,)), k, reference_label, label)


def noise_pairs(store: ScoreStore) -> list[tuple[str, str]]:
    """``(reference_label, label)`` for each entry naming a reference."""
    return [
        (reference, label)
        for label in store.labels
        if (reference := store.entry(label).reference_label) is not None
    ]


def pair_means(
    reference: QueryGrid,
    other: QueryGrid,
    qrels: Mapping[str, Mapping[str, int]],
    doc_ids: Sequence[str],
    *,
    k: int,
    depth: int,
) -> dict[str, dict[str, float]]:
    """Each metric's ``id_asc`` means for both grids and their gap.

    The depth is raised to ``k`` when ``k`` exceeds it.
    """
    asc = tie_order_label(TieOrder.ID_ASC)
    deep = max(k, depth)
    ref = evaluate_retrieval(reference, qrels, doc_ids, k=k, depth=deep, seeds=0).means[asc]
    cur = evaluate_retrieval(other, qrels, doc_ids, k=k, depth=deep, seeds=0).means[asc]
    return {
        name: {
            "reference": ref[name].mean,
            "variant": cur[name].mean,
            "abs_delta": abs(ref[name].mean - cur[name].mean),
        }
        for name in ref
    }


def _pair_report(
    reference: QueryGrid,
    other: QueryGrid,
    qrels: Mapping[str, Mapping[str, int]],
    doc_ids: Sequence[str],
    *,
    labels: tuple[str, str],
    cutoffs: Sequence[int],
    depth: int,
) -> dict[str, Any]:
    """One pair's bound test and metric means at every cutoff."""
    rows = _observe(reference, other, cutoffs)
    by_cutoff: dict[str, Any] = {}
    for k in cutoffs:
        noise = _summarise(rows, k, *labels)
        means = pair_means(reference, other, qrels, doc_ids, k=k, depth=depth)
        by_cutoff[str(k)] = {**noise.as_dict(), "means": means}
    return {"reference": labels[0], "variant": labels[1], "by_cutoff": by_cutoff}


def _spreads(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The E7 spread of each metric at each cutoff of one pair."""
    source = {"reference": pair["reference"], "variant": pair["variant"]}
    return [
        {"experiment": "e7", "metric": name, "spread": delta["abs_delta"], "source": dict(source)}
        for report in pair["by_cutoff"].values()
        for name, delta in report["means"].items()
    ]


def run_margin_noise(
    store: ScoreStore,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    cutoffs: Sequence[int],
    depth: int,
) -> dict[str, Any]:
    """E7 over every stored pair: confusions, falsifications and spreads.

    Consecutive pairs sharing a reference reuse its loaded grid.
    """
    pairs: list[dict[str, Any]] = []
    loaded: tuple[str, QueryGrid] | None = None
    for reference_label, label in noise_pairs(store):
        if loaded is None or loaded[0] != reference_label:
            loaded = (reference_label, store.grid(reference_label))
        pairs.append(
            _pair_report(
                loaded[1],
                store.grid(label),
                qrels,
                store.doc_ids,
                labels=(reference_label, label),
                cutoffs=cutoffs,
                depth=depth,
            )
        )
    return {
        "cutoffs": list(cutoffs),
        "depth": depth,
        "pairs": pairs,
        "n_falsifications": sum(
            len(report["falsifications"]) for pair in pairs for report in pair["by_cutoff"].values()
        ),
        "spreads": [spread for pair in pairs for spread in _spreads(pair)],
        "missing": None if pairs else "the store holds no entry with a reference label",
    }
