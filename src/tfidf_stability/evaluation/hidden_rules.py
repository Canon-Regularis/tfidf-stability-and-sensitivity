"""Re-rank each condition under the tie rules of trec_eval, Anserini, FAISS.

Each rule's means are set against the McSherry-Najork tie-aware expectation, which no
tie-break enters, so a rule's delta is the effect of that rule alone.
Rationale: docs/engineering_notes.md#hidden-tie-rules
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Context, Decimal
from enum import Enum
from typing import Any

from tfidf_stability.analysis.query_grid import EvaluatedQuery, QueryGrid
from tfidf_stability.evaluation.metrics import (
    MeanMetric,
    mean_over_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from tfidf_stability.evaluation.retrieval import evaluate_retrieval
from tfidf_stability.evaluation.score_store import ScoreStore, StoreEntry, StoreKind
from tfidf_stability.evaluation.tie_aware import expected_means
from tfidf_stability.utils.numerics import Reduction, float32_round

__all__ = [
    "HiddenRule",
    "anserini_adjust",
    "anserini_round",
    "java_round",
    "rank_anserini",
    "rank_faiss",
    "rank_trec_eval",
    "rule_means",
    "run_hidden_rules",
    "trec_text",
]

#: Java's ``long`` range, where ``Math.round`` saturates.
_LONG_MIN = -(2**63)
_LONG_MAX = 2**63 - 1
#: The adjuster's ``1e-4f`` tie gap and ``1e-6f`` step, as float32 values.
_TIE_GAP = float32_round(1e-4)
_STEP = float32_round(1e-6)
#: ``%f`` prints six decimals; the context holds every digit of a binary64 score.
_MICRO = Decimal("0.000001")
_DECIMAL = Context(prec=400)
_LENGTH_PAIR = re.compile(r"bm25-exact_n([0-9]+)\.naive")
#: ``numeric_limits<float>::max()``; FAISS's thresholds start at its negation.
_F32_MAX = 3.4028234663852886e38
_F32_LOWEST = -_F32_MAX
#: ``distance_compute_min_k_reservoir``: from this depth FAISS keeps a reservoir.
_RESERVOIR_MIN_K = 100
#: The stride ``sample_threshold_median3`` reads with, and the bisection's step limit.
_SAMPLE_PRIME = 6700417
_BISECTION_STEPS = 200


class HiddenRule(str, Enum):
    """A tie rule a retrieval tool applies without saying so."""

    TREC_EVAL = "trec_eval"
    ANSERINI = "anserini"
    ANSERINI_MATCHED_ONLY = "anserini_matched_only"
    FAISS_INSERTION = "faiss_insertion"

    def __str__(self) -> str:
        return self.value


def java_round(y: float) -> int:
    """``Math.round(double)``: the nearest integer, a half rounded toward +inf.

    ``y - floor(y)`` is exact unless ``-0.5 < y < 0``, where it exceeds a half either way,
    so the half test cannot err. Saturates to ``long``; refuses a NaN or an infinity.
    """
    if not math.isfinite(y):
        raise ValueError(f"Math.round of {y!r} is not modelled")
    r = math.floor(y)
    r += y - r >= 0.5
    return min(max(r, _LONG_MIN), _LONG_MAX)


def anserini_round(score32: float) -> float:
    """The adjuster's ``(float) (Math.round(score * 1e4d) / 1e4d)``.

    The product and the quotient are binary64, as Java evaluates them; the result is float32.
    """
    return float32_round(java_round(score32 * 1e4) / 1e4)


def anserini_adjust(ranked: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """ScoreTiesAdjusterReranker over float32 scores in Lucene order.

    Each score is rounded; one within ``1e-4f`` of its adjusted predecessor joins its run
    and drops by ``1e-6f`` times its place in the run. Every step is in float32.
    """
    out: list[tuple[str, float]] = []
    dup = 0
    for doc_id, score in ranked:
        current = anserini_round(score)
        if out and float32_round(out[-1][1] - current) <= _TIE_GAP:
            dup += 1
            current = float32_round(current - float32_round(_STEP * dup))
        else:
            dup = 0
        out.append((doc_id, current))
    return out


def trec_text(score: float) -> str:
    """The score as Anserini's run writer prints it with ``%f``.

    Java's Formatter rounds half up the decimal ``Double.toString`` selects, the shortest
    one, as ``repr`` gives it. Raises ``ValueError`` for a NaN or an infinity.
    """
    if not math.isfinite(score):
        raise ValueError(f"%f of {score!r} is not modelled")
    return format(Decimal(repr(score)).quantize(_MICRO, ROUND_HALF_UP, _DECIMAL), "f")


def _pairs(scores: Sequence[float], doc_ids: Sequence[str], depth: int) -> list[tuple[str, float]]:
    """``(doc_id, score)`` pairs; refuses mismatched lengths and ``depth < 1``."""
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    if len(scores) != len(doc_ids):
        raise ValueError(f"{len(scores)} scores for {len(doc_ids)} documents")
    return list(zip(doc_ids, scores, strict=True))


def _trec_eval_order(pairs: Iterable[tuple[str, float]], depth: int) -> list[str]:
    """Score descending, then docno descending by bytes, as ``comp_sim_docno`` sorts."""
    ordered = heapq.nlargest(depth, pairs, key=lambda p: (p[1], p[0].encode("utf-8")))
    return [doc_id for doc_id, _ in ordered]


def rank_trec_eval(scores: Sequence[float], doc_ids: Sequence[str], depth: int) -> list[str]:
    """The first ``depth`` documents in the order trec_eval scores a run."""
    return _trec_eval_order(_pairs(scores, doc_ids, depth), depth)


def _faiss_heap(stream: Iterable[tuple[float, int]], depth: int) -> list[tuple[float, int]]:
    """FAISS's Top1 and heap handlers: a full heap evicts its lowest (score, id)."""
    heap: list[tuple[float, int]] = []
    for pair in stream:
        if len(heap) < depth:
            heapq.heappush(heap, pair)
        elif heap[0][0] < pair[0]:
            heapq.heapreplace(heap, pair)
    return sorted(heap, reverse=True)


def _float32_up(x: float) -> float:
    """``nextafterf(x, +inf)`` of ``x`` as float32; past float32's max, inf."""
    x32 = min(float32_round(x), _F32_MAX) + 0.0
    (bits,) = struct.unpack("<I", struct.pack("<f", x32))
    (up,) = struct.unpack("<f", struct.pack("<I", bits + 1 if x32 >= 0.0 else bits - 1))
    return float(up)


def _median3(a: float, b: float, c: float) -> float:
    return sorted((a, b, c))[1]


def _sample_threshold(vals: Sequence[float], upper: float, lower: float) -> float:
    """``sample_threshold_median3``: a median of three in ``(lower, upper)``.

    The values are read with a prime stride; one or two found give the first, none
    gives ``upper``.
    """
    n = len(vals)
    found: list[float] = []
    for i in range(n):
        v = vals[i * _SAMPLE_PRIME % n]
        if lower < v < upper:
            found.append(v)
            if len(found) == 3:
                return _median3(found[0], found[1], found[2])
    return found[0] if found else upper


def _compress(vals: list[float], ids: list[int], thresh: float, ties: int) -> None:
    """``compress_array``: values above ``thresh`` and the first ``ties`` at it.

    Kept entries move to the front in array order; later slots keep their old contents.
    """
    write = 0
    for v, i in list(zip(vals, ids, strict=True)):
        if thresh < v or (ties > 0 and v == thresh):
            ties -= v == thresh
            vals[write], ids[write] = v, i
            write += 1


def _partition(vals: list[float], ids: list[int], q_min: int, q_max: int) -> tuple[float, int]:
    """``partition_fuzzy_median3`` keeping the largest values, compacted in place.

    Bisects on sampled medians for a threshold that keeps ``q_min`` to ``q_max`` values.
    Returns the threshold and the count FAISS records as stored.
    """
    upper, lower = _F32_MAX, _F32_LOWEST
    thresh = _median3(vals[0], vals[len(vals) // 2], vals[-1])
    q = n_above = 0
    for _ in range(_BISECTION_STEPS):
        n_above = sum(thresh < v for v in vals)
        n_equal = sum(v == thresh for v in vals)
        if n_above <= q_min:
            if n_above + n_equal >= q_min:
                q = q_min
                break
            upper = thresh
        elif n_above <= q_max:
            q = n_above
            break
        else:
            lower = thresh
        sampled = _sample_threshold(vals, upper, lower)
        if sampled == upper:
            break
        thresh = sampled
    ties = q - n_above
    if ties < 0:
        # No threshold found: FAISS steps it to the next float32 up and records q_min.
        q = ties = q_min
        thresh = _float32_up(thresh)
    _compress(vals, ids, thresh, ties)
    return thresh, q


def _faiss_reservoir(stream: Iterable[tuple[float, int]], depth: int) -> list[tuple[float, int]]:
    """FAISS's ReservoirTopN for one query, read out by ``to_result``.

    An arrival above the threshold that finds the reservoir full partitions it, then is
    stored regardless. ``to_result`` heaps the first ``depth`` stored, as the heap handler.
    """
    capacity = (2 * depth + 15) & ~15
    vals = [0.0] * capacity
    ids = [0] * capacity
    size = 0
    threshold = _F32_LOWEST
    for score, label in stream:
        if threshold < score:
            if size == capacity:
                threshold, size = _partition(vals, ids, depth, (capacity + depth) // 2)
            vals[size], ids[size] = score, label
            size += 1
    return _faiss_heap(zip(vals[:size], ids[:size], strict=True), depth)


def rank_faiss(
    scores: Sequence[float], doc_ids: Sequence[str], insertion: Mapping[str, int], depth: int
) -> list[str]:
    """The ``depth`` results of FAISS's sequential inner-product scan.

    Documents arrive by insertion id; only scores above float32's lowest enter. Below depth
    100 FAISS's Top1 or heap handler keeps them, from 100 its reservoir handler. The result
    reads score, then id, descending. The scores are ranked as given, not cast.
    """
    pairs = _pairs(scores, doc_ids, depth)
    missing = [doc_id for doc_id in doc_ids if doc_id not in insertion]
    if missing:
        raise ValueError(f"no insertion id for {missing[0]!r}")
    by_id = {insertion[doc_id]: doc_id for doc_id in doc_ids}
    stream = sorted(
        ((score, insertion[doc_id]) for doc_id, score in pairs if score > _F32_LOWEST),
        key=lambda p: p[1],
    )
    handler = _faiss_heap if depth < _RESERVOIR_MIN_K else _faiss_reservoir
    return [by_id[i] for _, i in handler(stream, depth)]


def rank_anserini(
    scores: Sequence[float], doc_ids: Sequence[str], depth: int, *, matched_only: bool
) -> list[str]:
    """What trec_eval ranks from an Anserini run over these scores.

    Float32 scores in Lucene order (score descending, docid ascending), adjusted, printed
    and read back. ``matched_only`` keeps only positive scores, as Lucene returns.
    """
    pairs = _pairs(scores, doc_ids, depth)
    cast = [(doc_id, float32_round(s)) for doc_id, s in pairs if s > 0.0 or not matched_only]
    lucene = heapq.nsmallest(depth, cast, key=lambda p: (-p[1], p[0].encode("utf-8")))
    printed = [(doc_id, float(trec_text(s))) for doc_id, s in anserini_adjust(lucene)]
    return _trec_eval_order(printed, depth)


def _ranked(
    rule: HiddenRule, query: EvaluatedQuery, depth: int, insertion: Mapping[str, int]
) -> list[str]:
    scores, ids = query.scores, query.candidate_ids
    if rule is HiddenRule.TREC_EVAL:
        return rank_trec_eval(scores, ids, depth)
    if rule is HiddenRule.FAISS_INSERTION:
        return rank_faiss(scores, ids, insertion, depth)
    return rank_anserini(scores, ids, depth, matched_only=rule is HiddenRule.ANSERINI_MATCHED_ONLY)


def _metrics(
    ranked: Sequence[str], judgements: Mapping[str, int], k: int, depth: int
) -> dict[str, float | None]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked, judgements, k),
        f"recall@{depth}": recall_at_k(ranked, judgements, depth),
        f"rr@{k}": reciprocal_rank_at_k(ranked, judgements, k),
    }


def rule_means(
    grid: QueryGrid,
    qrels: Mapping[str, Mapping[str, int]],
    doc_ids: Sequence[str],
    *,
    k: int,
    depth: int,
    insertion: Mapping[str, int],
) -> dict[str, dict[str, MeanMetric]]:
    """Each rule's means over every judged query, named as in E5.

    A judged query missing from ``grid`` is degenerate and ranks nothing. Raises
    ``ValueError`` unless ``1 <= k <= depth``, or for a query that misses a document.
    """
    if not 1 <= k <= depth:
        raise ValueError(f"need 1 <= k <= depth, got k={k}, depth={depth}")
    corpus = tuple(doc_ids)
    scored = {q.query_id: q for q in grid.queries}
    for whole in grid.queries:
        if whole.candidate_ids != corpus:
            raise ValueError(f"query {whole.query_id!r} does not rank the whole corpus")
    rules = list(HiddenRule)
    rows: dict[str, dict[str, dict[str, float | None]]] = {str(rule): {} for rule in rules}
    for query_id in sorted(qrels, key=lambda q: q.encode("utf-8")):
        query = scored.get(query_id)
        for rule in rules:
            ranked = [] if query is None else _ranked(rule, query, depth, insertion)
            rows[str(rule)][query_id] = _metrics(ranked, qrels[query_id], k, depth)
    names = (f"ndcg@{k}", f"recall@{depth}", f"rr@{k}")
    return {
        rule: {
            name: mean_over_queries({q: row[name] for q, row in per_query.items()})
            for name in names
        }
        for rule, per_query in rows.items()
    }


def _mean(mean: MeanMetric) -> dict[str, Any]:
    return {"mean": mean.mean, "n_queries": mean.n_queries, "n_undefined": mean.n_undefined}


def _float32_grid(grid: QueryGrid) -> QueryGrid:
    """``grid`` with every score cast to float32, as Lucene holds it."""
    queries = tuple(
        dataclasses.replace(q, scores=tuple(map(float32_round, q.scores))) for q in grid.queries
    )
    return dataclasses.replace(grid, queries=queries)


def _cutoff_row(
    grids: tuple[QueryGrid, QueryGrid],
    qrels: Mapping[str, Mapping[str, int]],
    doc_ids: Sequence[str],
    *,
    k: int,
    depth: int,
    insertion: Mapping[str, int],
) -> dict[str, Any]:
    """One label at one cutoff: the neutral values and each rule's delta from them."""
    grid, cast = grids
    neutral = expected_means(grid, qrels, k=k, depth=depth)
    by_rule = rule_means(grid, qrels, doc_ids, k=k, depth=depth, insertion=insertion)
    return {
        "depth": depth,
        "neutral": {name: _mean(m) for name, m in neutral.items()},
        "anserini_float32_neutral": {
            name: _mean(m) for name, m in expected_means(cast, qrels, k=k, depth=depth).items()
        },
        "rules": {
            rule: {
                name: {"mean": m.mean, "delta_vs_neutral": m.mean - neutral[name].mean}
                for name, m in means.items()
            }
            for rule, means in by_rule.items()
        },
    }


def _length_pairs(labels: Iterable[str]) -> list[tuple[str, str]]:
    """``(exact, lucene_byte)`` naive-policy BM25 labels stored for the same N."""
    present = set(labels)
    pairs: list[tuple[str, str]] = []
    for label in sorted(present, key=lambda s: s.encode("utf-8")):
        match = _LENGTH_PAIR.fullmatch(label)
        byte = f"bm25-lucene_byte_n{match[1]}.naive" if match else None
        if byte in present:
            pairs.append((label, byte))
    return pairs


def _length_row(
    store: ScoreStore,
    pair: tuple[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    rows: Mapping[str, Mapping[str, Any]],
    depths: Mapping[int, int],
) -> dict[str, Any]:
    """Byte lengths against exact lengths: neutral and ``id_asc`` deltas."""
    exact, byte = pair
    out: dict[str, dict[str, dict[str, float]]] = {}
    for k, depth in depths.items():
        asc = [
            evaluate_retrieval(
                store.grid(label), qrels, store.doc_ids, k=k, depth=depth, seeds=0
            ).means["id_asc"]
            for label in pair
        ]
        neutral = [rows[label][str(k)]["neutral"] for label in pair]
        out[str(k)] = {
            name: {
                "neutral_delta": neutral[1][name]["mean"] - neutral[0][name]["mean"],
                "id_asc_delta": asc[1][name].mean - asc[0][name].mean,
            }
            for name in neutral[0]
        }
    return {"labels": [exact, byte], "cutoffs": out}


def _spreads(
    rows: Mapping[str, Mapping[str, Any]], length: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Each rule's ``|delta_vs_neutral|`` and each length pair's ``|neutral_delta|``."""
    out: list[dict[str, Any]] = []
    for label, by_cutoff in rows.items():
        for row in by_cutoff.values():
            for rule, metrics in row["rules"].items():
                out += [
                    {
                        "experiment": "e8",
                        "metric": name,
                        "spread": abs(value["delta_vs_neutral"]),
                        "source": {"label": label, "rule": rule},
                    }
                    for name, value in metrics.items()
                ]
    for pair in length:
        for metrics in pair["cutoffs"].values():
            out += [
                {
                    "experiment": "e8",
                    "metric": name,
                    "spread": abs(value["neutral_delta"]),
                    "source": {"rule": "lucene_byte_length", "labels": list(pair["labels"])},
                }
                for name, value in metrics.items()
            ]
    return out


def _is_ranked(entry: StoreEntry) -> bool:
    """A naive-policy label, or one from a run or a matrix, as E6 selects."""
    return entry.variant == str(Reduction.NAIVE) or entry.kind in (StoreKind.RUN, StoreKind.MATRIX)


def run_hidden_rules(
    store: ScoreStore,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    cutoffs: Sequence[int],
    depth: int,
    insertion_order: Sequence[str],
    insertion_source: str,
) -> dict[str, Any]:
    """E8: each naive-policy, run or matrix label under each rule, against neutral.

    ``insertion_order`` lists the corpus as an index added it, giving FAISS ids; a cutoff
    past ``depth`` ranks to the cutoff. Raises ``ValueError`` for an insertion order that
    is not a permutation of the corpus.
    """
    if sorted(insertion_order) != sorted(store.doc_ids):
        raise ValueError("the insertion order is not a permutation of the corpus")
    insertion = {doc_id: i for i, doc_id in enumerate(insertion_order)}
    depths = {k: max(k, depth) for k in cutoffs}
    labels = [s for s in store.labels if _is_ranked(store.entry(s))]
    rows: dict[str, dict[str, Any]] = {}
    for label in labels:
        grid = store.grid(label)
        grids = (grid, _float32_grid(grid))
        rows[label] = {
            str(k): _cutoff_row(grids, qrels, store.doc_ids, k=k, depth=d, insertion=insertion)
            for k, d in depths.items()
        }
    length = [_length_row(store, pair, qrels, rows, depths) for pair in _length_pairs(labels)]
    return {
        "experiment": "e8",
        "dataset": store.dataset,
        "cutoffs": list(cutoffs),
        "depth": depth,
        "rules": [str(rule) for rule in list(HiddenRule)],
        "insertion": {"source": insertion_source, "n_documents": len(insertion_order)},
        "labels": rows,
        "length_rule": length,
        "spreads": _spreads(rows, length),
    }
