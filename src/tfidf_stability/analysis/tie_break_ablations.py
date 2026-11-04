"""Tie-break ablations for research question A2 (README sections 4.5 and 7.3).

pi = Sort(s_i, a_i), pi_score = Sort(s_i, id_i) and pi_alt share one sorted score array, so
margins (:mod:`~tfidf_stability.ranking.margins`) match and disagreement is the tie-break's
alone (``delta s = 0``). Rationale: docs/engineering_notes.md#tie-break-ablation-isolation
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.distances import TopKComparison, compare_top_k
from tfidf_stability.ranking.margins import Margin, boundary_margin
from tfidf_stability.ranking.ranker import Ranking, rank_all_operators
from tfidf_stability.ranking.sort_keys import OPERATORS, PI, SortKeySpec
from tfidf_stability.utils.validation import StrictMode

__all__ = [
    "AblationResult",
    "OperatorPair",
    "ablate_queries",
    "ablate_query",
    "disagreement_rate",
]


@dataclass(frozen=True, slots=True)
class OperatorPair:
    """One operator comparison at one ``k``, with the margin that governs it."""

    baseline: str
    variant: str
    comparison: TopKComparison
    #: ``m_k`` for this query. Identical across all three operators (it depends
    #: only on the sorted score multiset), so stratifying the disagreement rate
    #: by it is free of circularity.
    margin: Margin

    @property
    def k(self) -> int:
        return self.comparison.k

    @property
    def sets_differ(self) -> bool:
        return self.comparison.sets_differ


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Every operator comparison for one query, across the requested ``k`` set.

    With all scores distinct the operators coincide; :attr:`AblationResult.scores_all_distinct`
    records that case.
    """

    query_id: str
    n_documents: int
    pairs: tuple[OperatorPair, ...]
    rankings: dict[str, Ranking]
    #: True when no two documents share a score, in which case the three
    #: operators must agree. Recorded so a zero disagreement rate stays
    #: distinguishable from "the ablation had nothing to bite on".
    scores_all_distinct: bool
    query_degenerate: bool
    n_zero_norm_docs: int

    def for_pair(self, baseline: str, variant: str) -> tuple[OperatorPair, ...]:
        return tuple(p for p in self.pairs if p.baseline == baseline and p.variant == variant)

    def at_k(self, k: int) -> tuple[OperatorPair, ...]:
        return tuple(p for p in self.pairs if p.k == k)


def ablate_query(
    scores: Sequence[float],
    table: AttributeTable,
    *,
    query_id: str = "",
    ks: Sequence[int] = (5, 10, 20, 50),
    specs: Sequence[SortKeySpec] = OPERATORS,
    baseline: SortKeySpec = PI,
    n_zero_norm_docs: int = 0,
    mode: StrictMode = StrictMode.LENIENT,
) -> AblationResult:
    """Rank one query under every operator and compare each against the baseline.

    ``ks`` is the k-set of section 7.1. ``baseline`` defaults to ``pi`` (section 7.3 compares pi
    with pi_score and pi_alt) and is skipped against itself. ``mode`` is lenient because a ``k``
    above the corpus size is a valid sweep grid point. Returns the :class:`AblationResult`.
    """
    rankings = rank_all_operators(
        scores, table, specs, mode=mode, n_zero_norm_docs=n_zero_norm_docs
    )
    reference = rankings[baseline.name]
    sorted_scores = reference.sorted_scores
    n = len(scores)

    pairs: list[OperatorPair] = []
    for spec in specs:
        if spec.name == baseline.name:
            continue
        other = rankings[spec.name]
        for k in ks:
            k_eff = min(k, n)
            pairs.append(
                OperatorPair(
                    baseline=baseline.name,
                    variant=spec.name,
                    comparison=compare_top_k(reference.order, other.order, k_eff),
                    margin=boundary_margin(sorted_scores, k, mode=StrictMode.LENIENT),
                )
            )

    return AblationResult(
        query_id=query_id,
        n_documents=n,
        pairs=tuple(pairs),
        rankings=rankings,
        scores_all_distinct=len(set(scores)) == n,
        query_degenerate=reference.query_degenerate,
        n_zero_norm_docs=n_zero_norm_docs,
    )


def ablate_queries(
    queries: Sequence[tuple[str, Sequence[float]]],
    table: AttributeTable,
    **kwargs: object,
) -> list[AblationResult]:
    """Run :func:`ablate_query` over a whole query set."""
    return [
        ablate_query(scores, table, query_id=qid, **kwargs)  # type: ignore[arg-type]
        for qid, scores in queries
    ]


def disagreement_rate(
    results: Sequence[AblationResult],
    baseline: str,
    variant: str,
    k: int,
) -> tuple[float, int]:
    """Section 7.3's headline statistic, with its denominator.

    Returns ``(rate, n)``: the fraction of queries whose top-k set differs between the two
    operators, and the number of queries that contributed. Section 7.1 requires the query count
    to be reported alongside the rate.
    """
    considered = []
    for result in results:
        # At most one pair per query. `ablate_query` clamps k to the candidate count (G3's
        # lenient mode), so several requested k values share one effective k: on a 7-document
        # corpus k in (10, 20, 50) all become 7. Taking only the first match keeps the
        # denominator a query count.
        match = next(
            (
                p
                for p in result.pairs
                if p.baseline == baseline and p.variant == variant and p.k == k
            ),
            None,
        )
        if match is not None:
            considered.append(match)

    if not considered:
        return 0.0, 0
    return sum(p.sets_differ for p in considered) / len(considered), len(considered)
