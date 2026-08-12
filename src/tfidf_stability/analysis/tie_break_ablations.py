"""Tie-break ablations: research question A2 (README sections 4.5 and 7.3).

Section 7.3 recomputes each ranking under three sorting operators

    pi       = Sort(s_i, a_i)                   the full attribute tuple
    pi_score = Sort(s_i, id_i)                  attribute-independent baseline
    pi_alt   = Sort(s_i, a_i reordered)         alternate priority

and measures how far apart the results are. The point is to isolate
**decision-level** instability -- change that arises purely from the choice of
secondary ordering rule -- from numerical instability in the scores themselves.

What makes the isolation clean is a fact the paper does not state and this
implementation makes structural: **all three operators share the same sorted
score array**, so every margin is identical between them (see
:mod:`~tfidf_stability.ranking.margins`). Any disagreement observed here is
therefore attributable to the tie-break alone, with ``delta s = 0`` exactly. That
is what makes A1 and A2 independent questions rather than confounded ones.

Two guards on the interpretation, both worth stating because they are easy to
lose:

* the comparison is only meaningful where ties exist. With all scores distinct
  the three operators coincide exactly, and
  :attr:`AblationResult.scores_all_distinct` records when that was the case;
* ``pi_score`` is the empty-priority special case of ``pi``, so it agrees with
  ``pi`` whenever the attributes happen not to discriminate. That slightly
  weakens the ablation, and it is the honest reading of section 4.5.
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
    #: ``m_k`` for this query. Identical across all three operators -- it depends
    #: only on the sorted score multiset -- which is precisely why it can be used
    #: to *stratify* the disagreement rate without circularity.
    margin: Margin

    @property
    def k(self) -> int:
        return self.comparison.k

    @property
    def sets_differ(self) -> bool:
        return self.comparison.sets_differ


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Every operator comparison for one query, across the requested ``k`` set."""

    query_id: str
    n_documents: int
    pairs: tuple[OperatorPair, ...]
    rankings: dict[str, Ranking]
    #: True when no two documents share a score. The three operators must then
    #: agree exactly; recorded so that a zero disagreement rate can be
    #: distinguished from "the ablation had nothing to bite on".
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

    Args:
        scores: Similarity scores for this query.
        table: The tie-break attribute table.
        query_id: Recorded on the result for provenance.
        ks: The k-set of section 7.1.
        specs: Operators to evaluate; the baseline is skipped against itself.
        baseline: The operator everything is compared against -- ``pi``, since
            section 7.3 asks for "pi versus pi_score" and "pi versus pi_alt".
        n_zero_norm_docs: Passed through for reporting.
        mode: Lenient by default, because a ``k`` larger than the corpus is a
            legitimate grid point in a sweep rather than a configuration error.

    Returns:
        The :class:`AblationResult`.
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

    Returns ``(rate, n)`` -- the fraction of queries whose top-k *set* differs
    between the two operators, and how many queries contributed. The count is
    returned rather than discarded because a rate over three queries and a rate
    over thirty thousand are not the same claim, and section 7.1 requires the
    query count to be reported.
    """
    considered = []
    for result in results:
        # At most ONE pair per query. `ablate_query` clamps k to the candidate
        # count (G3's lenient mode), so several requested k values can collapse
        # onto the same effective k and emit several identical pairs for a single
        # query -- on a 7-document corpus, k in (10, 20, 50) all become 7. Taking
        # them all would count that query three times and report a denominator
        # this docstring promises is a query count.
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
