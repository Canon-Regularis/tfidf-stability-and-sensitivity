#!/usr/bin/env python3
"""E3/E4 -- A2: tie-breaking causes discontinuities independent of numerical error.

**E3, the ablation.** Rank every query under pi, pi_score and pi_alt and measure
how far the orderings differ. Because all three consume the *same* score vector,
any disagreement is caused by the tie-break alone -- the scores are bit-identical
across operators by construction, not by assumption, and the harness asserts it.

**E4, the section 7.4 case study.** Identify the closest-scoring pair the corpus
actually contains and show what the operators do with it.

The claim A2 needs, and how it is earned
----------------------------------------
"Independent of numerical error" is the load-bearing phrase. It is earned
structurally rather than statistically: ``rank_all_operators`` shares one
``sorted_scores`` object across the three operators, so there is no arithmetic
between them that could differ. A disagreement therefore has exactly one possible
cause.

Unlike E2, degenerate queries are **included** here -- G3 excludes them from
margin distributions but keeps them in ablations, and rightly so: a zero-score
query is ranked purely by the tie-break, which makes it the *most* informative
case for A2 rather than a nuisance.

Usage::

    python scripts/run_tie_break_ablations.py --dataset synthetic_tiny -o reports/
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.query_grid import build_query_grid, evaluate  # noqa: E402
from tfidf_stability.analysis.stratify import stratify_by_margin  # noqa: E402
from tfidf_stability.analysis.summarise import ExperimentResult, summarise_values  # noqa: E402
from tfidf_stability.analysis.tie_break_ablations import (  # noqa: E402
    ablate_queries,
    disagreement_rate,
)
from tfidf_stability.datasets.loaders import load_dataset  # noqa: E402
from tfidf_stability.datasets.synthetic import find_near_ties  # noqa: E402
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from tfidf_stability.profiles.query_modes import QueryMode  # noqa: E402
from tfidf_stability.ranking.attributes import AttributeTable  # noqa: E402
from tfidf_stability.ranking.ranker import rank_all_operators  # noqa: E402
from tfidf_stability.ranking.tie_groups import (  # noqa: E402
    chain_inflation_ratio,
    tie_chains,
    tie_cliques,
)
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.utils.numerics import same_bits  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402

DEFAULT_KS = (1, 5, 10, 20, 50)


def _rho_sweep(scores: list[float]) -> dict:
    """rho(tau) across the whole span, recorded so the discontinuity is checkable.

    ``rho = |largest chain| / |largest clique|`` is **piecewise constant in tau**,
    and its breakpoints are exactly the observed adjacent gaps: tau can move
    freely between two gaps without changing which pairs are related, and
    changing at a gap merges two chains at once. That is why this is sampled on a
    grid *and* evaluated at the gaps themselves -- a grid alone would land
    between breakpoints and draw a smooth ramp through a function that has none.

    The sampled points are what ``fig_rho_discontinuity`` plots, as a step, for
    the same reason: joining them with straight segments would assert
    intermediate values of rho that the function never takes.
    """
    sorted_scores = sorted(scores, reverse=True)
    gaps = sorted({a - b for a, b in itertools.pairwise(sorted_scores) if a - b > 0.0})
    if not gaps:
        return {"taus": [], "rho": [], "n_chains": [], "n_cliques": [], "breakpoints": []}

    # A log grid over the gap span, plus each gap and a hair either side of it,
    # so every breakpoint is bracketed rather than approached.
    lo, hi = gaps[0], gaps[-1]
    decades = math.log10(hi / lo) if hi > lo else 0.0
    grid = {lo * 10 ** (decades * i / 120) for i in range(121)} if decades else {lo}
    for gap in gaps:
        grid.update({math.nextafter(gap, 0.0), gap, math.nextafter(gap, math.inf)})

    taus = sorted(t for t in grid if t > 0.0)
    rho, n_chains, n_cliques = [], [], []
    for t in taus:
        rho.append(chain_inflation_ratio(sorted_scores, t))
        n_chains.append(len(tie_chains(sorted_scores, t)))
        n_cliques.append(len(tie_cliques(sorted_scores, t)))
    return {
        "taus": taus,
        "rho": rho,
        "n_chains": n_chains,
        "n_cliques": n_cliques,
        # The gaps are the only places rho can change; recorded so a reader can
        # check that every step in the plot sits on one.
        "breakpoints": gaps,
    }


def _case_study(scores: list[float], table: AttributeTable, tau: float, doc_ids: list[str]) -> dict:
    """E4: identify the closest pair and show what the tie-break does with it.

    Section 7.4 says two documents are "*identified* such that |s_A - s_B| <=
    tau". Identified, not constructed -- and G22 explains why that word has to be
    taken literally: ``tf = count / L`` makes a single-token edit a ``1/L``
    *relative* perturbation, so a near-tie at 1e-9 would need a billion-token
    document. The pair is therefore searched for, and whatever separation the
    corpus actually offers is reported rather than assumed.
    """
    sorted_scores = sorted(scores, reverse=True)
    rankings = rank_all_operators(scores, table)

    closest_positive = find_near_ties(sorted_scores, limit=1, strictly_positive=True)
    exact_ties = find_near_ties(sorted_scores, limit=1, strictly_positive=False)

    chains = tie_chains(sorted_scores, tau)
    cliques = tie_cliques(sorted_scores, tau)
    # Intervals are half-open [start, stop), so the size is `stop - start`.
    # Writing `+ 1` here overcounts every group by one and, worse, makes the
    # ratio wrong whenever the chain and clique sizes differ.
    largest_chain = max((b - a for a, b in chains), default=0)
    largest_clique = max((b - a for a, b in cliques), default=0)

    # Section 7.4 asks for the tuple (s_A, s_B, m_k, tau) together with the
    # tie-break attributes of both documents -- because with s_A == s_B the
    # attributes are the *entire* explanation of the outcome, and reporting the
    # scores alone would leave the decision unexplained.
    pair: dict[str, object] = {}
    tightest = exact_ties[0] if exact_ties else None
    if tightest is not None:
        rank = tightest.rank
        order = rankings["pi"].order
        a, b = order[rank - 1], order[rank]
        pair = {
            "rank_of_A": rank,
            "doc_A": doc_ids[a],
            "doc_B": doc_ids[b],
            "s_A": scores[a],
            "s_B": scores[b],
            "s_A_hex": float.hex(scores[a]),
            "s_B_hex": float.hex(scores[b]),
            "m_k": tightest.gap,
            "is_exact_tie": tightest.is_exact,
            "attributes_A": table.attributes_of(a),
            "attributes_B": table.attributes_of(b),
        }

    return {
        "tau": tau,
        "pair": pair,
        "closest_strictly_positive_gap": closest_positive[0].gap if closest_positive else None,
        "closest_pair_rank": closest_positive[0].rank if closest_positive else None,
        "tightest_gap_overall": exact_ties[0].gap if exact_ties else None,
        "tightest_is_exact_tie": exact_ties[0].is_exact if exact_ties else None,
        # The three tie-group objects of G1, which are not interchangeable.
        "n_chains": len(chains),
        "n_cliques": len(cliques),
        "largest_chain": largest_chain,
        "largest_clique": largest_clique,
        "rho_chain_inflation": chain_inflation_ratio(sorted_scores, tau),
        "top_10_by_operator": {name: list(r.order[:10]) for name, r in sorted(rankings.items())},
    }


def _print_case_study(case: dict) -> None:
    """E4's console report: the tuple section 7.4 asks for."""
    print()
    if case["pair"]:
        found = case["pair"]
        print(
            f"E4  identified pair at rank {found['rank_of_A']}: "
            f"{found['doc_A']} vs {found['doc_B']}"
        )
        print(f"    s_A = {found['s_A']!r}  ({found['s_A_hex']})")
        print(f"    s_B = {found['s_B']!r}  ({found['s_B_hex']})")
        print(f"    m_k = {found['m_k']!r}   exact tie: {found['is_exact_tie']}")
        # With s_A == s_B the attributes are the entire explanation of the
        # outcome, so printing the scores without them would leave the decision
        # unexplained -- which is precisely what section 7.4 asks to see.
        print(f"    attributes A: {found['attributes_A']}")
        print(f"    attributes B: {found['attributes_B']}")
    print(f"    closest strictly-positive gap {case['closest_strictly_positive_gap']}")
    print(
        f"    tightest gap overall {case['tightest_gap_overall']} "
        f"(exact tie: {case['tightest_is_exact_tie']})"
    )
    print(
        f"    chains={case['n_chains']} cliques={case['n_cliques']} "
        f"rho={case['rho_chain_inflation']}"
    )


def _parser() -> argparse.ArgumentParser:
    """The command line, kept out of main() so the experiment reads as one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic_tiny")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    parser.add_argument("--queries", type=int, default=40, help="cap on the query grid")
    parser.add_argument(
        "--query-mode",
        choices=[QueryMode.LEAVE_ONE_OUT.value, QueryMode.USER_PROFILE.value],
        default=QueryMode.LEAVE_ONE_OUT.value,
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="G10(4): eligibility is >= 5 qualifying interactions. This default "
        "was 3, contradicting both the addendum and the min_interactions: 5 "
        "pinned in configs/default.yaml, so unattended runs used a threshold "
        "the specification does not sanction.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        required=True,
        help="REQUIRED. tau has no default anywhere in this repository; derive it "
        "with scripts/run_stability_profile.py, which reports the admissible band.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    data = load_dataset(args.dataset, archive=args.archive)
    pipeline = PreprocessingPipeline()
    features = [pipeline.preprocess(str(r["text"])) for r in data.records]
    model = TfidfVectoriser().fit(features, data.doc_ids)

    # Section 7.1's protocol, not document prefixes -- see analysis/query_grid.py.
    grid = evaluate(
        build_query_grid(
            data.interactions,
            dict(zip(data.doc_ids, features, strict=True)),
            data.doc_ids,
            mode=QueryMode(args.query_mode),
            min_interactions=args.min_interactions,
            limit=args.queries,
        ),
        model,
        data.records,
        data.doc_ids,
    )
    if not grid.queries:
        print(
            f"no queries: no user has at least {args.min_interactions} interactions.",
            file=sys.stderr,
        )
        return 1
    scores_by_query = grid.score_vectors()
    tables = [q.table for q in grid.queries]
    print(f"section 7.1 grid: {len(grid)} {args.query_mode} queries")

    # A2's premise, asserted rather than assumed: the three operators consume
    # bit-identical scores, so any disagreement is caused by the tie-break.
    for scores, active in zip(scores_by_query, tables, strict=True):
        rankings = rank_all_operators(scores, active)
        reference = rankings["pi"].sorted_scores
        for name, ranking in rankings.items():
            if not all(
                same_bits(a, b) for a, b in zip(ranking.sorted_scores, reference, strict=True)
            ):
                print(
                    f"operator {name} saw different scores from pi. A2's premise is "
                    f"false and every disagreement rate below is uninterpretable.",
                    file=sys.stderr,
                )
                return 1

    # ---- E3: the ablation ------------------------------------------------
    # Each query has its own candidate set, so k is bounded by the smallest of
    # them and each query is ablated against its own table.
    smallest = min(len(s) for s in scores_by_query)
    ks = tuple(k for k in DEFAULT_KS if k < smallest)
    results = [
        ablate_queries([(q.query_id, list(q.scores))], q.table, ks=ks)[0] for q in grid.queries
    ]
    print(f"E3  {len(scores_by_query)} queries, tau={args.tau:.3e}")

    comparisons = sorted({(p.baseline, p.variant) for r in results for p in r.pairs})
    rates: dict[str, dict] = {}
    for baseline, variant in comparisons:
        label = f"{baseline}_vs_{variant}"
        # The denominator is kept beside every rate: a disagreement rate quoted
        # without its n cannot be distinguished from noise over three queries.
        rates[label] = {
            f"k{k}": dict(
                zip(("rate", "n"), disagreement_rate(results, baseline, variant, k), strict=True)
            )
            for k in ks
        }
        summary = "  ".join(
            f"k{k}={rates[label][f'k{k}']['rate']:.1%}(n={rates[label][f'k{k}']['n']})" for k in ks
        )
        print(f"    {label:22} {summary}")

    # Section 7.3 stratifies the disagreement rate by m_k relative to tau. This
    # is the step that separates A1's regime from A2's: the exact-tie band is
    # deliberately kept out of the smallest numeric band, because a gap of
    # exactly zero is not a small gap -- it is a different phenomenon.
    strata: dict[str, list[dict]] = {}
    for baseline, variant in comparisons:
        label = f"{baseline}_vs_{variant}"
        strata[label] = [
            {
                "band": s.label,
                "k": s.k,
                "lo": s.lo,
                "hi": s.hi,
                "n": s.n,
                "n_disagree": s.n_disagree,
                "rate": s.disagreement_rate,
                "mean_fks": s.mean_fks,
                "mean_jaccard": s.mean_jaccard,
            }
            for s in stratify_by_margin(
                results, args.tau, baseline=baseline, variant=variant, ks=ks
            )
        ]
    print()
    print("E3  stratified by margin band (section 7.3):")
    for label, rows in strata.items():
        for row in rows:
            if row["n"]:
                print(
                    f"    {label:22} {row['band']:14} k={row['k']:<3} "
                    f"{row['rate']:6.1%} (n={row['n']})"
                )

    fks = summarise_values(
        "kendall_fks",
        [p.comparison.fks for r in results for p in r.pairs if p.variant == "pi_alt"],
    )

    # ---- E4: the section 7.4 case study ----------------------------------
    first = grid.queries[0]
    case = _case_study(list(first.scores), first.table, args.tau, list(first.candidate_ids))
    rho_sweep = _rho_sweep(list(first.scores))
    _print_case_study(case)

    result = ExperimentResult(
        experiment="tie_break_ablations",
        parameters={
            "dataset": args.dataset,
            "tau": args.tau,
            "n_queries": len(grid),
            **grid.provenance(),
            "model_digest": model.digest(),
        },
        data_provenance=data.provenance,
        payload={
            "E3_disagreement_rates": rates,
            "E3_stratified_by_margin": strata,
            "E3_fks_distance": fks.as_dict(),
            "E3_degenerate_queries_included": True,  # G3: included here, unlike E1
            "E3_rho_sweep": rho_sweep,
            "E4_case_study": case,
        },
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "tie_break_ablations.json"
    write_json(destination, result.as_dict())
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
