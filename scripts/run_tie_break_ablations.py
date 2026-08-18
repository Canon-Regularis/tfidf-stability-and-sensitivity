#!/usr/bin/env python3
"""E3/E4 (A2): tie-breaking causes discontinuities independent of numerical error.

E3, the ablation. Rank every query under pi, pi_score and pi_alt and measure how
far the orderings differ. ``rank_all_operators`` shares one ``sorted_scores``
object across the three, so no arithmetic between them can differ and a
disagreement has one available cause. The harness checks the sharing rather than
assuming it.

E4, the section 7.4 case study. Identify the closest-scoring pair the corpus
contains and show what the operators do with it.

Degenerate queries are included here, unlike E2: G3 excludes them from margin
distributions but keeps them in ablations, since a zero-score query is ranked
purely by the tie-break and is therefore the most informative case for A2.

Usage::

    python scripts/run_tie_break_ablations.py --dataset synthetic_tiny \
        --tau 4.8e-13 -o reports/

``--tau`` has no default and the run refuses to start without it: section 7.1
makes every tie-break result conditional on it, so it is passed in from the
derivation in E0 rather than picked up from a default here.
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
    """rho(tau) as an exact step function, evaluated wherever it can move.

    ``rho = |largest chain| / |largest clique|``, and the numerator and the
    denominator move at different places. A chain is single-linkage, so its size
    changes only when tau crosses an adjacent gap. A clique is complete-linkage:
    the largest one reaches size ``m + 1`` exactly when some window of ``m + 1``
    consecutive scores has diameter at most tau, which happens at

        c_m = min over a of (S[a] - S[a + m]),

    a span between scores ``m`` apart rather than between neighbours. Sweeping the
    adjacent gaps therefore locates the chain steps and misses most of the clique
    steps, and a log grid laid over them lands between breakpoints and holds a
    level that is already stale.

    The union of the two sets is every tau at which rho can change and nothing
    else, so the recorded curve is exact rather than sampled. It is also the cheap
    way to get there: at most ``2N - 2`` values, against the ``N(N-1)/2`` distinct
    pairwise differences that enumerating every span would evaluate for the same
    answer. Building the ``c_m`` costs ``O(N^2)`` comparisons and ``O(N)`` space.
    """
    sorted_scores = sorted(scores, reverse=True)
    n = len(sorted_scores)
    empty = {
        "taus": [],
        "rho": [],
        "n_chains": [],
        "n_cliques": [],
        "chain_breakpoints": [],
        "clique_breakpoints": [],
        "rho_breakpoints": [],
        "rho_below_range": math.nan,
    }
    if n < 2:
        return empty

    chain_gaps = {a - b for a, b in itertools.pairwise(sorted_scores) if a - b > 0.0}
    clique_spans = set()
    for m in range(1, n):
        span = min(sorted_scores[a] - sorted_scores[a + m] for a in range(n - m))
        if span > 0.0:
            clique_spans.add(span)

    taus = sorted(chain_gaps | clique_spans)
    if not taus:
        return empty

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
        # Kept apart so the mechanism stays checkable: a step in the curve that
        # sits on a clique span and not on a chain gap is the case the previous
        # sweep could not represent.
        "chain_breakpoints": sorted(chain_gaps),
        "clique_breakpoints": sorted(clique_spans),
        "rho_breakpoints": [t for i, t in enumerate(taus) if i and rho[i] != rho[i - 1]],
        # Below the smallest span nothing is related but the exact ties, so this
        # is rho for every tau under the swept range, including the tau the
        # experiments actually use. Recorded rather than left to be inferred.
        "rho_below_range": chain_inflation_ratio(sorted_scores, 0.0),
    }


def _pair_geometry(
    query_features: list[str],
    model,
    corpus_ids: list[str],
    case: dict,
    tau: float,
) -> dict:
    """Exact 2-D coordinates for the section 7.4 pair, in the plane they span.

    A query ranks A above B iff ``q . d > 0`` for ``d = w_A/||w_A|| -
    w_B/||w_B||``. ``d`` lies in the plane spanned by the two document
    directions, so q's out-of-plane component is orthogonal to ``d`` and
    contributes zero: the projection preserves the ranking decision exactly,
    unlike a t-SNE or PCA scatter where apparent distance is an artefact.

    The projection loses q's magnitude, so the drawn angles from q to each
    document are wrong. Scores are therefore taken from the record, and the share
    of ``||q||`` lying in the plane is reported.

    Presentation only, hence ``fsum`` throughout; the quoted scores come from the
    run's own reduction policy.
    """
    pair = case.get("pair") or {}
    if not pair:
        return {}

    query = TfidfVectoriser.transform_query(list(query_features), model)
    width = model.n_features

    def densify(vector) -> list[float]:
        out = [0.0] * width
        for index, value in zip(vector.indices, vector.values, strict=True):
            out[index] = value
        return out

    def inner(a: list[float], b: list[float]) -> float:
        return math.fsum(x * y for x, y in zip(a, b, strict=True))

    def magnitude(a: list[float]) -> float:
        return math.sqrt(math.fsum(x * x for x in a))

    q = densify(query)
    a = densify(model.document(corpus_ids.index(pair["doc_A"])))
    b = densify(model.document(corpus_ids.index(pair["doc_B"])))
    q_norm, a_norm, b_norm = magnitude(q), magnitude(a), magnitude(b)
    if min(q_norm, a_norm, b_norm) == 0.0:
        return {}

    unit_a = [x / a_norm for x in a]
    unit_b = [x / b_norm for x in b]

    # Gram-Schmidt: e1 along A, e2 the part of B orthogonal to it.
    e1 = unit_a
    overlap = inner(unit_b, e1)
    residual = [unit_b[i] - overlap * e1[i] for i in range(width)]
    residual_norm = magnitude(residual)
    coincident = residual_norm == 0.0
    e2 = [0.0] * width if coincident else [x / residual_norm for x in residual]

    point_a = (1.0, 0.0)
    point_b = (overlap, residual_norm)
    q_plane = (inner(q, e1), inner(q, e2))

    # Check the claim above: the decision statistic agrees between the full space
    # and the plane.
    direction = [unit_a[i] - unit_b[i] for i in range(width)]
    decision_full = inner(q, direction)
    decision_plane = q_plane[0] * (point_a[0] - point_b[0]) + q_plane[1] * (point_a[1] - point_b[1])
    # Scaled by ||q||. The first version divided by the statistic itself, which
    # for an exact tie is 0.0, so a discrepancy of 1e-17 came out as a relative
    # error of 1.0. The statistic is q . d, so q . d over ||q|| is the score gap.
    residual_error = abs(decision_full - decision_plane)

    return {
        "doc_A": pair["doc_A"],
        "doc_B": pair["doc_B"],
        "unit_A": list(point_a),
        "unit_B": list(point_b),
        "q_in_plane": list(q_plane),
        "q_norm": q_norm,
        "q_in_plane_norm": math.hypot(*q_plane),
        # How much of the query is not drawn. 1.0 means it lies wholly in the plane.
        "q_share_in_plane": math.hypot(*q_plane) / q_norm,
        "angle_between_documents_rad": math.acos(max(-1.0, min(1.0, overlap))),
        "documents_coincident": coincident,
        # cos(q, w) as recorded; nothing is measured off the drawing.
        "s_A": pair["s_A"],
        "s_B": pair["s_B"],
        "score_gap": pair["s_A"] - pair["s_B"],
        "tau": tau,
        "decision_statistic_full_space": decision_full,
        "decision_statistic_in_plane": decision_plane,
        "projection_error_absolute": residual_error,
        "projection_error_over_q_norm": residual_error / q_norm,
        # A drawn angle below ~2e-3 rad is under one pixel at 200 dpi, so this
        # says whether a geometric rendering of the pair shows anything at all.
        # For an exact tie it shows nothing and invents structure.
        "angle_is_renderable": math.degrees(math.acos(max(-1.0, min(1.0, overlap)))) > 0.05,
    }


def _case_study(scores: list[float], table: AttributeTable, tau: float, doc_ids: list[str]) -> dict:
    """E4: identify the closest pair and show what the tie-break does with it.

    Section 7.4 says two documents are "identified such that |s_A - s_B| <= tau",
    and G22 forces "identified" to be read literally: ``tf = count / L`` makes a
    single-token edit a ``1/L`` relative perturbation, so a near-tie at 1e-9 needs
    a billion-token document. The pair is searched for, and whatever separation
    the corpus offers is reported.
    """
    sorted_scores = sorted(scores, reverse=True)
    rankings = rank_all_operators(scores, table)

    closest_positive = find_near_ties(sorted_scores, limit=1, strictly_positive=True)
    exact_ties = find_near_ties(sorted_scores, limit=1, strictly_positive=False)

    chains = tie_chains(sorted_scores, tau)
    cliques = tie_cliques(sorted_scores, tau)
    # Intervals are half-open [start, stop), so the size is `stop - start`. A
    # `+ 1` here overcounts every group and skews rho whenever the chain and
    # clique sizes differ.
    largest_chain = max((b - a for a, b in chains), default=0)
    largest_clique = max((b - a for a, b in cliques), default=0)

    # Section 7.4 asks for (s_A, s_B, m_k, tau) together with both documents'
    # tie-break attributes: at s_A == s_B the attributes are the whole
    # explanation of the outcome.
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
        # G1's three tie-group objects; they disagree, so all three are kept.
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
        # At s_A == s_B the attributes are the whole explanation of the outcome.
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

    # Section 7.1's protocol; document prefixes are a different and much easier
    # one. See analysis/query_grid.py.
    query_set = build_query_grid(
        data.interactions,
        dict(zip(data.doc_ids, features, strict=True)),
        data.doc_ids,
        mode=QueryMode(args.query_mode),
        min_interactions=args.min_interactions,
        limit=args.queries,
    )
    grid = evaluate(query_set, model, data.records, data.doc_ids)
    if not grid.queries:
        print(
            f"no queries: no user has at least {args.min_interactions} interactions.",
            file=sys.stderr,
        )
        return 1
    scores_by_query = grid.score_vectors()
    tables = [q.table for q in grid.queries]
    print(f"section 7.1 grid: {len(grid)} {args.query_mode} queries")

    # A2's premise, checked here: the three operators consume bit-identical
    # scores, so any disagreement comes from the tie-break.
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
        # n sits beside every rate: a disagreement rate quoted without its
        # denominator cannot be told from noise over three queries.
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

    # Section 7.3 stratifies the disagreement rate by m_k relative to tau, which
    # separates A1's regime from A2's. The exact-tie band stays out of the
    # smallest numeric band: a zero gap is a different phenomenon from a small one.
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
    geometry = _pair_geometry(
        list(query_set.queries[0].features), model, list(data.doc_ids), case, args.tau
    )
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
            "E4_pair_geometry": geometry,
        },
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "tie_break_ablations.json"
    write_json(destination, result.as_dict())
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
