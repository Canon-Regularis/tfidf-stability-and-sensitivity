#!/usr/bin/env python3
"""E1/E2: derive tau, then measure A1 (margins govern ranking stability).

Three things in one pass, since they share a corpus fit and a query grid:

E0, the tau derivation. Measures the arithmetic noise floor against exact
summation, sandwiches tau between it and the smallest observed score gap, and
verifies the tie structure is invariant across the whole band. See
``analysis/noise_floor.py`` and ``docs/spec_addenda.md#g23``.

E1, the margin distribution. ``m_k`` across queries at each ``k``. Degenerate
queries are excluded per G3, with the exclusion count reported.

E2, the stability transition. Empirical top-k flip rate against
``eps / (m_k / 2)``, plus a soundness/conservatism audit of section 4.4's
certificate.

Usage::

    python scripts/run_stability_profile.py --dataset synthetic_tiny -o reports/
    python scripts/run_stability_profile.py --dataset synthetic_small --queries 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.analysis.noise_floor import (  # noqa: E402
    measure_noise_floor,
    tau_band,
    verify_band_invariance,
)
from tfidf_stability.analysis.query_grid import build_query_grid, evaluate  # noqa: E402
from tfidf_stability.analysis.stability_profile import (  # noqa: E402
    certificate_audit,
    transition_curve,
)
from tfidf_stability.analysis.summarise import ExperimentResult, summarise_values  # noqa: E402
from tfidf_stability.datasets.loaders import load_dataset  # noqa: E402
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from tfidf_stability.profiles.query_modes import QueryMode  # noqa: E402
from tfidf_stability.ranking.attributes import AttributeTable  # noqa: E402
from tfidf_stability.ranking.margins import (  # noqa: E402
    boundary_margin,
    min_adjacent_margin_top,
)
from tfidf_stability.ranking.ranker import rank_top_k  # noqa: E402
from tfidf_stability.similarity.cosine import cosine_against_corpus  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.utils.numerics import Reduction  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402

DEFAULT_KS = (1, 5, 10, 20, 50)


def _margin_distributions(
    scores_by_query: list[list[float]], n_documents: int
) -> tuple[dict[str, dict], dict[str, dict[str, int]]]:
    """E1: the distribution of m_k across queries, at each k."""
    dists: dict[str, dict] = {}
    excluded: dict[str, dict[str, int]] = {}
    for k in DEFAULT_KS:
        if k >= n_documents:
            continue
        sorted_vectors = [sorted(v, reverse=True) for v in scores_by_query]
        margins = [boundary_margin(s, k) for s in sorted_vectors]
        # Section 7.2 asks for both margins. They constrain disjoint sets of gaps
        # (m_k at the boundary, m_min^top the smallest gap inside the top-k), so
        # neither bounds the other.
        interior = [min_adjacent_margin_top(s, k) for s in sorted_vectors]

        # G3: degenerate queries are excluded here, counted so the exclusion is
        # auditable. Both lists are counted: the counter once covered the boundary
        # margins alone, which are defined whenever k < n (guaranteed above), so
        # it read a structural zero while G16 left m_min^top undefined at k = 1
        # and all 40 queries dropped out of the interior distribution.
        usable = [m.value for m in margins if m.defined]
        usable_top = [m.value for m in interior if m.defined]
        excluded[f"k{k}"] = {
            "m_k": len(margins) - len(usable),
            "m_min_top": len(interior) - len(usable_top),
        }
        dists[f"k{k}"] = {
            "m_k": summarise_values(f"m_{k}", usable).as_dict(),
            # Undefined at k = 1, where the minimum is over an empty set (G16).
            "m_min_top": summarise_values(f"m_min_top_{k}", usable_top).as_dict(),
        }
        d = dists[f"k{k}"]["m_k"]
        top = dists[f"k{k}"]["m_min_top"]
        print(
            f"E1  k={k:3}  n={d['n']:4}  exact ties {d['share_zero']:6.1%}  "
            f"m_k p50={d['percentiles']['p50']:.3e}  "
            f"m_min^top p50={top['percentiles']['p50']:.3e} (n={top['n']})"
        )
    return dists, excluded


def _rank_trajectories(
    scores: list[float],
    table: AttributeTable,
    k: int,
    *,
    seed: int,
    n_tracked: int = 12,
    n_steps: int = 80,
) -> dict:
    """Each document's rank as a function of eps, along one fixed direction.

    The direction is drawn once and scaled. Redrawing at every eps would make
    each column an unrelated sample, and joining them would draw a path no single
    perturbation ever traced; scaling one direction gives a one-parameter family,
    so a crossing in the picture is a crossing that happens as eps grows.

    Recorded against ``eps / (m_k / 2)`` so the certified radius sits at 1.0,
    matching ``fig_transition``: section 4.4 forbids any crossing left of it.
    """
    import random

    if k >= len(scores):
        return {}
    sorted_scores = sorted(scores, reverse=True)
    margin = boundary_margin(sorted_scores, k)
    if not margin.defined or margin.value <= 0.0:
        # An exact tie at the boundary: the radius is zero, so eps/(m_k/2) is
        # undefined and there is no transition to trace. A2's regime.
        return {}
    radius = margin.value / 2.0

    rng = random.Random(seed)
    direction = [rng.uniform(-1.0, 1.0) for _ in scores]
    baseline = rank_top_k(scores, table, k=len(scores)).order
    tracked = list(baseline[:n_tracked])

    ratios = [10 ** (-1.0 + 3.0 * i / (n_steps - 1)) for i in range(n_steps)]
    positions: dict[str, list[int]] = {str(doc): [] for doc in tracked}
    realised: list[float] = []
    for ratio in ratios:
        eps = radius * ratio
        perturbed = [s + eps * d for s, d in zip(scores, direction, strict=True)]
        # The realised movement; fl(s + d) rounds, so it differs from the intent.
        realised.append(max(abs(p - s) for p, s in zip(perturbed, scores, strict=True)) / radius)
        order = rank_top_k(perturbed, table, k=len(perturbed)).order
        where = {doc: i for i, doc in enumerate(order)}
        for doc in tracked:
            positions[str(doc)].append(where[doc] + 1)

    return {
        "k": k,
        "certified_radius": radius,
        "m_k": margin.value,
        "seed": seed,
        "ratios": ratios,
        "realised_ratios": realised,
        "tracked_documents": [str(d) for d in tracked],
        "ranks": positions,
        "n_candidates": len(scores),
    }


def _parser() -> argparse.ArgumentParser:
    """The command line, kept out of main() so the experiment reads as one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic_tiny")
    parser.add_argument("--archive", type=Path, default=None, help="MovieLens zip")
    parser.add_argument("-o", "--output", type=Path, default=REPO / "reports")
    parser.add_argument("--queries", type=int, default=40, help="cap on the query grid")
    parser.add_argument(
        "--query-mode",
        choices=[QueryMode.LEAVE_ONE_OUT.value, QueryMode.USER_PROFILE.value],
        default=QueryMode.LEAVE_ONE_OUT.value,
        help="section 7.1's construction; item-as-query is implemented but the "
        "paper excludes it from the reported experiments",
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
    parser.add_argument("--k", type=int, default=10, help="k for the transition curve")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


def main() -> int:
    args = _parser().parse_args()

    data = load_dataset(args.dataset, archive=args.archive)
    pipeline = PreprocessingPipeline()
    features = [pipeline.preprocess(str(r["text"])) for r in data.records]
    model = TfidfVectoriser().fit(features, data.doc_ids)
    table = AttributeTable.from_records(data.records)
    documents = [model.document(i) for i in range(model.n_documents)]

    # Section 7.1's protocol: user-profile or leave-one-out queries, each with
    # its own candidate set. Document prefixes are a much easier protocol (a
    # prefix always retrieves its source document) and would move every A1 number.
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
            f"no queries: no user has at least {args.min_interactions} interactions "
            f"in {args.dataset}. Lower --min-interactions or use a larger dataset.",
            file=sys.stderr,
        )
        return 1
    print(
        f"section 7.1 grid: {len(grid)} {args.query_mode} queries, "
        f"{grid.provenance()['n_candidates_min']}-"
        f"{grid.provenance()['n_candidates_max']} candidates each"
    )

    # The noise floor is a property of the arithmetic, so it is measured over
    # the whole corpus rather than per candidate set.
    queries = [
        TfidfVectoriser.transform_query(list(features[i]), model)
        for i in range(0, len(features), max(1, len(features) // 25))
    ]

    # ---- E0: the tau derivation -----------------------------------------
    floor = measure_noise_floor(model, queries)
    exact_norms = model.matrix.row_norms(Reduction.EXACT)
    sorted_vectors = [
        sorted(cosine_against_corpus(q, documents, exact_norms, Reduction.EXACT), reverse=True)
        for q in queries
    ]
    band = tau_band(floor, sorted_vectors)
    invariant = verify_band_invariance(band, sorted_vectors)

    print(f"E0  eta = {floor.eta:.4e}   tau_floor = {band.tau_floor:.4e}")
    print(f"    g_min = {band.g_min:.4e}   band = {band.decades:.2f} decades")
    print(f"    valid={band.is_valid}  invariant={band.is_invariant}  recomputed={invariant}")
    if not band.is_valid:
        print(
            "\n    THE BAND IS EMPTY. Arithmetic noise reaches the decision boundary on\n"
            "    this corpus, so no tau separates numerical error from tie structure and\n"
            "    the A1/A2 separation does not hold here. This is a finding, not a bug.",
            file=sys.stderr,
        )

    # ---- E1: margin distributions ---------------------------------------
    scores_by_query = grid.score_vectors()
    tables = [q.table for q in grid.queries]
    smallest_candidate_set = min(len(s) for s in scores_by_query)
    margin_dists, excluded_by_k = _margin_distributions(scores_by_query, smallest_candidate_set)

    # ---- E2: the transition curve and the certificate audit --------------
    points, n_used, n_excluded = transition_curve(
        scores_by_query, table, args.k, seed=args.seed, trials=args.trials, tables=tables
    )
    trajectories = _rank_trajectories(
        list(grid.queries[0].scores), grid.queries[0].table, args.k, seed=args.seed + 2
    )
    audit = certificate_audit(scores_by_query, table, args.k, seed=args.seed + 1, tables=tables)

    print(f"\nE2  k={args.k}  queries used={n_used} excluded (m_k == 0, A2's regime)={n_excluded}")
    for point in points:
        flag = "  <- 4.4 guarantees 0" if point.within_certificate else ""
        print(f"    eps/(m_k/2)={point.ratio:6.2f}  flip rate {point.flip_rate:7.2%}{flag}")
    violations = [p for p in points if p.within_certificate and p.n_flips]
    print(
        f"    certificate sound: {audit.is_sound} over {audit.n_certified} certified "
        f"perturbations (0 certified-but-changed required)"
    )
    print(f"    conservatism: {audit.conservatism:.1%} of uncertified cases were unchanged")
    print(f"    excluded from the audit: {audit.n_exact_tie} exact-tie queries (A2's regime)")

    if violations or not audit.is_sound:
        print(
            "\n    SECTION 4.4 WAS VIOLATED. This falsifies the theorem or the code.",
            file=sys.stderr,
        )
    elif not audit.is_conclusive:
        # Soundness is "the certified cell holds no failures", which also holds
        # when the cell is empty. Reporting that as a pass is how a gate comes to
        # certify a theorem it never exercised.
        print(
            "\n    THE AUDIT WAS VACUOUS. No perturbation landed inside the certified\n"
            "    radius, so section 4.4 was never exercised and its soundness here is\n"
            "    an empty statement rather than evidence.",
            file=sys.stderr,
        )

    result = ExperimentResult(
        experiment="stability_profile",
        parameters={
            "dataset": args.dataset,
            "n_queries": len(grid),
            **grid.provenance(),
            "k": args.k,
            "trials": args.trials,
            "seed": args.seed,
            "reduction": str(model.reduction),
            "model_digest": model.digest(),
        },
        data_provenance=data.provenance,
        payload={
            "E0_tau_derivation": {
                "noise_floor": floor.as_dict(),
                "band": band.as_dict(),
                "band_invariance_recomputed": invariant,
            },
            "E1_margin_distributions": margin_dists,
            "E1_excluded_degenerate": excluded_by_k,
            "E2_transition": {
                "k": args.k,
                "n_queries_used": n_used,
                "n_queries_excluded_exact_tie": n_excluded,
                "points": [p.as_dict() for p in points],
                "certificate_audit": audit.as_dict(),
                "rank_trajectories": trajectories,
                "violations": len(violations),
            },
        },
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "stability_profile.json"
    write_json(destination, result.as_dict())
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0 if audit.is_sound and audit.is_conclusive and not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
