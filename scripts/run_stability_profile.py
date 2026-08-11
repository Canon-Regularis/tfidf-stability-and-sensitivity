#!/usr/bin/env python3
"""E1/E2 -- derive tau, then measure A1 (margins govern ranking stability).

Runs three things in one pass, because they share a corpus fit and a query grid:

**E0, the tau derivation.** Measures the arithmetic noise floor against exact
summation, sandwiches tau between it and the smallest observed score gap, and
verifies the tie structure is invariant across the whole band. See
``analysis/noise_floor.py`` and ``docs/spec_addenda.md#g23``.

**E1, the margin distribution.** The distribution of ``m_k`` across queries at
each ``k``. Degenerate queries are excluded per G3; the exclusion count is
reported, never silently applied.

**E2, the stability transition.** The empirical top-k flip rate as a function of
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
from tfidf_stability.similarity.cosine import cosine_against_corpus  # noqa: E402
from tfidf_stability.utils.io import write_json  # noqa: E402
from tfidf_stability.utils.numerics import Reduction  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser  # noqa: E402

DEFAULT_KS = (1, 5, 10, 20, 50)


def _margin_distributions(
    scores_by_query: list[list[float]], n_documents: int
) -> tuple[dict[str, dict], dict[str, int]]:
    """E1: the distribution of m_k across queries, at each k."""
    dists: dict[str, dict] = {}
    excluded: dict[str, int] = {}
    for k in DEFAULT_KS:
        if k >= n_documents:
            continue
        sorted_vectors = [sorted(v, reverse=True) for v in scores_by_query]
        margins = [boundary_margin(s, k) for s in sorted_vectors]
        # Section 7.2 asks for BOTH margins. They constrain disjoint sets of
        # gaps -- m_k is the gap at the boundary, m_min^top the smallest gap
        # inside the top-k -- so neither bounds the other and reporting only one
        # would leave the ordering guarantee unquantified.
        interior = [min_adjacent_margin_top(s, k) for s in sorted_vectors]

        # G3: degenerate queries are excluded from margin distributions. Counted
        # and reported, so the exclusion is auditable rather than assumed.
        usable = [m.value for m in margins if m.defined]
        usable_top = [m.value for m in interior if m.defined]
        excluded[f"k{k}"] = len(margins) - len(usable)
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
    parser.add_argument("--min-interactions", type=int, default=3)
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
    # its own candidate set. Document prefixes would be a different and much
    # easier protocol -- a prefix always retrieves its own source document -- so
    # using them would change every A1 number.
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
    audit = certificate_audit(scores_by_query, table, args.k, seed=args.seed + 1, tables=tables)

    print(f"\nE2  k={args.k}  queries used={n_used} excluded (m_k == 0, A2's regime)={n_excluded}")
    for point in points:
        flag = "  <- 4.4 guarantees 0" if point.within_certificate else ""
        print(f"    eps/(m_k/2)={point.ratio:6.2f}  flip rate {point.flip_rate:7.2%}{flag}")
    violations = [p for p in points if p.within_certificate and p.n_flips]
    print(f"    certificate sound: {audit.is_sound} (0 certified-but-changed required)")
    print(f"    conservatism: {audit.conservatism:.1%} of uncertified cases were unchanged")

    if violations or not audit.is_sound:
        print(
            "\n    SECTION 4.4 WAS VIOLATED. This falsifies the theorem or the code.",
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
                "violations": len(violations),
            },
        },
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "stability_profile.json"
    write_json(destination, result.as_dict())
    print(f"\nwritten {destination}\nresult digest {result.digest()}")
    return 0 if audit.is_sound and not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
