#!/usr/bin/env python3
"""Render the figures from the experiment JSON, never from a live computation.

A figure is a view of a recorded result and its caption carries that result's
digest, so a plot traces back to the run that produced it. A figure that
recomputed its own data would drift from the numbers in the text the first time a
default changed.

Matplotlib is an optional dependency (``pip install tfidf-stability[viz]``); the
normative pipeline never imports it.

Each figure and its falsification
---------------------------------
``fig_transition``: flip rate against ``eps / (m_k / 2)``. If A1 were false the
curve would rise before 1.0, so any non-zero point left of the dashed line
falsifies section 4.4 or the code.

``fig_margins``: the distribution of ``m_k`` at each ``k``, log-scaled, with the
exact-tie share annotated. Comfortably large margins would sit far from zero; the
exact-tie mass is the finding.

``fig_tau_band``: the admissible band for ``tau``, from the arithmetic noise floor
to the smallest observed gap, on a log axis. An empty band would mean no ``tau``
separates numerical error from tie structure, a finding about the corpus.

``fig_rho_discontinuity``: ``rho(tau)`` as a step function. ``rho`` changes only
at an observed gap, so interpolating would assert values it never takes.
``rho = 1`` means chains and cliques agree; any step above it is single-linkage
chaining.

``fig_ablation``: disagreement rate by operator pair and ``k``. The operators
consume bit-identical scores, so a false A2 would leave every bar at zero and a
non-zero bar can only come from the tie-break.

Usage::

    python scripts/make_figures.py --reports reports/ -o reports/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Slots 1 and 2 of a validated light-mode categorical palette: adjacent CVD
#: delta-E 24.7 (target >= 8), normal-vision delta-E 33.6 (floor >= 15). Keeps the
#: two series separable for colour-blind readers and in greyscale print.
_INK = "#2a78d6"
_ACCENT = "#eb6834"
#: Recessive ink for chrome and annotation. Text never wears a series colour; the
#: coloured mark beside it carries the identity.
_MUTED = "#52514e"
#: Five slots for the cascade, in the palette's fixed order. Validated as a set:
#: worst adjacent CVD delta-E 9.1, normal-vision 19.6. Three sit below 3:1
#: against the surface, so every series carries a direct label.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"skipping {path.name} (not found -- run its experiment first)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stamp(figure, text: str) -> None:
    """Every figure carries the digest of the result it was built from."""
    figure.text(0.01, 0.01, text, fontsize=6, color="#666666")


def fig_transition(record: dict, out: Path) -> None:
    """A1: the empirical flip rate against the certified radius."""
    import matplotlib.pyplot as plt

    transition = record["payload"]["E2_transition"]
    points = transition["points"]
    ratios = [p["ratio"] for p in points]
    rates = [100.0 * p["flip_rate"] for p in points]
    peak = max(rates) if rates else 100.0

    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    axes.plot(ratios, rates, marker="o", linewidth=1.6, color=_INK)
    axes.axvline(1.0, linestyle="--", color=_ACCENT, linewidth=1.2)
    axes.annotate(
        "certified radius\n$\\epsilon = m_k/2$",
        xy=(1.15, peak * 0.55),
        fontsize=8,
        color=_ACCENT,
    )
    axes.set_xscale("log")
    axes.set_xlabel("$\\epsilon \\,/\\, (m_k/2)$")
    axes.set_ylabel("top-$k$ set flip rate (%)")
    axes.set_title(
        f"Ranking stability transition (k={transition['k']}, "
        f"{transition['n_queries_used']} queries)",
        fontsize=10,
    )
    axes.grid(alpha=0.3, linewidth=0.5)
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  "
        f"{transition['n_queries_excluded_exact_tie']} queries excluded "
        f"(m_k = 0, A2's regime)",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def fig_tau_band(record: dict, out: Path) -> None:
    """G23: tau is derived as a band, and the band has width.

    Lower bound from arithmetic noise, upper bound from the data. A log axis
    shows the gap between them; three numbers in a table make the reader do the
    subtraction. One interval, so no legend; the title names the quantity.

    Falsification: a noise floor reaching the smallest observed gap leaves the
    band empty, so no tau separates numerical error from tie structure. A finding
    about the corpus, visible here.
    """
    import matplotlib.pyplot as plt

    derivation = record["payload"]["E0_tau_derivation"]
    eta = derivation["noise_floor"]["eta"]
    band = derivation["band"]
    g_min, tau, decades = band["g_min"], band["display_tau"], band["decades"]

    # G23 doubles the floor so a difference of two noisy scores clears it: the
    # band is [2 eta, g_min). Drawing from eta would show a band the derivation
    # never claims, so eta is marked separately as the measured floor.
    lower = 2.0 * eta
    figure, axes = plt.subplots(figsize=(6.4, 2.4))
    axes.hlines(0.5, lower, g_min, color=_INK, linewidth=2.0)
    axes.vlines([lower, g_min], 0.38, 0.62, color=_INK, linewidth=2.0)
    axes.vlines([eta], 0.44, 0.56, color=_MUTED, linewidth=1.2)
    axes.plot([tau], [0.5], marker="o", markersize=8, color=_ACCENT, zorder=3)

    # Three labels only: the endpoints define the band, and the marker is the
    # value chosen inside it.
    axes.annotate(
        f"$\\eta$ = {eta:.3e}\nnoise floor",
        xy=(eta, 0.68),
        ha="left",
        fontsize=8,
        color=_MUTED,
    )
    axes.annotate(
        f"$g_{{\\min}}$ = {g_min:.3e}\nsmallest observed gap",
        xy=(g_min, 0.68),
        ha="right",
        fontsize=8,
        color=_MUTED,
    )
    axes.annotate(f"$\\tau$ = {tau:.3e}", xy=(tau, 0.30), ha="center", fontsize=8, color=_ACCENT)

    axes.set_xscale("log")
    axes.set_ylim(0.15, 0.95)
    axes.set_yticks([])
    axes.set_xlabel("score separation")
    axes.set_title(f"The admissible band for $\\tau$ spans {decades:.2f} decades", fontsize=10)
    axes.grid(axis="x", alpha=0.3, linewidth=0.5)
    for side in ("left", "right", "top"):
        axes.spines[side].set_visible(False)
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  "
        f"band [2 eta, g_min) = [{2 * eta:.3e}, {g_min:.3e})",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def fig_rho_discontinuity(record: dict, out: Path) -> None:
    """G1: rho(tau) is piecewise constant, and the steps are the subject.

    rho changes only at an observed gap, so joining the samples with straight
    segments would assert intermediate values the function never takes, smoothing
    away the discontinuity.

    Falsification: rho = 1 means chains and cliques agree, so tie groups are
    unambiguous at that tau. Any step above 1 is single-linkage chaining, and
    its height is how much the chosen tau inflates the largest tie group.
    """
    import matplotlib.pyplot as plt

    sweep = record["payload"].get("E3_rho_sweep")
    if not sweep or not sweep["taus"]:
        print("skipping fig_rho_discontinuity (no rho sweep recorded)")
        return

    taus, rho = sweep["taus"], sweep["rho"]
    peak = max(rho)
    peak_tau = taus[rho.index(peak)]

    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    axes.step(taus, rho, where="post", linewidth=1.6, color=_INK)
    axes.axhline(1.0, color=_MUTED, linewidth=0.8)
    # Headroom, or the top spine clips the peak's label. The baseline note also
    # has to clear the step, which runs along rho = 1 across the left half.
    axes.set_ylim(0.85, peak * 1.14)
    axes.annotate(
        "$\\rho = 1$: chains and cliques agree",
        xy=(taus[0], 1.0),
        xytext=(4, 10),
        textcoords="offset points",
        fontsize=8,
        color=_MUTED,
    )
    # One direct label, on the extreme. Never a number on every step.
    axes.plot([peak_tau], [peak], marker="o", markersize=7, color=_ACCENT, zorder=3)
    axes.annotate(
        f"$\\rho$ = {peak:.2f}",
        xy=(peak_tau, peak),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=_ACCENT,
    )

    axes.set_xscale("log")
    axes.set_xlabel("$\\tau$")
    axes.set_ylabel("$\\rho(\\tau)$ = largest chain / largest clique")
    axes.set_title(
        f"Chain inflation is a step function of $\\tau$ ({len(sweep['breakpoints'])} breakpoints)",
        fontsize=10,
    )
    axes.grid(alpha=0.3, linewidth=0.5)
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  "
        f"{len(taus)} sampled tau, steps only at observed gaps",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def fig_rank_cascade(record: dict, out: Path) -> None:
    """A1, per document: which ranks cross, and when.

    ``fig_transition`` gives the rate of top-k changes; this gives the paths
    behind it. Rank is an integer, so nothing is projected, unlike a 2-D
    embedding of the document vectors where apparent distance is an artefact of
    the projection.

    Emphasis rather than twelve hues: documents crossing the top-k boundary are
    coloured, the rest drawn in recessive grey. Colour follows the entity, so it
    stays put as the rank moves.

    Falsification: section 4.4 certifies the top-k set below eps = m_k/2, so no
    line may cross the k boundary left of 1.0. Crossings wholly inside the top-k
    are permitted there, since the certificate covers set membership rather than
    the order within it; the two are drawn distinctly.
    """
    import matplotlib.pyplot as plt

    trajectories = record["payload"]["E2_transition"].get("rank_trajectories")
    if not trajectories:
        print("skipping fig_rank_cascade (no trajectories recorded)")
        return

    ratios = trajectories["ratios"]
    k = trajectories["k"]
    ranks = trajectories["ranks"]

    # A document "crosses" if it changes side of the top-k boundary anywhere.
    crossers = [
        doc for doc, series in ranks.items() if any((r <= k) != (series[0] <= k) for r in series)
    ]

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    for doc, series in ranks.items():
        if doc in crossers:
            continue
        axes.plot(ratios, series, linewidth=1.0, color="#c9c9c6", zorder=1)
    for index, doc in enumerate(crossers):
        colour = _SERIES[index % len(_SERIES)]
        axes.plot(ratios, ranks[doc], linewidth=1.8, color=colour, zorder=3)
        # Direct labels at the endpoint rather than a legend: three of the five
        # hues sit below 3:1 against the surface and need a visible label anyway.
        axes.annotate(
            f"  {doc}",
            xy=(ratios[-1], ranks[doc][-1]),
            fontsize=7,
            color=colour,
            va="center",
            annotation_clip=False,
        )

    axes.axvline(1.0, linestyle="--", color=_MUTED, linewidth=1.2)
    axes.axhline(k + 0.5, color=_MUTED, linewidth=0.8)
    axes.annotate(
        f"top-{k} boundary",
        xy=(ratios[0], k + 0.5),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=8,
        color=_MUTED,
    )

    axes.set_xscale("log")
    # Ranks are 1-based, and the inverted axis needs headroom or the note above
    # the highest line is clipped by the top spine.
    deepest = max(max(s) for s in ranks.values())
    axes.set_ylim(deepest + 2, -1.5)
    # Placed low-left, where the plot is empty: at the top it collided with the
    # spine and the title, and every line is flat there anyway.
    axes.annotate(
        "certified radius\n$\\epsilon = m_k/2$",
        xy=(1.0, deepest * 0.78),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color=_MUTED,
    )
    axes.set_xlabel("$\\epsilon \\,/\\, (m_k/2)$")
    axes.set_ylabel("rank")
    axes.set_title(
        f"Rank trajectories along one perturbation direction "
        f"({len(crossers)} of {len(ranks)} cross the boundary)",
        fontsize=10,
    )
    axes.grid(alpha=0.3, linewidth=0.5)
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  "
        f"k={k}, one direction scaled over {len(ratios)} steps, seed {trajectories['seed']}",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def fig_margins(record: dict, out: Path) -> None:
    """A1: where the margins actually are, including the exact-tie mass."""
    import matplotlib.pyplot as plt

    dists = record["payload"]["E1_margin_distributions"]
    labels = sorted(dists, key=lambda s: int(s[1:]))
    figure, axes = plt.subplots(figsize=(6.4, 4.0))

    for offset, label in enumerate(labels):
        # E1 reports both margins per k; the boundary margin governs top-k
        # membership, which is this figure's subject.
        d = dists[label]["m_k"]
        percentiles = d["percentiles"]
        # Zero cannot be drawn on a log axis, so the exact-tie share is annotated
        # rather than dropped; it is the finding, and inferring it from a gap in
        # the plot asks too much.
        lo = max(percentiles["p5"], 1e-18)
        hi = max(percentiles["p95"], 1e-18)
        mid = max(percentiles["p50"], 1e-18)
        axes.plot([offset, offset], [lo, hi], color=_INK, linewidth=2.0)
        axes.plot([offset], [mid], marker="o", color=_ACCENT, markersize=5)
        if d["share_zero"] > 0:
            axes.annotate(
                f"{d['share_zero']:.0%} exact",
                xy=(offset, hi * 2.2),
                fontsize=7,
                ha="center",
                color=_ACCENT,
            )

    axes.set_yscale("log")
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels([f"$k$={label[1:]}" for label in labels])
    axes.set_ylabel("$m_k$  (p5-p95, median marked)")
    # m_min^top stays off this axis: it constrains a disjoint set of gaps, so
    # sharing one axis would invite the reading that either bounds the other. It
    # sits in the JSON alongside.
    axes.set_title("Score-separation margins by rank", fontsize=10)
    axes.grid(alpha=0.3, axis="y", linewidth=0.5)
    _stamp(figure, f"result {record['result_digest'][:16]}")
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def fig_ablation(record: dict, out: Path) -> None:
    """A2: disagreement caused by the tie-break alone."""
    import matplotlib.pyplot as plt

    rates = record["payload"]["E3_disagreement_rates"]
    pairs = sorted(rates)
    if not pairs:
        return
    ks = sorted(rates[pairs[0]], key=lambda s: int(s[1:]))
    width = 0.8 / len(pairs)

    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    for i, pair in enumerate(pairs):
        values = [100.0 * rates[pair][k]["rate"] for k in ks]
        positions = [x + i * width for x in range(len(ks))]
        axes.bar(positions, values, width=width, label=pair.replace("_", " "))

    axes.set_xticks([x + width * (len(pairs) - 1) / 2 for x in range(len(ks))])
    axes.set_xticklabels([f"$k$={k[1:]}" for k in ks])
    axes.set_ylabel("top-$k$ set disagreement (%)")
    axes.set_title("Tie-break ablation: disagreement at identical scores", fontsize=10)
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3, axis="y", linewidth=0.5)
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  scores are bit-identical "
        f"across operators, so every bar is caused by the tie-break",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=REPO / "reports")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        import matplotlib
    except ImportError:
        print(
            "matplotlib is not installed. It is an optional dependency:\n"
            '    pip install "tfidf-stability[viz]"',
            file=sys.stderr,
        )
        return 1
    # Non-interactive: this runs in CI, where there is no display.
    matplotlib.use("Agg")

    output = args.output or args.reports / "figures"
    output.mkdir(parents=True, exist_ok=True)

    written = 0
    profile = _load(args.reports / "stability_profile.json")
    if profile:
        fig_transition(profile, output / "fig_transition.png")
        fig_tau_band(profile, output / "fig_tau_band.png")
        fig_rank_cascade(profile, output / "fig_rank_cascade.png")
        fig_margins(profile, output / "fig_margins.png")
        written += 4

    ablation = _load(args.reports / "tie_break_ablations.json")
    if ablation:
        fig_ablation(ablation, output / "fig_ablation.png")
        fig_rho_discontinuity(ablation, output / "fig_rho_discontinuity.png")
        written += 2

    if not written:
        print("no experiment results found; nothing to plot", file=sys.stderr)
        return 1
    print(f"wrote {written} figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
