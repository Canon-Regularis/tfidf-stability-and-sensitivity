#!/usr/bin/env python3
"""Render the figures from the experiment JSON, never from a live computation.

Figures are built from the files the runners wrote, not by recomputing. That
separation is deliberate: a figure that recomputes its own data can silently
disagree with the numbers in the text, and the two would drift the first time a
default changed. Here a figure is a *view* of a recorded result, and its caption
carries that result's digest, so a plot can always be traced to the run that
produced it.

Matplotlib is an optional dependency (``pip install tfidf-stability[viz]``); the
normative pipeline never imports it.

Each figure and its falsification
---------------------------------
**fig_transition** -- flip rate against ``eps / (m_k / 2)``. If A1 were false the
curve would rise before 1.0; section 4.4 forbids that, so any non-zero point left
of the dashed line falsifies the theorem or the code.

**fig_margins** -- the distribution of ``m_k`` at each ``k``, log-scaled, with the
exact-tie share annotated. If margins were comfortably large this would sit far
from zero; the exact-tie mass is the finding.

**fig_ablation** -- disagreement rate by operator pair and ``k``. If A2 were
false every bar would be zero, because the operators consume bit-identical
scores. Any non-zero bar is caused by the tie-break and nothing else.

Usage::

    python scripts/make_figures.py --reports reports/ -o reports/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_INK = "#1f4e79"
_ACCENT = "#b03030"


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


def fig_margins(record: dict, out: Path) -> None:
    """A1: where the margins actually are, including the exact-tie mass."""
    import matplotlib.pyplot as plt

    dists = record["payload"]["E1_margin_distributions"]
    labels = sorted(dists, key=lambda s: int(s[1:]))
    figure, axes = plt.subplots(figsize=(6.4, 4.0))

    for offset, label in enumerate(labels):
        # E1 reports both margins per k; the boundary margin is what governs
        # top-k *membership*, which is what this figure is about.
        d = dists[label]["m_k"]
        percentiles = d["percentiles"]
        # Zero margins cannot be drawn on a log axis, so the exact-tie share is
        # annotated rather than silently dropped -- it is the finding, not an
        # inconvenience, and a reader must not have to infer it from a gap.
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
    # m_min^top is deliberately not overlaid: it constrains a disjoint set of
    # gaps, so plotting the two on one axis would invite the reading that one
    # bounds the other. It is in the JSON alongside.
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
        fig_margins(profile, output / "fig_margins.png")
        written += 2

    ablation = _load(args.reports / "tie_break_ablations.json")
    if ablation:
        fig_ablation(ablation, output / "fig_ablation.png")
        written += 1

    if not written:
        print("no experiment results found; nothing to plot", file=sys.stderr)
        return 1
    print(f"wrote {written} figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
