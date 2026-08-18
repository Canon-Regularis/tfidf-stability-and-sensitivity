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
curve would rise before 1.0. The first few flips are under a pixel, though: each
point is 1320 trials, so one flip is 0.076% and moves the marker 0.46 px on a
linear axis. The caption therefore carries the recorded violation count, and that
number rather than the curve's shape is what falsifies section 4.4.

``fig_margins``: the distribution of ``m_k`` at each ``k``, log-scaled, with the
exact-tie share annotated. Comfortably large margins would sit far from zero; the
exact-tie mass is the finding.

``fig_tau_band``: the admissible band for ``tau``, from the arithmetic noise floor
to the smallest non-zero gap, on a log axis. An empty band would mean no ``tau``
separates numerical error from tie structure, a finding about the corpus.

``fig_rho_discontinuity``: ``rho(tau)`` as a step function, so that interpolating
does not assert values it never takes. ``rho = 1`` means the largest chain and the
largest clique are the same size; any step above it is single-linkage chaining.
Every riser is at the exact ``tau`` where ``rho`` jumps, because the sweep
evaluates the adjacent gaps and the clique thresholds together rather than a grid.

``fig_ablation``: disagreement rate by operator pair and ``k``. The operators
consume bit-identical scores, so a false A2 would leave every bar at zero and a
non-zero bar can only come from the tie-break.

``fig_rank_cascade``: the rank of each document against perturbation size, with
the documents that cross the top-k boundary picked out. The rate behind
``fig_transition``, resolved into the paths that produce it.

``fig_stratified``: the same disagreement conditioned on the margin. A rate right
of the exact-tie band would contradict A2; the empty bands either side of ``tau``
show that the choice of ``tau`` within them cannot move a result.

Usage::

    python scripts/make_figures.py --reports reports/ -o reports/figures/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import pairwise
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

#: The operators as section 2.3.1 writes them. The legend previously showed the
#: payload key with its underscores replaced, so the figure named its series
#: after a JSON identifier while every other axis label used mathtext.
_PAIR_LABELS = {
    "pi_vs_pi_alt": "$\\pi$ vs $\\pi_{\\mathrm{alt}}$",
    "pi_vs_pi_score": "$\\pi$ vs $\\pi_{\\mathrm{score}}$",
}


def _contiguous_runs(populated: list[bool]) -> list[list[int]]:
    """Index runs of consecutive empty bands, for shading.

    Shading each empty band separately leaves white seams between them that read
    as slots holding something. Spanning first-to-last empty index instead is
    simpler and wrong: it would shade over a populated band whenever the empty
    ones are not contiguous. They happen to be on this corpus, which is the
    finding rather than something to rely on.
    """
    runs: list[list[int]] = []
    for position, ok in enumerate(populated):
        if ok:
            continue
        if runs and runs[-1][-1] == position - 1:
            runs[-1].append(position)
        else:
            runs.append([position])
    return runs


def _series_handles(ks: list[int]) -> list:
    """One legend proxy per k, built from the series list rather than from what was
    drawn.

    Labelling inside the plotting loop omitted k=5 entirely: its only appearance is
    in the separated band, and the label was attached in the exact-tie band, where
    k=5 has no queries. The series was on the chart and missing from the key.
    """
    import matplotlib.pyplot as plt

    return [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=_SERIES[i % len(_SERIES)],
            markeredgecolor="white",
            label=f"$k$={k}",
        )
        for i, k in enumerate(ks)
    ]


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, as percentages.

    Every rate in these reports comes from a small sample: the ablation grid is 40
    queries, and a stratified cell can hold one. A bare point estimate says 100%
    where the evidence is one query out of one, whose interval runs from 21% to
    100%. Drawing the estimate without the interval asserts a precision the run
    does not have, which is the one thing these figures exist not to do.

    Wilson rather than the normal approximation: it does not run below 0 or above
    100, and it stays sensible at 0 and n successes, which is exactly where these
    counts sit. Reported at 95%, two-sided.
    """
    if trials <= 0:
        return (math.nan, math.nan)
    p_hat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p_hat + z * z / (2.0 * trials)) / denominator
    spread = z * math.sqrt(p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials))
    spread /= denominator
    return (100.0 * max(0.0, centre - spread), 100.0 * min(1.0, centre + spread))


def _uniform_log_minor(axis) -> None:
    """Put a log axis's minor ticks at half-decades, so every mark is equidistant.

    The default subdivides each decade at 2,3,...,9 times its base. That is what a
    log scale means, and it is unreadable as a ruler: the marks bunch towards the
    top of every decade and the gap between neighbours changes by a factor of four
    across it. Nothing about styling fixes that, because the positions themselves
    are uneven.

    A tick at sqrt(10) times each decade sits exactly half a decade up, so the
    marks are uniformly spaced in device coordinates: one gap, everywhere, at half
    the major spacing. It still denotes something, unlike an arbitrary subdivision.
    """
    from matplotlib.ticker import LogLocator

    axis.set_minor_locator(LogLocator(base=10.0, subs=(10.0**0.5,), numticks=100))


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"skipping {path.name} (not found -- run its experiment first)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stamp(figure, text: str) -> None:
    """Every figure carries the digest of the result it was built from."""
    figure.text(0.01, 0.01, text, fontsize=6, color="#666666")


def fig_transition(record: dict, out: Path) -> bool:
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
        xy=(1.15, peak * 0.30),
        fontsize=8,
        color=_ACCENT,
    )
    axes.set_xscale("log")
    _uniform_log_minor(axes.xaxis)
    axes.tick_params(axis="x", which="minor", length=3.0, width=0.6, color=_MUTED)
    axes.set_xlabel("$\\epsilon \\,/\\, (m_k/2)$")
    # Symlog, with the linear region ending at one flip. Six of the ten points are
    # exactly zero and the first non-zero is a single-digit count, so on a linear
    # axis the whole sub-percent regime collapses onto the axis line: one flip in
    # 1320 trials moves the marker 0.46 px, indistinguishable from zero. That
    # regime is what section 4.4 is about, so the axis has to resolve it. Below
    # linthresh the scale is linear, which is what lets the exact zeros be drawn;
    # a plain log axis has no zero.
    one_flip = 100.0 / max(point["n_trials"] for point in points) if points else 0.1
    axes.set_yscale("symlog", linthresh=one_flip, linscale=0.9)
    # Explicit decades, plus zero. The default symlog locator put two ticks inside
    # the linear window and one just above it, giving gaps of 4, 38 and 83 px on
    # one axis: correct for the scale and unreadable as a ruler. Zero, then the
    # decades, has one deliberate change of gauge at the bottom where the scale
    # genuinely changes, and a uniform ruler above it.
    axes.set_yticks([0.0, 0.1, 1.0, 10.0, 100.0])
    # A flip rate cannot be negative, and symlog draws the negative decades unless
    # told otherwise: the first attempt showed ticks down to -10^0 and collided
    # -10^-2 with 10^-2 at the origin. Clamping at zero keeps the linear window
    # that makes the exact zeros drawable without inventing a negative half.
    axes.set_ylim(0.0, peak * 1.6)
    axes.axhline(one_flip, color=_MUTED, linewidth=0.7, linestyle=":")
    axes.annotate(
        f"one flip = {one_flip:.2f}%",
        xy=(min(ratios), one_flip),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=7,
        color=_MUTED,
    )
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
        f"(m_k = 0, A2's regime)  |  {transition['violations']} recorded "
        f"certificate violations",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return True


def fig_tau_band(record: dict, out: Path) -> bool:
    """G23: tau is derived as a band, and the band has width.

    Lower bound from arithmetic noise, upper bound from the data. A log axis
    shows the gap between them; three numbers in a table make the reader do the
    subtraction. One interval, so no legend; the title names the quantity.

    Falsification: a noise floor reaching the smallest non-zero gap leaves the
    band empty, so no tau separates numerical error from tie structure. A finding
    about the corpus, visible here.

    The exact ties are not on this axis and cannot be: a log scale has no zero, and
    9.5% of adjacent gaps here are exactly 0. They are the subject of fig_margins
    and fig_stratified, and their count is in the caption so the band is not read
    as covering every gap.
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

    # Three labels only: the endpoints define the band, and the marker sits inside
    # it.
    axes.annotate(
        f"$\\eta$ = {eta:.3e}\nnoise floor",
        xy=(eta, 0.68),
        ha="left",
        fontsize=8,
        color=_MUTED,
    )
    axes.annotate(
        f"$g_{{\\min}}$ = {g_min:.3e}\nsmallest non-zero gap",
        xy=(g_min, 0.68),
        ha="right",
        fontsize=8,
        color=_MUTED,
    )
    # display_tau is the band's geometric centre, not the value any experiment ran
    # with: tie_break_ablations.json uses its own parameters.tau = 4.8e-13, and the
    # two differ by a factor of 13. Nothing needs reconciling, because the upper
    # endpoint is the smallest observed positive gap, so every positive gap is at
    # least g_min and none can fall below it. For any tau in the band the relation
    # abs(s_i - s_j) <= tau therefore holds only where the scores are equal, and
    # the tie groups are the exact-equality classes whichever admissible tau is
    # picked. The marker is labelled for what it is, and that consequence is
    # printed rather than n_gaps_in_band, which is 0 by construction and would read
    # as a measurement.
    axes.annotate(
        f"$\\tau$ = {tau:.3e}\nband centre",
        xy=(tau, 0.30),
        ha="center",
        va="top",
        fontsize=8,
        color=_ACCENT,
    )

    axes.set_xscale("log")
    _uniform_log_minor(axes.xaxis)
    axes.tick_params(axis="x", which="minor", length=3.0, width=0.6, color=_MUTED)
    axes.set_ylim(0.15, 0.95)
    axes.set_yticks([])
    axes.set_xlabel("score separation")
    axes.set_title(
        f"Every $\\tau$ in the {decades:.2f}-decade band gives the same tie groups",
        fontsize=10,
    )
    axes.grid(axis="x", alpha=0.3, linewidth=0.5)
    for side in ("left", "right", "top"):
        axes.spines[side].set_visible(False)
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  "
        f"band [2 eta, g_min) = [{2 * eta:.3e}, {g_min:.3e})  |  "
        f"{band['n_positive_gaps']} non-zero gaps, "
        f"{band['n_exact_ties']} exact ties off a log axis",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return True


def fig_rho_discontinuity(record: dict, out: Path) -> bool:
    """G1: rho(tau) is piecewise constant, and the steps are the subject.

    A step plot rather than a line, because joining the samples with straight
    segments would assert intermediate values the function never takes.

    Every riser is at the tau where rho actually jumps, which took getting the
    breakpoints right rather than sampling finely. rho is the largest chain over
    the largest clique, and the two move in different places: a chain is
    single-linkage and changes only at an adjacent gap, while a clique is
    complete-linkage and changes at a span between scores that are not
    neighbours. Of the 129 jumps here only 18 are at an adjacent gap, so a sweep
    over the gaps alone located one in seven of them and a log grid filled the
    rest in at whichever sample came next. ``_rho_sweep`` now evaluates the union
    of the adjacent gaps and the clique thresholds, 215 tau in place of a
    432-point grid, and that set is provably complete: rho cannot move anywhere
    else.

    Falsification: rho = 1 means the largest chain and the largest clique are the
    same size, which does not by itself make the tie groups agree; 7 of the 24
    samples at rho = 1 have differing chain and clique counts. Any step above 1 is
    single-linkage chaining, and its height is how much that tau inflates the
    largest tie group.
    """
    import matplotlib.pyplot as plt

    sweep = record["payload"].get("E3_rho_sweep")
    if not sweep or not sweep["taus"]:
        print("skipping fig_rho_discontinuity (no rho sweep recorded)")
        return False

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
        "$\\rho = 1$: largest chain = largest clique",
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
    _uniform_log_minor(axes.xaxis)
    axes.tick_params(axis="x", which="minor", length=3.0, width=0.6, color=_MUTED)
    axes.set_xlabel("$\\tau$")
    axes.set_ylabel("$\\rho(\\tau)$ = largest chain / largest clique")
    # Counted off the curve rather than read from a breakpoint list. The old title
    # used the adjacent-gap count, which is how often the chain count changes and
    # not how often rho does, so it put 104 over a curve that rose 64 times.
    risers = sum(1 for previous, current in pairwise(rho) if previous != current)
    axes.set_title(
        f"Chain inflation is a step function of $\\tau$ ({risers} jumps, "
        f"each at its exact $\\tau$)",
        fontsize=10,
    )
    axes.grid(alpha=0.3, linewidth=0.5)
    # Where the peak sits relative to the tau in use. The sweep covers the range
    # over which rho varies at all, and this record's tau is seven decades below
    # its lower end, so the peak belongs to a regime none of the results occupy.
    # Left unsaid, the figure reads as though chain inflation threatened them.
    # Extending the axis to reach tau would compress the informative decades into
    # nothing, so it is stated instead of drawn. rho_below_range is recorded by
    # the sweep, so the value is read rather than inferred from the leftmost
    # sample.
    operative = record.get("parameters", {}).get("tau")
    below = sweep.get("rho_below_range")
    if isinstance(operative, int | float) and operative < taus[0] and below is not None:
        note = f"  |  run tau = {operative:.3e} is below the sweep, rho = {below:.0f}"
    else:
        note = ""
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  {len(taus)} exact tau: "
        f"every adjacent gap and clique threshold{note}",
    )
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return True


def fig_rank_cascade(record: dict, out: Path) -> bool:
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
        return False

    ratios = trajectories["ratios"]
    k = trajectories["k"]
    ranks = trajectories["ranks"]

    # A document "crosses" if it changes side of the top-k boundary anywhere.
    #
    # The reference is the recorded unperturbed order, not the first sample. They
    # agree on this run, but taking series[0] would define away the case the
    # figure exists to catch: a document already across the boundary at the
    # smallest perturbation would compare equal to itself and be drawn as grey.
    baseline = trajectories.get("tracked_documents") or []
    inside_at_rest = {
        doc: (baseline.index(doc) + 1 <= k) if doc in baseline else (series[0] <= k)
        for doc, series in ranks.items()
    }
    crossers = [
        doc for doc, series in ranks.items() if any((r <= k) != inside_at_rest[doc] for r in series)
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

    # Grey against coloured is the whole encoding, so it is named rather than left
    # to be inferred. The five hues are not in the legend: each already carries its
    # document id at the end of its own line, and repeating them would be a second
    # key for the same thing.
    handles = [
        plt.Line2D([], [], color=_SERIES[0], linewidth=1.8, label=f"crosses the top-{k} boundary"),
        plt.Line2D([], [], color="#c9c9c6", linewidth=1.0, label="stays on its side"),
    ]
    axes.legend(
        handles=handles,
        fontsize=8,
        loc="lower left",
        framealpha=0.95,
        title="line colour  (labels at right are document ids)",
        title_fontsize=7,
    )

    axes.axvline(1.0, linestyle="--", color=_MUTED, linewidth=1.2)
    axes.axhline(k + 0.5, color=_MUTED, linewidth=0.8)
    # Right-aligned: at the left end the boundary sits among the flat trajectories
    # and the label was drawn over them. Every line has descended well below it by
    # the right-hand end, so the space there is genuinely empty.
    axes.annotate(
        f"top-{k} boundary",
        xy=(ratios[-1], k + 0.5),
        xytext=(-4, -5),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=_MUTED,
    )

    axes.set_xscale("log")
    _uniform_log_minor(axes.xaxis)
    axes.tick_params(axis="x", which="minor", length=3.0, width=0.6, color=_MUTED)
    # Ranks are 1-based and the axis is inverted, so rank 1 sits at the top. The
    # headroom is for the lines themselves: two documents hold rank 1 across the
    # sweep and would otherwise be drawn along the spine. There is no annotation up
    # there to clip, which is what the previous note claimed.
    # Ticked from 1, not 0: ranks are 1-based, so a tick at rank 0 labels a
    # position no document can occupy. The top of the axis sits one rank above the
    # best rank so the lines that hold rank 1 are not drawn along the spine.
    deepest = max(max(s) for s in ranks.values())
    axes.set_ylim(deepest + 2, 0.0)
    axes.set_yticks([1, *range(20, deepest + 1, 20)])
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
    return True


def fig_margins(record: dict, out: Path) -> bool:
    """A1: where the margins actually are, including the exact-tie mass."""
    import matplotlib.pyplot as plt

    dists = record["payload"]["E1_margin_distributions"]
    labels = sorted(dists, key=lambda s: int(s[1:]))
    figure, axes = plt.subplots(figsize=(6.4, 4.0))

    # A log axis has no zero, and p5 is exactly zero wherever the exact-tie share
    # exceeds 5% (k = 1, 10, 20 on the synthetic corpus). Clamping those to a
    # small positive number draws a bar reaching 1e-18, which asserts a margin
    # the data does not contain. The floor is set one decade below the smallest
    # positive percentile instead, and a bar that reaches zero is capped with a
    # marker rather than drawn to a fictitious value.
    positive = [
        v
        for label in labels
        for v in dists[label]["m_k"]["percentiles"].values()
        if isinstance(v, int | float) and v > 0.0
    ]
    floor = min(positive) / 3.0 if positive else 1e-18

    for offset, label in enumerate(labels):
        # E1 reports both margins per k; the boundary margin governs top-k
        # membership, which is this figure's subject.
        d = dists[label]["m_k"]
        percentiles = d["percentiles"]
        hi = max(percentiles["p95"], floor)
        mid = max(percentiles["p50"], floor)
        reaches_zero = percentiles["p5"] <= 0.0
        lo = floor if reaches_zero else percentiles["p5"]
        axes.plot([offset, offset], [lo, hi], color=_INK, linewidth=2.0)
        if reaches_zero:
            axes.plot([offset], [lo], marker="v", color=_INK, markersize=7)
        axes.plot([offset], [mid], marker="o", color=_ACCENT, markersize=5)
        # One row, in axes coordinates. Anchoring each label to its own whisker
        # top put k=1's outside the axes entirely, since an annotation does not
        # extend the data limits and k=1 has both the tallest p95 and the largest
        # exact-tie share: the figure's headline number was the one clipped. A
        # shared baseline also makes the five comparable, which five different
        # heights did not. k=5 is labelled 0% rather than skipped, so a gap in the
        # row cannot be read as "not measured".
        axes.annotate(
            f"{d['share_zero']:.0%} exact",
            xy=(offset, 0.97),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            ha="center",
            va="top",
            color=_ACCENT,
        )

    axes.set_yscale("log")
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels([f"$k$={label[1:]}" for label in labels])
    # Headroom for the label row. The top would otherwise autoscale to k=1's p95
    # and leave its label sitting on the whisker.
    # Whole decades, and minor ticks kept but made recessive.
    #
    # The limits were data-derived (2.78e-5 to 1.14), so the axis began and ended
    # part-way through a decade and the gaps around the outermost labels matched
    # no other gap on the ruler. Snapping to decades fixes that.
    #
    # The minor ticks are a separate matter. Deleting them was tried and was
    # wrong: four of the five medians fall inside the 1e-3 decade, so without
    # marks between the labels this figure loses the comparison it exists to make.
    # Keeping the default 2..9 subdivision was also wrong, for the reason in
    # _uniform_log_minor. Half-decades give an evenly spaced ruler that still
    # resolves those medians.

    ceiling = max(dists[label]["m_k"]["percentiles"]["p95"] for label in labels)
    axes.set_ylim(
        10.0 ** math.floor(math.log10(floor / 2.0)),
        10.0 ** math.ceil(math.log10(ceiling)),
    )
    _uniform_log_minor(axes.yaxis)
    axes.tick_params(axis="y", which="minor", length=3.0, width=0.6, color=_MUTED)
    axes.grid(which="minor", axis="y", alpha=0.12, linewidth=0.4)
    # Half a slot either side. The default 5% margin is narrower than the labels
    # over the first and last column, which then overhang the spines.
    axes.set_xlim(-0.5, len(labels) - 0.5)
    axes.set_ylabel("$m_k$  (p5 to p95, median marked; triangle = reaches 0)")
    # m_min^top stays off this axis: it constrains a disjoint set of gaps, so
    # sharing one axis would invite the reading that either bounds the other. It
    # sits in the JSON alongside.
    axes.set_title("Score-separation margins by rank", fontsize=10)
    axes.grid(alpha=0.3, axis="y", linewidth=0.5)
    _stamp(figure, f"result {record['result_digest'][:16]}")
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return True


def fig_ablation(record: dict, out: Path) -> bool:
    """A2: disagreement caused by the tie-break alone."""
    import matplotlib.pyplot as plt

    rates = record["payload"]["E3_disagreement_rates"]
    pairs = sorted(rates)
    if not pairs:
        return False
    ks = sorted(rates[pairs[0]], key=lambda s: int(s[1:]))
    width = 0.8 / len(pairs)

    # Without an explicit colour this fell through to matplotlib's default cycle,
    # making it the one figure not drawn from the validated palette.
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    for i, pair in enumerate(pairs):
        values = [100.0 * rates[pair][k]["rate"] for k in ks]
        positions = [x + i * width for x in range(len(ks))]
        # 40 queries per cell, so the interval is wide enough to change the
        # reading: the tallest bar here is 42.5% and its interval spans 28 to 58.
        # Without it the chart invites a comparison between bars that the sample
        # cannot support.
        bounds = [
            _wilson(round(rates[pair][k]["rate"] * rates[pair][k]["n"]), rates[pair][k]["n"])
            for k in ks
        ]
        errors = [
            [v - lo for v, (lo, _) in zip(values, bounds, strict=True)],
            [hi - v for v, (_, hi) in zip(values, bounds, strict=True)],
        ]
        axes.bar(
            positions,
            values,
            width=width * 0.92,  # a surface gap between adjacent bars
            label=_PAIR_LABELS.get(pair, pair.replace("_", " ")),
            color=_SERIES[i % len(_SERIES)],
            yerr=errors,
            capsize=2.5,
            error_kw={"ecolor": _MUTED, "elinewidth": 0.9, "capthick": 0.9},
        )

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
    return True


#: Section 7.3's margin bands, ordered by magnitude, as the payload spells them.
#: ``undefined`` is deliberately absent: it records that no margin exists (k >= N),
#: which is a validity state rather than a small margin, so placing it on an
#: ordered magnitude axis would invite reading it as one.
_BANDS = (
    ("exact_tie", "$m_k = 0$"),
    ("(0, tau/100]", "$(0, \\tau/100]$"),
    ("(tau/100, tau/10]", "$(\\tau/100, \\tau/10]$"),
    ("(tau/10, tau]", "$(\\tau/10, \\tau]$"),
    ("(tau, 10*tau]", "$(\\tau, 10\\tau]$"),
    ("(10*tau, 100*tau]", "$(10\\tau, 100\\tau]$"),
    ("(100*tau, inf)", "$(100\\tau, \\infty)$"),
)


def fig_stratified(record: dict, out: Path) -> bool:
    """A2, stratified: where the disagreements actually live.

    ``fig_ablation`` gives one rate per k, which mixes tied and separated queries
    into a single number and so understates the result. Conditioning on the
    margin separates them, and the aggregate follows from this figure rather than
    the other way round.

    Two things are being shown. The first is the rate in each band. The second is
    which bands contain anything at all. Every margin here is either exactly zero
    or above ``100 tau``, which is not a coincidence: a margin is a gap between
    adjacent scores, so the smallest positive one is ``g_min``, and E0 measures
    ``g_min`` at 3.7e-08 against ``100 tau`` of 4.8e-11. Three orders of magnitude
    of headroom is what empties the intervening bands, and it is why tau's position
    inside them cannot move a published number. An empty band is drawn as an
    explicit gap, never as a rate of zero, since "no query was tested here" and
    "every query tested here agreed" are opposite claims.

    Falsification: a non-zero rate anywhere right of ``m_k = 0`` would mean the
    tie-break changed a top-k set that the scores had already separated, which
    contradicts A2 and would have to be a bug in the operator.
    """
    import matplotlib.pyplot as plt

    strata = record["payload"].get("E3_stratified_by_margin")
    if not strata:
        print("skipping fig_stratified (no stratified table recorded)")
        return False

    pairs = sorted(strata)
    ks = sorted({row["k"] for row in strata[pairs[0]]})
    # Band populations are a property of the corpus, not of the operator pair, so
    # one pair's rows describe both panels.
    census = {(row["k"], row["band"]): row["n"] for row in strata[pairs[0]]}

    figure, axes_pair = plt.subplots(1, len(pairs), figsize=(9.6, 4.2), sharey=True, sharex=True)
    axes_list = list(axes_pair) if len(pairs) > 1 else [axes_pair]

    # A band is empty for the corpus, not per operator, so the shading is computed
    # once from any pair and is identical in both panels.
    populated = [any(census.get((k, band), 0) for k in ks) for band, _ in _BANDS]

    runs = _contiguous_runs(populated)

    for axes, pair in zip(axes_list, pairs, strict=True):
        rows = {(r["k"], r["band"]): r for r in strata[pair]}
        for run in runs:
            axes.axvspan(run[0] - 0.5, run[-1] + 0.5, color="#f0efec", zorder=0)
        for position, (band, _) in enumerate(_BANDS):
            if not populated[position]:
                continue
            for series, k in enumerate(ks):
                row = rows.get((k, band))
                if not row or not row["n"]:
                    continue
                # Area proportional to n, so a rate resting on one query cannot
                # look like one resting on forty.
                x = position + (series - (len(ks) - 1) / 2) * 0.19
                # The interval matters more here than anywhere: a cell can hold a
                # single query, where 100% and 21% are the same evidence. Marker
                # area already shows n; the whisker shows what n buys.
                lo, hi = _wilson(row["n_disagree"], row["n"])
                axes.plot([x, x], [lo, hi], color=_MUTED, linewidth=0.9, zorder=2)
                axes.scatter(
                    [x],
                    [100.0 * row["rate"]],
                    s=14.0 + 2.4 * row["n"],
                    color=_SERIES[series % len(_SERIES)],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )

        # Labelled once, on the widest run, so a narrow gap does not carry a
        # sentence longer than itself.
        if runs:
            widest = max(runs, key=len)
            axes.annotate(
                "no query has a margin here",
                xy=((widest[0] + widest[-1]) / 2, 50.0),
                ha="center",
                va="center",
                rotation=90,
                fontsize=8,
                color=_MUTED,
            )
        axes.set_title(_PAIR_LABELS.get(pair, pair.replace("_", " ")), fontsize=10)
        axes.grid(axis="y", alpha=0.3, linewidth=0.5)
        # sharex propagates the tick locations but not their formatting, so the
        # second panel kept the default horizontal labels and they collided.
        axes.set_xticks(range(len(_BANDS)))
        axes.set_xticklabels([label for _, label in _BANDS], rotation=35, ha="right", fontsize=8)

    handles = _series_handles(ks)
    axes_list[0].set_ylim(-6.0, 106.0)
    axes_list[0].set_ylabel("top-$k$ set disagreement (%)")
    axes_list[-1].legend(
        handles=handles,
        fontsize=8,
        title="marker area $\\propto$ queries",
        title_fontsize=7,
        loc="center right",
        framealpha=0.95,
    )

    # Counted from the rows rather than asserted. Writing "0 disagreements" as a
    # literal would keep printing it after the first run that found one, which is
    # the failure this figure exists to make visible.
    separated = [
        row for pair in pairs for row in strata[pair] if row["band"] != "exact_tie" and row["n"]
    ]
    n_separated = sum(row["n"] for row in separated)
    n_disagree = sum(row["n_disagree"] for row in separated)

    # The axis is denominated in tau, and this record's tau is not the band centre
    # fig_tau_band marks, so the value is named rather than left to be assumed.
    tau = record.get("parameters", {}).get("tau")
    tau_note = f"tau = {tau:.3e}  |  " if isinstance(tau, int | float) else ""

    # The title states the result, so it is derived from the result rather than
    # written in. A run that broke A2 would retitle itself.
    figure.suptitle(
        "Disagreement is confined to exact ties"
        if n_disagree == 0
        else f"A2 violated: {n_disagree} disagreements at separated scores",
        fontsize=11,
        y=0.98,
    )
    figure.supxlabel("score-separation margin $m_k$", fontsize=9, y=0.02)
    _stamp(
        figure,
        f"result {record['result_digest'][:16]}  |  {tau_note}"
        f"{n_disagree} of {n_separated} separated query-k observations disagree "
        f"(95% upper bound {_wilson(n_disagree, n_separated)[1]:.1f}%)",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return True


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

    # Counted, not assumed. Three of these decline to draw when their payload is
    # absent, so a hardcoded total reported figures that were never written.
    written = 0
    profile = _load(args.reports / "stability_profile.json")
    if profile:
        written += sum(
            (
                fig_transition(profile, output / "fig_transition.png"),
                fig_tau_band(profile, output / "fig_tau_band.png"),
                fig_rank_cascade(profile, output / "fig_rank_cascade.png"),
                fig_margins(profile, output / "fig_margins.png"),
            )
        )

    ablation = _load(args.reports / "tie_break_ablations.json")
    if ablation:
        written += sum(
            (
                fig_ablation(ablation, output / "fig_ablation.png"),
                fig_rho_discontinuity(ablation, output / "fig_rho_discontinuity.png"),
                fig_stratified(ablation, output / "fig_stratified.png"),
            )
        )

    if not written:
        print("no experiment results found; nothing to plot", file=sys.stderr)
        return 1
    print(f"wrote {written} figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
