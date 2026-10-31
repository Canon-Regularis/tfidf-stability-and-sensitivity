#!/usr/bin/env python3
"""Render the committed experiment reports as one Markdown document.

Every number is read from a report under ``reports/`` and none is recomputed,
so the summary cannot disagree with the digests it quotes. Nothing outside the
repository is read, so ``--check`` compares the output byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
# `tooling` sits at the repository root and the package under `src`. Python
# puts only `scripts/` on the path, so both roots are added here.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.preprocessing.ngrams import JOINER  # noqa: E402
from tooling.gate import report  # noqa: E402

#: The reports this renders, in the order the paper reaches them. A missing one
#: is skipped rather than fatal: `export_intermediates.py` is optional, and a
#: checkout that has not run it should still get a summary of what it does have.
_REPORTS = (
    ("similarity", "similarity.json"),
    ("stability_profile", "stability_profile.json"),
    ("tie_break_ablations", "tie_break_ablations.json"),
    ("intermediates", "intermediates_d000000.json"),
)

#: Which figure is built from which report, so the table below can pair each one
#: with the digest it is stamped with. `make_figures.py` owns the same mapping in
#: its `main`; this is the reader-facing half of it.
_FIGURES = {
    "stability_profile": ("fig_transition", "fig_tau_band", "fig_rank_cascade", "fig_margins"),
    "tie_break_ablations": ("fig_ablation", "fig_rho_discontinuity", "fig_stratified"),
}


def _ks(block: dict[str, Any]) -> list[str]:
    """Return the ``k`` keys of a block in numeric order.

    ``canonical_json`` sorts keys as strings, so a k-block on disk reads
    ``k1, k10, k20, k5, k50``. The digest rests on that sort, so the numeric
    order is restored here rather than in the file.
    """
    return sorted(block, key=lambda s: int(s[1:]))


def _num(value: Any, places: int = 4) -> str:
    """Format a number as the report states it.

    Magnitudes below a millionth or above a million use exponential form,
    where a decimal expansion is a run of zeros; others use a plain decimal.
    ``None`` is the JSON spelling of a non-finite value and renders as ``n/a``.
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if value == 0.0:
        return "0"
    return f"{value:.{places}e}" if abs(value) < 1e-6 or abs(value) >= 1e6 else f"{value:.6g}"


def _term(token: str) -> str:
    """Render a feature so that each token of an n-gram is visible.

    The n-gram joiner is U+001F, which cannot occur inside a token and has no
    visible glyph, so it is replaced by a visible separator.
    """
    return "`" + token.replace(JOINER, "` + `") + "`"


def _pct(value: Any, places: int = 2) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{places}f}%"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Build a Markdown table with columns padded to a common width.

    A column no row fills is dropped, so an always-empty cell cannot read as a
    value that is missing.
    """
    keep = [i for i in range(len(header)) if header[i] or any(r[i] for r in rows)]
    header = [header[i] for i in keep]
    rows = [[r[i] for i in keep] for r in rows]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |")
    return out


def _provenance(loaded: dict[str, dict[str, Any]]) -> list[str]:
    """One row per report: what produced it and what identifies it."""
    rows = []
    for name, record in loaded.items():
        parameters = record.get("parameters", {})
        rows.append(
            [
                f"`{name}`",
                str(record.get("experiment", "-")),
                str(parameters.get("dataset", "-")),
                f"`{str(parameters.get('model_digest', '-'))[:12]}`",
                f"`{record['result_digest'][:16]}`",
            ]
        )
    return [
        "## Provenance",
        "",
        "Every number below is read from one of these files. The result digest is",
        "taken over the payload and parameters with volatile fields stripped, so a",
        "rerun on the same data reproduces it.",
        "",
        *_table(["report", "experiment", "dataset", "model", "result digest"], rows),
        "",
    ]


def _similarity(record: dict[str, Any]) -> list[str]:
    """Render the similarity section: top document and margin per query."""
    queries = record["payload"]["queries"]
    rows = []
    for query in queries:
        top = query["top"][0] if query["top"] else None
        margin = query["margins"].get("k1", {})
        rows.append(
            [
                f"`{query['query_id']}`",
                f"`{top['doc_id']}`" if top else "-",
                _num(top["score"]) if top else "-",
                _num(margin.get("value")),
                "exact tie" if margin.get("is_exact_tie") else "",
            ]
        )
    ties = sum(1 for q in queries if q["margins"].get("k1", {}).get("is_exact_tie"))
    return [
        "## Similarity",
        "",
        f"{len(queries)} queries scored against the corpus. {ties} of them are decided",
        "by the tie-break rather than by the scores: their top two documents carry the",
        "same value, so `m_1` is exactly zero.",
        "",
        *_table(["query", "top document", "score", "m_1", "note"], rows),
        "",
    ]


def _stability(record: dict[str, Any]) -> list[str]:
    """Render E0, E1 and E2: the tau band, the margins and the certificate."""
    payload = record["payload"]
    floor = payload["E0_tau_derivation"]["noise_floor"]
    band = payload["E0_tau_derivation"]["band"]
    out = [
        "## Stability profile",
        "",
        "### E0 - the admissible band for tau",
        "",
        "`tau` has to sit above the arithmetic noise floor and below the smallest",
        "non-zero gap. A band that was empty would mean no threshold separates",
        "rounding error from tie structure on this corpus.",
        "",
        *_table(
            ["quantity", "value", "as written"],
            [
                ["eta (noise floor)", _num(floor["eta"]), f"`{floor['eta_hex']}`"],
                ["tau_floor = 2 eta", _num(band["tau_floor"]), f"`{band['tau_floor_hex']}`"],
                ["g_min (smallest non-zero gap)", _num(band["g_min"]), f"`{band['g_min_hex']}`"],
                ["band width", f"{band['decades']:.2f} decades", ""],
                ["gaps inside the band", _num(band["n_gaps_in_band"]), ""],
                ["non-zero gaps", _num(band["n_positive_gaps"]), ""],
                ["exact ties", _num(band["n_exact_ties"]), ""],
                ["band is valid", _num(band["is_valid"]), ""],
                ["tie groups invariant across it", _num(band["is_invariant"]), ""],
            ],
        ),
        "",
        "Per reduction policy, over "
        f"{_num(floor['n_documents'])} documents and {_num(floor['n_queries'])} queries:",
        "",
    ]

    rows = [
        [
            f"`{policy['policy']}`",
            _pct(policy["share_differing"]),
            _num(policy["max_abs"]),
            _num(policy["max_ulps"]),
        ]
        for policy in floor["per_policy"]
    ]
    out += _table(["policy", "scores differing", "max abs difference", "max ulps"], rows)

    out += ["", "### E1 - where the margins are", ""]
    dists = payload["E1_margin_distributions"]
    rows = []
    for key in _ks(dists):
        m_k = dists[key]["m_k"]
        rows.append(
            [
                key[1:],
                _num(m_k["n"]),
                _pct(m_k["share_zero"]),
                _num(m_k["percentiles"]["p5"]),
                _num(m_k["percentiles"]["p50"]),
                _num(m_k["percentiles"]["p95"]),
            ]
        )
    out += [
        "`m_k` is the score separation at the top-k boundary. The exact-tie share is",
        "the headline: a percentile summary hides it, because past 50% it only makes",
        "several percentiles read zero.",
        "",
        *_table(["k", "n", "exactly zero", "p5", "p50", "p95"], rows),
    ]

    transition = payload["E2_transition"]
    audit = transition["certificate_audit"]
    out += [
        "",
        "### E2 - the transition, and the certificate",
        "",
        f"At k = {transition['k']}, over {transition['n_queries_used']} queries "
        f"({transition['n_queries_excluded_exact_tie']} excluded for an exact tie at the",
        "boundary). Section 4.4 certifies that no perturbation smaller than `m_k / 2`",
        "can change the top-k set, so every row at a ratio below 1.0 must read zero.",
        "",
        *_table(
            ["eps / (m_k / 2)", "flip rate", "flips", "trials", "certified"],
            [
                [
                    f"{point['ratio']:.2f}",
                    _pct(point["flip_rate"]),
                    _num(point["n_flips"]),
                    _num(point["n_trials"]),
                    "yes" if point["within_certificate"] else "",
                ]
                for point in transition["points"]
            ],
        ),
        "",
        f"Recorded certificate violations: **{transition['violations']}**.",
        "",
        *_table(
            ["audit", "value"],
            [
                ["certified perturbations", _num(audit["n_certified"])],
                ["certified and changed", _num(audit["certified_changed"])],
                ["uncertified and unchanged", _num(audit["uncertified_unchanged"])],
                ["uncertified and changed", _num(audit["uncertified_changed"])],
                ["conservatism", _pct(audit["conservatism"])],
                ["sound", _num(audit["is_sound"])],
                ["conclusive", _num(audit["is_conclusive"])],
            ],
        ),
        "",
    ]
    return out


def _ablations(record: dict[str, Any]) -> list[str]:
    """Render E3 and E4: tie-break disagreement rates and one pair in detail."""
    payload = record["payload"]
    rates = payload["E3_disagreement_rates"]
    pairs = sorted(rates)
    ks = _ks(rates[pairs[0]]) if pairs else []

    out = [
        "## Tie-break ablations",
        "",
        "### E3 - disagreement caused by the tie-break alone",
        "",
        "The three operators consume one shared score array, so a disagreement here",
        "cannot come from the arithmetic. It comes from how ties are broken.",
        "",
        *_table(
            ["operators", *[f"k={k[1:]}" for k in ks], "n"],
            [
                [
                    f"`{pair}`",
                    *[_pct(rates[pair][k]["rate"]) for k in ks],
                    _num(rates[pair][ks[0]]["n"]) if ks else "-",
                ]
                for pair in pairs
            ],
        ),
        "",
    ]

    fks = payload["E3_fks_distance"]
    out += [
        "Kendall's tau with the Fagin-Kumar-Sivakumar penalty, over "
        f"{_num(fks['n'])} query-k observations:",
        "",
        *_table(
            ["statistic", "value"],
            [
                ["mean", _num(fks["mean"])],
                ["zero (the rankings agree)", f"{_num(fks['n_zero'])} ({_pct(fks['share_zero'])})"],
                ["maximum", _num(fks["maximum"])],
            ],
        ),
        "",
        "### E4 - the closest pair on the corpus",
        "",
    ]

    case = payload["E4_case_study"]
    pair = case["pair"]
    out += [
        *_table(
            ["quantity", "value"],
            [
                ["documents", f"`{pair['doc_A']}` and `{pair['doc_B']}`"],
                ["score of A", f"{_num(pair['s_A'])}  `{pair['s_A_hex']}`"],
                ["score of B", f"{_num(pair['s_B'])}  `{pair['s_B_hex']}`"],
                ["separation m_k", _num(pair["m_k"])],
                ["an exact tie", _num(pair["is_exact_tie"])],
                ["tau in force", _num(case["tau"])],
                ["largest chain / largest clique", _num(case["rho_chain_inflation"])],
                ["largest chain", _num(case["largest_chain"])],
                ["largest clique", _num(case["largest_clique"])],
            ],
        ),
        "",
        "Top 10 under each operator:",
        "",
    ]
    for operator, order in sorted(case["top_10_by_operator"].items()):
        out.append(f"- `{operator}`: {', '.join(str(d) for d in order)}")
    out.append("")
    return out


def _intermediates(record: dict[str, Any]) -> list[str]:
    """Render every quantity of one document with its hex beside the decimal."""
    payload = record["payload"]
    terms = payload["terms"]
    return [
        "## Intermediates",
        "",
        f"Every quantity behind one document, `{payload['doc_id']}`. The hex is what",
        "makes this evidence: a decimal rendering of a binary64 is lossy at whatever",
        "precision the formatter chose, so two values one ulp apart can print",
        "identically.",
        "",
        *_table(
            ["quantity", "value"],
            [
                ["features before the vocabulary", _num(payload["n_features"])],
                ["terms in the vocabulary", _num(payload["n_terms_in_vocabulary"])],
                ["in-vocabulary length L", _num(payload["in_vocabulary_length"])],
                ["norm", f"{_num(payload['norm'])}  `{payload['norm_hex']}`"],
                ["zero norm", _num(payload["is_zero_norm"])],
            ],
        ),
        "",
        "The ten heaviest terms:",
        "",
        *_table(
            ["feature", "count", "df", "tf", "idf", "weight"],
            [
                [
                    _term(t["term"]),
                    _num(t["count"]),
                    _num(t["df"]),
                    _num(t["tf"]),
                    _num(t["idf"]),
                    _num(t["weight"]),
                ]
                for t in terms[:10]
            ],
        ),
        "",
    ]


def _figures(loaded: dict[str, dict[str, Any]]) -> list[str]:
    """Pair each figure with the digest it is stamped with.

    The stamp ties an image to the run behind it, so a figure whose digest is
    absent from this table comes from a report that is not in the tree.
    """
    rows = []
    for source, names in _FIGURES.items():
        record = loaded.get(source)
        if record is None:
            continue
        for name in names:
            path = REPO / "reports" / "figures" / f"{name}.png"
            rows.append(
                [
                    f"`{name}.png`",
                    f"`{source}`",
                    f"`{record['result_digest'][:16]}`",
                    "" if path.exists() else "not rendered",
                ]
            )
    return [
        "## Figures",
        "",
        *_table(["figure", "built from", "stamped digest", ""], rows),
        "",
    ]


_SECTIONS = {
    "similarity": _similarity,
    "stability_profile": _stability,
    "tie_break_ablations": _ablations,
    "intermediates": _intermediates,
}


def render(reports: Path) -> str:
    """Render the whole document from whichever reports are present."""
    loaded: dict[str, dict[str, Any]] = {}
    for name, filename in _REPORTS:
        path = reports / filename
        if path.exists():
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))

    lines = [
        "# Experiment summary",
        "",
        "Generated by `scripts/render_summary.py` from the committed reports. Do not",
        "edit: run the script. Every number here is read from a report rather than",
        "recomputed, so this cannot disagree with the digests it quotes.",
        "",
    ]
    if not loaded:
        lines += ["No reports found.", ""]
        return "\n".join(lines) + "\n"

    lines += _provenance(loaded)
    for name, section in _SECTIONS.items():
        if name in loaded:
            lines += section(loaded[name])
    lines += _figures(loaded)
    # Every section ends with a blank line; on the last one that leaves a blank
    # line at end of file, which `end-of-file-fixer` strips. Trimming to one
    # trailing newline keeps the render identical to the committed file.
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=REPO / "reports")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="fail if the summary is out of date")
    args = parser.parse_args()

    destination = args.output or args.reports / "summary.md"
    rendered = render(args.reports)

    if args.check:
        current = destination.read_text(encoding="utf-8") if destination.exists() else ""
        if current == rendered:
            print(f"{destination.name} is current")
            return 0
        return report(
            [f"{destination.name} does not match the reports; run scripts/render_summary.py"],
            "summary check",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Through the same writer the reports use, so the summary carries LF endings
    # on every platform and cannot differ from a checkout by its line endings.
    from tfidf_stability.utils.io import atomic_write_text

    atomic_write_text(destination, rendered)
    print(f"written {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
