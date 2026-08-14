#!/usr/bin/env python3
"""A2 in miniature: a ranking discontinuity with provably zero numerical error.

Research question A2 asks whether deterministic tie-breaking induces
decision-level discontinuities independent of numerical error. Every
corpus-scale measurement of it (section 7.3's disagreement rates, section 7.4's
case study) is a rate over thousands of queries, and a rate always invites the
objection that some of it is arithmetic noise leaking in. This removes the
objection once, small enough to read in full.

Six documents, three of which carry one bit pattern between them. Under that
exact tie the three ranking operators of section 4.5 produce three different
answers, and the file checks that all three consumed byte-identical scores. No
arithmetic runs between the operators, so numerical error here is zero and the
disagreement has one available cause.

Two kinds of discontinuity appear, and they are separate phenomena. At k = 3 the
tie straddles the boundary and the three operators return three different top-k
sets; at k = 4 the boundary margin is 0.25 and all three return the same set in
three different orders. The note under section 4.4 in ``docs/spec_addenda.md``
gives the reason neither bounds the other: ``m_k`` constrains the boundary,
``m_min^top`` the interior, and the gap index sets are disjoint.

The second half turns to G1. Section 2.3.3's tie group is a ball and the paper
calls its members "indistinguishable", but the relation is intransitive, so the
implementation carries three separately named objects. The dyadic ladder
``s_i = i * 2^-20`` at ``tau = 2^-20`` is the standard witness: every score, gap
and comparison in it is exact in binary64, so nothing observed can be blamed on
rounding.

Run with::

    python examples/tie_break_discontinuity_demo.py
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.ranking.attributes import AttributeTable  # noqa: E402
from tfidf_stability.ranking.margins import (  # noqa: E402
    adjacent_gaps,
    boundary_margin,
    min_adjacent_margin_top,
)
from tfidf_stability.ranking.ranker import Ranking, rank_all_operators  # noqa: E402
from tfidf_stability.ranking.tie_groups import TieGroupIndex, tie_ball_interval  # noqa: E402
from tfidf_stability.utils.numerics import bits_of, same_bits  # noqa: E402

WIDTH: Final[int] = 78

#: ``(doc_id, score, popularity, rating_sum2, rating_count, engagement)``.
#:
#: Dyadic scores, so each is exact in binary64 and the three at 0.5 are one bit
#: pattern rather than three nearby ones. Written down rather than scored from
#: text because of G22: ``tf = count / L`` makes the finest text edit a ``1/L``
#: relative perturbation, so no corpus can be authored onto a chosen separation.
#:
#: Each operator orders the three tied documents differently: popularity puts
#: charlie first (pi), identifier byte order alpha first (pi_score), engagement
#: charlie first and bravo second (pi_alt). Rating never fires; it sits second in
#: both priorities and the leading attribute has already decided.
CORPUS: Final[tuple[tuple[str, float, int, int, int, int], ...]] = (
    ("doc-zulu", 0.75, 2, 12, 2, 2),
    ("doc-alpha", 0.5, 5, 18, 2, 1),
    ("doc-bravo", 0.5, 1, 14, 2, 5),
    ("doc-charlie", 0.5, 10, 16, 2, 9),
    ("doc-echo", 0.25, 4, 16, 2, 4),
    ("doc-foxtrot", 0.125, 6, 10, 2, 6),
)

#: G1's witness: ``s_i = i * 2^-20`` for ``i = 0 .. 5``, with ``tau = 2^-20``
#: exactly. Six rungs is the shortest ladder on which the chain is visibly wider
#: than any clique while the whole thing still prints.
LADDER_EXPONENT: Final[int] = -20
LADDER_RUNGS: Final[int] = 6


def _heading(text: str) -> None:
    print(f"\n{'=' * WIDTH}\n{text}\n{'=' * WIDTH}")


def _rule(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def _hex_bits(x: float) -> str:
    """The binary64 payload as a hex word.

    "Bit-identical" is a claim about these sixteen digits, and two floats that
    print the same at repr precision can still differ in the last place.
    """
    return f"0x{int.from_bytes(bits_of(x), 'little'):016x}"


def _records() -> list[dict[str, Any]]:
    """The corpus in the record shape :class:`AttributeTable` reads."""
    return [
        {
            "doc_id": doc_id,
            "popularity": popularity,
            "rating_sum2": rating_sum2,
            "rating_count": rating_count,
            "engagement": engagement,
        }
        for doc_id, _, popularity, rating_sum2, rating_count, engagement in CORPUS
    ]


def _require_bit_identical_scores(rankings: dict[str, Ranking]) -> int:
    """Verify A2's premise and return the number of comparisons that verified it.

    The premise is that the operators differ in their tie-break and in nothing
    else. It holds structurally (``rank_all_operators`` builds one sorted score
    array and hands the same object to each operator), but a refactor can remove
    that silently, so it is checked against the bit patterns themselves.

    Raises:
        AssertionError: If any operator saw a score differing from ``pi``'s in a
            single bit, at which point every disagreement printed by this file
            would be uninterpretable and the run must stop.
    """
    reference = rankings["pi"]
    checked = 0
    for name, ranking in rankings.items():
        for label, mine, theirs in (
            ("raw", ranking.scores, reference.scores),
            ("sorted", ranking.sorted_scores, reference.sorted_scores),
        ):
            for i, (a, b) in enumerate(zip(mine, theirs, strict=True)):
                if not same_bits(a, b):
                    raise AssertionError(
                        f"{name} saw a different {label} score at position {i}: "
                        f"{_hex_bits(a)} vs pi's {_hex_bits(b)}. A2's premise is false."
                    )
                checked += 1
    return checked


def demonstrate_tie_break_discontinuity() -> dict[str, Ranking]:
    """The exact tie, the three orderings, and the two kinds of disagreement."""
    _heading("A2 -- a decision-level discontinuity with zero numerical error")

    doc_ids = [row[0] for row in CORPUS]
    scores = [row[1] for row in CORPUS]
    table = AttributeTable.from_records(_records())
    rankings = rank_all_operators(scores, table)

    _rule("The corpus")
    print(
        f"  {'doc_id':<13} {'score':>7}  {'bit pattern':<18}  {'pop':>4} {'rating':>7} {'eng':>4}"
    )
    for doc_id, score, popularity, sum2, count, engagement in CORPUS:
        # G8 stores the mean exactly as (2 * sum of ratings, count); divide only
        # to print it, never to compare it.
        rating = sum2 / (2 * count)
        print(
            f"  {doc_id:<13} {score:>7}  {_hex_bits(score):<18}  {popularity:>4} "
            f"{rating:>7.1f} {engagement:>4}"
        )

    _rule("The tie is exact, not close")
    tied = [(d, s) for d, s in zip(doc_ids, scores, strict=True) if s == 0.5]
    print(f"  {len(tied)} documents share the single bit pattern {_hex_bits(0.5)}:")
    print(f"    {', '.join(d for d, _ in tied)}")
    same = all(same_bits(s, tied[0][1]) for _, s in tied)
    print(f"  same_bits across all {len(tied)} of them: {same}")

    sorted_scores = rankings["pi"].sorted_scores
    print(f"  adjacent gaps, best first: {list(adjacent_gaps(sorted_scores))}")
    for k in (3, 4):
        m_k = boundary_margin(sorted_scores, k)
        m_top = min_adjacent_margin_top(sorted_scores, k)
        print(
            f"  k={k}: m_k = {m_k.value!r} (exact tie: {m_k.is_exact_tie}), "
            f"flip radius {m_k.flip_radius!r}, m_min^top = {m_top.value!r}"
        )
    print("  At k=3 the boundary sits inside the tie, so top-3 membership is decided")
    print("  entirely by the tie-break. At k=4 the boundary is clear (0.25) but the")
    print("  interior minimum is still 0, so the *order* is decided by the tie-break.")

    _rule("Premise: every operator consumed byte-identical scores")
    checked = _require_bit_identical_scores(rankings)
    shared = all(r.sorted_scores is rankings["pi"].sorted_scores for r in rankings.values())
    print(f"  same_bits comparisons against pi, all passing: {checked}")
    print(f"  all three hold the identical sorted-score object: {shared}")
    print("  There is no arithmetic between the operators, so nothing below can be")
    print("  attributed to rounding, reduction order, or platform libm.")

    _rule("The three orderings")
    for name in sorted(rankings):
        print(f"  {name:<9} {' > '.join(doc_ids[i] for i in rankings[name].order)}")

    _rule("Discontinuity 1 (k=3): the top-k *set* differs")
    sets = {name: {doc_ids[i] for i in r.top_k(3)} for name, r in rankings.items()}
    for name in sorted(sets):
        print(f"  {name:<9} {{{', '.join(sorted(sets[name]))}}}")
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            difference = sorted(sets[a] ^ sets[b])
            print(f"  {a} vs {b}: sets differ -- symmetric difference {{{', '.join(difference)}}}")

    _rule("Discontinuity 2 (k=4): identical sets, three different orders")
    orders = {name: tuple(doc_ids[i] for i in r.top_k(4)) for name, r in rankings.items()}
    print(
        f"  set is identical across operators: {len({frozenset(o) for o in orders.values()}) == 1}"
    )
    for name in sorted(orders):
        print(f"  {name:<9} {' > '.join(orders[name])}")
    print(f"  distinct orders: {len(set(orders.values()))} of {len(orders)}")

    return rankings


def demonstrate_tie_group_objects() -> None:
    """G1's ball, chain and clique on the dyadic ladder, where all three differ."""
    _heading("G1 -- ball, chain and clique are three different objects")

    unit = math.ldexp(1.0, LADDER_EXPONENT)
    # Descending, as every tie-group function requires: s[rank] = (rungs-1-rank)*u.
    ladder = tuple(math.ldexp(float(i), LADDER_EXPONENT) for i in reversed(range(LADDER_RUNGS)))
    tau = unit
    rank_of = {i: LADDER_RUNGS - 1 - i for i in range(LADDER_RUNGS)}

    _rule(f"The ladder: s_i = i * 2^{LADDER_EXPONENT}, tau = 2^{LADDER_EXPONENT}")
    print(f"  u = 2^{LADDER_EXPONENT} = {unit!r}")
    print(f"  scores, best first: {', '.join(f's_{i}' for i in reversed(range(LADDER_RUNGS)))}")
    gaps = adjacent_gaps(ladder)
    print(f"  every adjacent gap is exactly u: {all(g == unit for g in gaps)}")
    print(f"  gaps in units of u: {[g / unit for g in gaps]}")

    _rule("The relation |s_i - s_j| <= tau is not transitive")
    for i, j in ((0, 1), (1, 2), (0, 2)):
        difference = ladder[rank_of[j]] - ladder[rank_of[i]]
        verdict = "related" if difference <= tau else "NOT related"
        print(f"  |s_{j} - s_{i}| = {difference / unit:.0f}u <= tau ? {verdict}")
    print("  s_0 ~ s_1 and s_1 ~ s_2 but not s_0 ~ s_2, so this is no equivalence relation")
    print("  and 'the tie group of document i' does not name anything.")

    _rule("Balls (section 2.3.3, verbatim) -- they overlap and disagree")
    for centre in (1, 0):
        lo, hi = tie_ball_interval(ladder, rank_of[centre], tau)
        members = sorted(LADDER_RUNGS - 1 - r for r in range(lo, hi))
        print(f"  ball(s_{centre}) = {{{', '.join(f's_{i}' for i in members)}}}")
    lo1, hi1 = tie_ball_interval(ladder, rank_of[1], tau)
    lo0, hi0 = tie_ball_interval(ladder, rank_of[0], tau)
    print(f"  s_2 in ball(s_1): {lo1 <= rank_of[2] < hi1}")
    print(f"  s_2 in ball(s_0): {lo0 <= rank_of[2] < hi0}")
    print("  Same tau, same scores, different centre, different answer -- which is why")
    print("  the ball is reported per-rank and never as 'the' group of a document.")

    # The constructor emits G1's diagnostics; captured so the warning prints in
    # sequence instead of landing on stderr out of order with the narrative.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = TieGroupIndex.build(ladder, tau)

    _rule("Chains (single linkage) -- a partition, and it swallows the ladder")
    for lo, hi in index.chains:
        members = sorted(LADDER_RUNGS - 1 - r for r in range(lo, hi))
        print(f"  {{{', '.join(f's_{i}' for i in members)}}}")
    print(f"  {index.n_chains} chain(s) covering all {LADDER_RUNGS} rungs: the endpoints")
    print(
        f"  s_0 and s_{LADDER_RUNGS - 1} are {LADDER_RUNGS - 1}u apart, which is "
        f"{LADDER_RUNGS - 1}x tau."
    )

    _rule("Cliques (complete linkage) -- only adjacent pairs are mutually within tau")
    for lo, hi in index.cliques:
        members = sorted(LADDER_RUNGS - 1 - r for r in range(lo, hi))
        print(f"  {{{', '.join(f's_{i}' for i in members)}}}")
    print(f"  {len(index.cliques)} clique(s), largest {index.largest_clique}")

    _rule("The diagnostic")
    print(
        f"  rho = largest chain / largest clique = {index.largest_chain} / "
        f"{index.largest_clique} = {index.rho:g}"
    )
    for warning in caught:
        print(f"  {warning.category.__name__} fired")
    print("  rho far above 1 says the reported tie group is held together by a chain of")
    print("  small steps, not by indistinguishability -- G1's reason for reporting all")
    print("  three objects instead of conflating them under one name.")


def main() -> int:
    rankings = demonstrate_tie_break_discontinuity()
    demonstrate_tie_group_objects()

    _heading("What this shows")
    n_distinct = len({r.order for r in rankings.values()})
    print(f"  {n_distinct} distinct total orders from one bit-identical score vector.")
    print("  A2 does not need numerical error to produce a decision-level discontinuity;")
    print("  an exact tie plus a deterministic tie-break is sufficient. Conversely, a")
    print("  reported disagreement rate is only evidence about the tie-break if the")
    print("  operators provably saw the same scores -- which is why that check runs")
    print("  here, and in scripts/run_tie_break_ablations.py, before any rate is quoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
