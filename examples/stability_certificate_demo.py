#!/usr/bin/env python3
"""Section 4.4's certificate, and the two ways it is misread (A1).

Section 4.4 states one implication: ``|ds_i| <= eps`` for every document and
``eps < m_k / 2`` together guarantee the top-k set is invariant. Being merely
sufficient, it gets read as two stronger claims, in opposite directions by
different readers, from the same sentence.

Read as a prediction, "not certified" becomes "will break" and the certificate
turns into a false alarm: perturbations of many times the radius usually leave
the top-k where it was, since a flip needs a direction that scattered noise
almost never takes. Read as a safety margin, "sufficient" becomes "conservative"
and someone shaves the radius; but ``m_k / 2`` is exact, and the perturbation
that breaks it at ``m_k / 2 + delta`` can be written down.

Section 7.2 reports certified radii as "empirical certificates of stability",
and the transition curve in ``analysis/stability_profile.py`` only becomes
interpretable once worst case and average case have been separated. So all three
cases are exhibited on one hand-built ranking:

    1. below the radius, in the worst possible direction: unchanged, provably;
    2. four times the radius, scattered: unchanged anyway, so the converse of
       the theorem is false;
    3. a hair above the radius, adversarially aimed: flipped, so the bound is
       tight.

Every score, perturbation and intermediate value is an integer multiple of
2^-20 with magnitude below 2, so each occupies at most 21 significant bits of a
53-bit binary64 significand. Nothing here rounds; a demonstration about the
decision boundary carrying its own floating-point story leaves a reader unable to
tell the two effects apart. Same dyadic construction as
``tests/test_margins_and_flip_radii.py``, and the terms in which
``docs/spec_addenda.md`` states its tightness note.

Run::

    python examples/stability_certificate_demo.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.perturbation.score_bounds import (  # noqa: E402
    certified_radius,
    flip_witness,
    is_top_k_stable,
)
from tfidf_stability.ranking.attributes import AttributeSpec, AttributeTable  # noqa: E402
from tfidf_stability.ranking.margins import boundary_margin  # noqa: E402
from tfidf_stability.ranking.ranker import rank, sorted_scores_desc  # noqa: E402
from tfidf_stability.ranking.sort_keys import SortKeySpec  # noqa: E402
from tfidf_stability.utils.numerics import same_bits  # noqa: E402

#: Six documents with dyadic scores. Hand-built rather than fitted: a real
#: TF-IDF score is a quotient of sums and would drag its own rounding into a
#: demonstration about the decision boundary.
DOC_IDS: tuple[str, ...] = ("d0", "d1", "d2", "d3", "d4", "d5")
SCORES: tuple[float, ...] = (0.75, 0.25, 1.0, 0.5, 0.125, 0.0)

#: The only tie-break attribute, so the closing exact-tie case has a single
#: visible cause. d1 sits just outside the top-k and is the popular one, so it
#: wins any tie it is given.
POPULARITY: tuple[int, ...] = (50, 90, 40, 10, 30, 20)

K = 3
POP_ONLY = SortKeySpec("popularity_only", ("popularity",))

#: Values are *checked* against this grid rather than trusted to be exact.
_SCALE = 2**20

_RULE = "=" * 74
_THIN = "-" * 74


def _require(condition: bool, message: str) -> None:
    """Fail loudly rather than print a claim the run did not establish.

    ``assert`` would vanish under ``python -O``, and this file exists to be run.
    """
    if not condition:
        raise AssertionError(message)


def _is_exact(values: Iterable[float]) -> bool:
    """Whether every value sits on the 2^-20 grid below 2, hence needs <= 21 bits."""
    return all(abs(v) < 2.0 and (v * _SCALE).is_integer() for v in values)


def _attribute_table(popularity: Sequence[int]) -> AttributeTable:
    return AttributeTable.from_records(
        [{"doc_id": DOC_IDS[i], "popularity": p} for i, p in enumerate(popularity)],
        (AttributeSpec("popularity"),),
    )


def _order(scores: Sequence[float], table: AttributeTable) -> tuple[int, ...]:
    return rank(scores, table, POP_ONLY).order


def _names(docs: Iterable[int]) -> str:
    return " ".join(DOC_IDS[d] for d in docs)


def _fmt_set(members: Iterable[int]) -> str:
    return "{" + ", ".join(DOC_IDS[i] for i in sorted(members)) + "}"


def _report(
    title: str,
    claim: str,
    perturbed: Sequence[float],
    table: AttributeTable,
    base_order: tuple[int, ...],
) -> frozenset[int]:
    """Print one perturbation in full and return the resulting top-k set."""
    deltas = [p - s for s, p in zip(SCORES, perturbed, strict=True)]
    _require(_is_exact(deltas) and _is_exact(perturbed), f"{title}: left the dyadic grid")

    # Section 4.4 bounds the movement that happened; the eps someone intended to
    # apply never enters.
    eps = max(abs(d) for d in deltas)
    radius = certified_radius(sorted_scores_desc(SCORES), K).set_radius
    certified = is_top_k_stable(sorted_scores_desc(SCORES), K, eps)

    new_order = _order(perturbed, table)
    new_top = frozenset(new_order[:K])
    base_top = frozenset(base_order[:K])
    new_rank = {doc: j + 1 for j, doc in enumerate(new_order)}

    print(f"\n{title}\n  {claim}\n")
    print(f"  {'doc':<5}{'rank':>5}{'score':>14}{'delta':>15}{'after':>14}{'rank':>7}")
    print("  " + "-" * 60)
    for j, doc in enumerate(base_order):
        crossed = " <-- crossed the boundary" if (doc in new_top) != (doc in base_top) else ""
        print(
            f"  {DOC_IDS[doc]:<5}{j + 1:>5}{SCORES[doc]:>14.9f}{deltas[doc]:>+15.9f}"
            f"{perturbed[doc]:>14.9f}{new_rank[doc]:>7}{crossed}"
        )

    print(f"\n  {'largest |ds_i| applied':<32}{eps:.9f}   ({eps / radius:g} x m_k/2)")
    print(f"  {'section 4.4 certifies the set':<32}{'yes' if certified else 'NO'}")
    print(f"  {f'top-{K} set after':<32}{_fmt_set(new_top)}")
    return new_top


def _show_corpus(base_order: tuple[int, ...]) -> None:
    print(_RULE)
    print(f"Section 4.4 stability certificate -- {len(SCORES)} documents, k = {K}")
    print(_RULE)
    print("\n  scores, in rank order under pi restricted to popularity:\n")
    print(f"  {'doc':<5}{'rank':>5}{'score':>14}{'popularity':>13}")
    print("  " + "-" * 37)
    for j, doc in enumerate(base_order):
        print(f"  {DOC_IDS[doc]:<5}{j + 1:>5}{SCORES[doc]:>14.9f}{POPULARITY[doc]:>13}")
    print(f"\n  ranking   {_names(base_order)}")
    print(f"  top-{K} set  {_fmt_set(base_order[:K])}")


def _show_certificate(base_order: tuple[int, ...]) -> float:
    """Print the certificate at ``K`` and across every k, and return its radius."""
    sorted_scores = sorted_scores_desc(SCORES)
    margin = boundary_margin(sorted_scores, K)
    radius = certified_radius(sorted_scores, K).set_radius
    _require(margin.value == 0.25 and radius == 0.125, "the worked margin is not as printed")
    _require(radius * 2.0 == margin.value, "halving the margin was not exact")

    print(f"\n{_THIN}\nThe certificate\n{_THIN}\n")
    print(
        f"  m_k = score(r_{K}) - score(r_{K + 1}) = {SCORES[base_order[K - 1]]:.9f} - "
        f"{SCORES[base_order[K]]:.9f} = {margin.value:.9f}"
    )
    print(f"  eps_k^flip = m_k / 2 = {radius:.9f}")
    print("  halving is exact -- 2 * eps_k^flip recovers m_k bit-for-bit.")

    print("\n  the certificate at every k, both radii (per the note under 4.4):\n")
    print(f"  {'k':>3}{'m_k':>13}{'set radius':>13}{'m_min^top/2':>14}{'joint':>13}")
    print("  " + "-" * 56)
    differ: list[int] = []
    for k in range(1, len(SCORES)):
        cert = certified_radius(sorted_scores, k)
        m_k = boundary_margin(sorted_scores, k).value
        order_txt = "undefined" if math.isnan(cert.order_radius) else f"{cert.order_radius:.9f}"
        print(
            f"  {k:>3}{m_k:>13.9f}{cert.set_radius:>13.9f}{order_txt:>14}{cert.joint_radius:>13.9f}"
        )
        if not math.isnan(cert.order_radius) and cert.order_radius != cert.set_radius:
            differ.append(k)

    print("\n  undefined at k = 1: the ordering minimum is over an empty set (G16).")
    print("  the two radii constrain disjoint sets of gaps -- the boundary gap k -> k+1 against")
    print("  the gaps strictly inside the top-k -- so neither bounds the other; here they")
    print(
        f"  disagree at k = {', '.join(str(k) for k in differ)}. A radius quoted without saying "
        "which invariant it"
    )
    print("  certifies is therefore ambiguous.")
    return radius


def _case_certified(table: AttributeTable, base_order: tuple[int, ...]) -> None:
    """Below the radius, in the direction that hurts the boundary most."""
    eps = 0.0625  # 2^-4, exactly half the certified radius
    inside = frozenset(base_order[:K])
    perturbed = tuple(s - eps if i in inside else s + eps for i, s in enumerate(SCORES))
    top = _report(
        "1. CERTIFIED -- below the radius the top-k cannot change",
        f"every score moves by the full {eps:.6f}, in the *worst* direction for the boundary:\n"
        "  everything inside the top-k pushed down, everything outside pushed up.",
        perturbed,
        table,
        base_order,
    )
    _require(top == frozenset(base_order[:K]), "a certified perturbation changed the top-k")
    print("\n  unchanged, as section 4.4 requires. No choice of directions could have done")
    print("  better: this is already the adversary's best move at this magnitude.")


def _case_uncertified_but_unchanged(table: AttributeTable, base_order: tuple[int, ...]) -> None:
    """Above the radius, aimed away from the boundary. The converse is false."""
    deltas = (-0.375, +0.0625, -0.5, +0.5, +0.125, +0.125)
    perturbed = tuple(s + d for s, d in zip(SCORES, deltas, strict=True))
    top = _report(
        "2. NOT CERTIFIED, YET UNCHANGED -- 'not certified' does not mean 'will change'",
        "a perturbation four times the radius, scattered across the corpus rather than\n"
        "  aimed at the boundary.",
        perturbed,
        table,
        base_order,
    )
    _require(top == frozenset(base_order[:K]), "the non-adversarial example changed the set")
    print("\n  the certificate says nothing here, and the top-k survives regardless. The")
    print("  theorem is an implication, not an equivalence; failing its hypothesis buys")
    print("  no information at all about the outcome.")
    after = _names(_order(perturbed, table)[:K])
    print(f"\n  the *ordering* inside the set does not survive: {_names(base_order[:K])} became")
    print(f"  {after} -- the separate guarantee that m_min^top/2 governs.")


def _case_adversarial(
    table: AttributeTable, base_order: tuple[int, ...], radius: float
) -> tuple[int, int]:
    """Just above the radius, aimed. Returns the two documents that swap."""
    delta = 2.0**-20
    witness = flip_witness(SCORES, base_order, K, delta=delta)
    _require(witness is not None, "no flip witness exists for this ranking")
    if witness is None:  # unreachable after _require; narrows the type for mypy
        raise SystemExit(1)
    perturbed, eps = witness
    _require(eps == radius + delta, "the witness eps is not m_k/2 + 2^-20")

    top = _report(
        "3. TIGHT -- an adversarial perturbation a hair above the radius DOES flip it",
        f"eps = m_k/2 + 2^-20 = {eps:.9f}, applied as -eps to the rank-{K} document and\n"
        f"  +eps to the rank-{K + 1} document. Nothing else moves at all.",
        perturbed,
        table,
        base_order,
    )
    _require(top != frozenset(base_order[:K]), "the witness failed to flip the top-k")

    a, b = base_order[K - 1], base_order[K]
    _require(
        perturbed[a] == 0.375 - delta and perturbed[b] == 0.375 + delta,
        "the witness did not land on the exact dyadic values",
    )
    print(
        f"\n  the two scores straddle 0.375 by exactly 2^-20, so {DOC_IDS[b]} overtakes "
        f"{DOC_IDS[a]}."
    )
    print("  m_k/2 is therefore the exact flip radius, not a conservative under-estimate:")
    print("  no larger radius is certifiable, because this witness exists for every delta > 0.")
    return a, b


def _case_exact_tie(table: AttributeTable, radius: float, a: int, b: int) -> None:
    """At eps = m_k/2 exactly, the margin is spent and the tie-break decides."""
    print(f"\n{_THIN}\nAt exactly eps = m_k/2 -- where A1 stops and A2 begins\n{_THIN}\n")
    tied = list(SCORES)
    tied[a] = SCORES[a] - radius
    tied[b] = SCORES[b] + radius
    _require(same_bits(tied[a], tied[b]), "the two halves did not meet bit-exactly")
    print(
        f"  {DOC_IDS[a]} - {radius:.6f} and {DOC_IDS[b]} + {radius:.6f} are bit-identical "
        f"({tied[a]:.9f}),"
    )
    tied_margin = boundary_margin(sorted_scores_desc(tied), K).value
    print(f"  so m_{K} is now exactly {tied_margin:.9f}: the margin has been spent in full and")
    print("  membership is decided by the tie-break alone. Swapping two popularity values,")
    print("  and changing nothing else whatsoever:\n")

    swapped = list(POPULARITY)
    swapped[a], swapped[b] = swapped[b], swapped[a]
    as_given = frozenset(_order(tied, table)[:K])
    reversed_pop = frozenset(_order(tied, _attribute_table(swapped))[:K])
    print(
        f"    popularity as given  ({DOC_IDS[b]}={POPULARITY[b]}, {DOC_IDS[a]}="
        f"{POPULARITY[a]})  ->  top-{K} {_fmt_set(as_given)}"
    )
    print(
        f"    popularity swapped   ({DOC_IDS[b]}={swapped[b]}, {DOC_IDS[a]}="
        f"{swapped[a]})  ->  top-{K} {_fmt_set(reversed_pop)}"
    )
    _require(as_given != reversed_pop, "the tie-break did not discriminate at the exact tie")
    print("\n  identical scores, identical ds, two different answers. That is why 4.4's")
    print("  inequality is strict, and it is exactly the boundary between A1 and A2.")


def main() -> int:
    _require(_is_exact(SCORES), "the base scores are not dyadic")
    table = _attribute_table(POPULARITY)
    base_order = _order(SCORES, table)

    _show_corpus(base_order)
    radius = _show_certificate(base_order)
    _case_certified(table, base_order)
    _case_uncertified_but_unchanged(table, base_order)
    a, b = _case_adversarial(table, base_order, radius)
    _case_exact_tie(table, radius, a, b)

    print(f"\n{_RULE}")
    print("  1. eps <  m_k/2, worst direction  ->  top-k unchanged  (guaranteed)")
    print("  2. eps = 4 m_k/2, scattered       ->  top-k unchanged  (not guaranteed: lucky)")
    print("  3. eps >  m_k/2, adversarial      ->  top-k CHANGED    (guaranteed possible)")
    print(f"{_RULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
