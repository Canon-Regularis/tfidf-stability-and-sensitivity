"""Score-separation margins and flip radii (README sections 2.3.2 and 4.4).

The centrepiece is the pair of tests on section 4.4's stability condition.

The paper states ``eps < m_k / 2`` as **sufficient** for top-k invariance and
never addresses whether it is tight. Both directions are established here:
perturbations below the radius provably never change the top-k set, and an
explicit adversarial witness just above it provably does. The witness is built
from dyadic rationals so that every intermediate value is exactly representable
in binary64 -- the test therefore has no floating-point slack at all, and
demonstrates that ``m_k / 2`` is the *exact* flip radius rather than merely a
safe under-estimate.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tfidf_stability.perturbation.score_bounds import certified_radius
from tfidf_stability.ranking.attributes import AttributeSpec, AttributeTable
from tfidf_stability.ranking.margins import (
    adjacent_gaps,
    boundary_margin,
    margin_profile,
    min_adjacent_margin_top,
    summarise,
)
from tfidf_stability.ranking.ranker import (
    rank,
    rank_all_operators,
    rank_top_k,
    sorted_scores_desc,
)
from tfidf_stability.ranking.sort_keys import PI, PI_ALT, SortKeySpec
from tfidf_stability.utils.numerics import same_bits
from tfidf_stability.utils.validation import KOutOfRangeError, StrictMode

LENIENT = StrictMode.LENIENT
POP = SortKeySpec("pop_only", ("popularity",))


def table_of(n: int, pops: list[int] | None = None) -> AttributeTable:
    pops = pops if pops is not None else [n - i for i in range(n)]
    return AttributeTable.from_records(
        [{"doc_id": f"d{i:04d}", "popularity": pops[i]} for i in range(n)],
        (AttributeSpec("popularity"),),
    )


# ---------------------------------------------------------------------------
# Basic quantities
# ---------------------------------------------------------------------------
def test_adjacent_gaps_are_the_consecutive_differences() -> None:
    assert adjacent_gaps((1.0, 0.75, 0.5, 0.5)) == (0.25, 0.25, 0.0)
    assert adjacent_gaps((1.0,)) == ()
    assert adjacent_gaps(()) == ()


def test_boundary_margin_on_a_known_ranking() -> None:
    s = (1.0, 0.75, 0.5, 0.5, 0.25)
    assert boundary_margin(s, 1).value == 0.25
    assert boundary_margin(s, 2).value == 0.25
    assert boundary_margin(s, 3).value == 0.0  # the exact tie
    assert boundary_margin(s, 4).value == 0.25


def test_min_adjacent_margin_is_the_smallest_gap_inside_the_top_k() -> None:
    s = (1.0, 0.75, 0.5, 0.5, 0.25)
    assert min_adjacent_margin_top(s, 2).value == 0.25
    assert min_adjacent_margin_top(s, 4).value == 0.0  # the tie is inside the top-4
    assert min_adjacent_margin_top(s, 3).value == 0.25


def test_flip_radius_is_exactly_half_the_margin_bitwise() -> None:
    """Division by two only shifts the exponent, so it must round-trip."""
    for s in [(1.0, 0.75), (0.3, 0.1), (1e-300, 0.0), (1.0, 1.0)]:
        m = boundary_margin(s, 1)
        assert same_bits(m.flip_radius * 2.0, m.value)


def test_margins_are_non_negative(mini_model) -> None:  # type: ignore[no-untyped-def]
    """A negative margin would mean the sort is broken."""
    s = sorted_scores_desc([0.4, 0.9, 0.1, 0.9, 0.0, 0.3])
    for k in range(1, len(s) + 1):
        m = boundary_margin(s, k, mode=LENIENT)
        assert not m.defined or m.value >= 0.0


# ---------------------------------------------------------------------------
# Tie-break independence -- what makes A1 and A2 separate questions
# ---------------------------------------------------------------------------
def test_margins_are_identical_under_all_three_operators(
    mini_attributes: AttributeTable,
) -> None:
    """``m_k`` depends only on the score *multiset*, never on the tie-break.

    The paper does not state this, and it is what makes research questions A1
    (margins) and A2 (tie-breaking) independent rather than confounded.
    """
    scores = [0.9, 0.9, 0.5, 0.5, 0.0, 0.2]
    rankings = rank_all_operators(scores, mini_attributes)

    # Structural: one shared array object, so it cannot drift.
    arrays = [r.sorted_scores for r in rankings.values()]
    assert all(a is arrays[0] for a in arrays)

    # And bitwise, when computed independently of that sharing.
    independent = sorted_scores_desc(scores)
    for k in range(1, 6):
        reference = boundary_margin(independent, k).value
        for r in rankings.values():
            assert same_bits(boundary_margin(r.sorted_scores, k).value, reference)


def test_no_negative_zero_in_scores(mini_model) -> None:  # type: ignore[no-untyped-def]
    """The precondition for ``sorted_scores`` being bit-determined by the multiset.

    ``+0.0`` and ``-0.0`` are equal but not bit-identical, so if both could
    occur, the sorted array's *bits* would depend on which copy landed where and
    permutation identity would not imply array identity. Cosine returns a
    literal ``0.0`` for the zero convention, and a quotient of non-negatives is
    ``+0.0``, so this holds -- but it is asserted rather than assumed.
    """
    from tfidf_stability.similarity.cosine import cosine_against_corpus
    from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

    q = TfidfVectoriser.transform_query(["numerical", "stability"], mini_model)
    docs = [mini_model.document(i) for i in range(mini_model.n_documents)]
    for s in cosine_against_corpus(q, docs, mini_model.norms):
        assert not (s == 0.0 and math.copysign(1.0, s) < 0), "a -0.0 escaped"


# ---------------------------------------------------------------------------
# G3 edge cases -- NaN plus a validity flag, never a coerced number
# ---------------------------------------------------------------------------
def test_k_equals_n_gives_nan_and_a_validity_flag() -> None:
    s = (1.0, 0.5, 0.25)
    m = boundary_margin(s, 3)
    assert math.isnan(m.value)
    assert m.defined is False
    assert m.reason
    # Explicitly *not* coerced: 0 would look like an exact tie, and infinity
    # would look like perfect stability. Both would corrupt a distribution.
    assert m.value != 0.0
    assert not math.isinf(m.value)
    assert m.is_exact_tie is False


def test_k_greater_than_n_strict_raises() -> None:
    with pytest.raises(KOutOfRangeError):
        boundary_margin((1.0, 0.5), 9, mode=StrictMode.STRICT)


def test_k_greater_than_n_lenient_clamps_and_records() -> None:
    m = boundary_margin((1.0, 0.5), 9, mode=LENIENT)
    assert m.k == 9
    assert m.k_effective == 2
    assert m.defined is False


def test_non_positive_k_is_rejected_in_both_modes() -> None:
    for mode in (StrictMode.STRICT, LENIENT):
        with pytest.raises(KOutOfRangeError):
            boundary_margin((1.0, 0.5), 0, mode=mode)


def test_exact_tie_gives_a_defined_zero_margin() -> None:
    """The interesting case, and it must be counted rather than dropped."""
    m = boundary_margin((0.5, 0.5, 0.1), 1)
    assert m.defined is True
    assert m.value == 0.0
    assert m.is_exact_tie is True
    assert m.flip_radius == 0.0


def test_min_adjacent_margin_at_k_equals_one_is_nan() -> None:
    """An empty minimum. G3 does not cover it; addendum G16 adopts NaN.

    ``+inf`` would be worse than wrong: it would silently claim "no constraint"
    and would pollute every percentile summary it entered.
    """
    m = min_adjacent_margin_top((1.0, 0.5), 1)
    assert math.isnan(m.value)
    assert m.defined is False
    assert not math.isinf(m.value)


def test_single_document_corpus_has_no_margins() -> None:
    assert boundary_margin((0.7,), 1, mode=LENIENT).defined is False
    assert min_adjacent_margin_top((0.7,), 1, mode=LENIENT).defined is False


def test_zero_query_makes_every_margin_zero() -> None:
    s = sorted_scores_desc([0.0] * 6)
    for k in range(1, 6):
        m = boundary_margin(s, k)
        assert m.value == 0.0
        assert m.is_exact_tie


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def test_undefined_margins_are_counted_not_dropped() -> None:
    s = (1.0, 0.5, 0.5)
    margins = [boundary_margin(s, k, mode=LENIENT) for k in (1, 2, 3, 99)]
    summary = summarise(margins)
    assert summary.n == 4
    assert summary.n_defined == 2
    assert summary.n_undefined == 2
    assert summary.n_exact_tie == 1
    assert summary.p_exact_tie == 0.5


def test_margin_profile_covers_the_paper_k_set() -> None:
    s = sorted_scores_desc([i / 100 for i in range(60)])
    profile = margin_profile(s)
    assert [m.k for m in profile] == [5, 10, 20, 50]
    assert all(m.defined for m in profile)


# ---------------------------------------------------------------------------
# Section 4.4 -- sufficiency
# ---------------------------------------------------------------------------
@given(
    st.lists(st.sampled_from([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), min_size=6, max_size=25),
    st.integers(min_value=1, max_value=5),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_perturbation_below_half_the_margin_never_changes_the_top_k_set(
    scores: list[float], k: int, fraction: float
) -> None:
    """Section 4.4: ``|ds_i| <= eps`` and ``eps < m_k / 2`` implies the top-k set
    is invariant.

    One detail decides whether this is a theorem check or a flaky annoyance. The
    proof is over the reals, but ``fl(s_i + d_i)`` rounds, so the *realised*
    perturbation can exceed the drawn ``eps`` by up to half an ulp. Drawing
    ``eps`` just below ``m_k / 2`` would then fail spuriously. The assumption is
    therefore made on the **realised** deltas.
    """
    n = len(scores)
    assume(k < n)
    table = table_of(n)
    sorted_s = sorted_scores_desc(scores)
    m = boundary_margin(sorted_s, k)
    assume(m.defined and m.value > 0.0)

    eps = m.flip_radius * fraction
    perturbed = [s + (eps if i % 2 else -eps) for i, s in enumerate(scores)]
    realised = max(abs(p - s) for p, s in zip(perturbed, scores, strict=True))
    assume(realised < m.flip_radius)

    before = set(rank(scores, table, POP).order[:k])
    after = set(rank(perturbed, table, POP).order[:k])
    assert before == after


# ---------------------------------------------------------------------------
# Section 4.4 -- necessity, which the paper does not address
# ---------------------------------------------------------------------------
def test_the_flip_radius_bound_is_tight() -> None:
    """An explicit witness just above ``m_k / 2`` that *does* flip the top-k.

    Every value is a dyadic rational, so every step below is exact in binary64
    and the test carries no floating-point slack whatsoever:

        score(r_k)   = 0.5      score(r_{k+1}) = 0.25
        m            = 0.25     m / 2          = 0.125
        delta        = 2^-30    eps            = 0.125 + 2^-30
        s'_a = 0.5  - eps = 0.375 - 2^-30
        s'_b = 0.25 + eps = 0.375 + 2^-30

    ``|ds_i| <= eps`` holds for every document, yet ``s'_b > s'_a``, so ``b``
    displaces ``a``. Therefore ``m_k / 2`` is the **exact** flip radius, not
    merely a sufficient bound -- a statement the paper makes only in one
    direction.
    """
    scores = [1.0, 0.5, 0.25, 0.0]  # a = index 1, b = index 2
    table = table_of(4)
    k = 2
    m = boundary_margin(sorted_scores_desc(scores), k)
    assert m.value == 0.25
    assert m.flip_radius == 0.125

    delta = 2.0**-30
    eps = m.flip_radius + delta
    perturbed = list(scores)
    perturbed[1] = scores[1] - eps
    perturbed[2] = scores[2] + eps

    # Exactness of the construction -- no rounding anywhere.
    assert perturbed[1] == 0.375 - delta
    assert perturbed[2] == 0.375 + delta
    assert max(abs(p - s) for p, s in zip(perturbed, scores, strict=True)) == eps
    # And no third document can intrude on the boundary.
    assert scores[0] - scores[1] > eps
    assert scores[2] - scores[3] > eps

    before = set(rank(scores, table, POP).order[:k])
    after = set(rank(perturbed, table, POP).order[:k])
    assert before == {0, 1}
    assert after == {0, 2}, "the witness must flip the top-k"


def test_at_exactly_half_the_margin_the_tie_break_decides() -> None:
    """The hinge between research questions A1 and A2.

    At ``eps = m / 2`` exactly, the two scores become bit-identical, so the
    margin has been consumed entirely and membership passes to the tie-break.
    Reproducible to the bit because the construction is dyadic.
    """
    scores = [1.0, 0.5, 0.25, 0.0]
    eps = 0.125
    a, b = scores[1] - eps, scores[2] + eps
    assert same_bits(a, b), "the perturbation exactly closes the gap"

    perturbed = [scores[0], a, b, scores[3]]
    assert boundary_margin(sorted_scores_desc(perturbed), 2).value == 0.0

    # With the scores now tied, the attribute priority alone picks the winner.
    high_first = table_of(4, pops=[9, 5, 7, 1])
    low_first = table_of(4, pops=[9, 7, 5, 1])
    assert rank(perturbed, high_first, POP).order[:2] == (0, 2)
    assert rank(perturbed, low_first, POP).order[:2] == (0, 1)


def test_zero_margin_flips_under_a_change_of_operator_alone(
    mini_attributes: AttributeTable,
) -> None:
    """Section 4.5 with ``ds = 0``: no numerical perturbation at all.

    When ``m_k = 0`` the flip radius is not a small radius, it is *no* radius:
    membership already depends only on the tie-break, so swapping the attribute
    priority changes the outcome with the scores held bit-identical.
    """
    scores = [0.0] * 6
    assert boundary_margin(sorted_scores_desc(scores), 3).value == 0.0

    pi_top = set(rank(scores, mini_attributes, PI).order[:3])
    alt_top = set(rank(scores, mini_attributes, PI_ALT).order[:3])
    assert pi_top != alt_top, "with zero margin the operator alone decides"


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
scores_strategy = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=30,
)


@given(scores_strategy)
def test_every_defined_margin_is_non_negative(scores: list[float]) -> None:
    s = sorted_scores_desc(scores)
    for k in range(1, len(s) + 1):
        m = boundary_margin(s, k, mode=LENIENT)
        assert not m.defined or m.value >= 0.0


@given(scores_strategy, st.integers(min_value=2, max_value=10))
def test_min_adjacent_margin_lower_bounds_every_top_k_gap(scores: list[float], k: int) -> None:
    s = sorted_scores_desc(scores)
    assume(k <= len(s))
    m = min_adjacent_margin_top(s, k)
    if m.defined:
        for j in range(k - 1):
            assert s[j] - s[j + 1] >= m.value


@given(scores_strategy)
def test_min_adjacent_margin_is_non_increasing_in_k(scores: list[float]) -> None:
    """Widening the window can only expose a smaller gap."""
    s = sorted_scores_desc(scores)
    previous = math.inf
    for k in range(2, len(s) + 1):
        m = min_adjacent_margin_top(s, k)
        assert m.defined
        assert m.value <= previous
        previous = m.value


# ---------------------------------------------------------------------------
# The adversarial attack on section 4.4
# ---------------------------------------------------------------------------
def _adversarial_table(n: int) -> AttributeTable:
    return AttributeTable.from_records(
        [
            {
                "doc_id": f"d{i:03d}",
                "popularity": (i * 37) % 11,
                "rating_sum2": (i * 5) % 9,
                "rating_count": 3,
                "engagement": (i * 13) % 7,
            }
            for i in range(n)
        ]
    )


@pytest.mark.slow
def test_no_certified_perturbation_ever_changes_the_top_k() -> None:
    """Section 4.4, attacked rather than sampled.

    Random perturbation almost never finds the worst case: the adversarial
    configuration -- drive the rank-k score down and the rank-(k+1) score up, both
    by the largest amount strictly inside the radius -- is a measure-zero corner
    of the perturbation cube. This constructs that corner directly, at every
    ``(n, k)`` and over dyadic scores so no step introduces rounding of its own.

    A single failure here falsifies the theorem or the implementation. It is the
    strongest correctness claim in the repository.
    """
    violations = []
    applied = 0
    for n, k in itertools.product((6, 12, 40), (1, 2, 5)):
        if k >= n:
            continue
        table = _adversarial_table(n)
        for trial in range(60):
            rng = random.Random(trial * 7919 + n * 31 + k)
            # Dyadic: every score and every difference is exact in binary64.
            scores = [rng.randrange(0, 64) / 64.0 for _ in range(n)]
            cert = certified_radius(sorted(scores, reverse=True), k)
            if not cert.defined or math.isnan(cert.set_radius) or cert.set_radius == 0.0:
                continue

            base = frozenset(rank_top_k(scores, table, k=k).order[:k])
            order = sorted(range(n), key=lambda i: (-scores[i], i))
            # NOT nextafter(radius, 0): adding that to a dyadic score rounds
            # straight back up to the radius, so the realised movement lands
            # exactly *on* the boundary the theorem excludes and every trial is
            # discarded. An earlier version of this attack made that mistake and
            # reported thousands of "certified" perturbations, none of which were
            # inside the radius at all. Backing off by 2^-30 keeps the realised
            # delta strictly inside while staying as adversarial as possible.
            eps = cert.set_radius * (1.0 - 2.0**-30)

            for direction in (+1, -1):
                perturbed = list(scores)
                for position, index in enumerate(order):
                    perturbed[index] = scores[index] + (-eps if position < k else eps) * direction
                # The realised movement, not the intended one: fl(s + d) rounds.
                realised = max(abs(a - b) for a, b in zip(perturbed, scores, strict=True))
                if realised >= cert.set_radius:
                    continue
                applied += 1
                if frozenset(rank_top_k(perturbed, table, k=k).order[:k]) != base:
                    violations.append((n, k, trial, realised, cert.set_radius))

    assert applied > 500, f"the attack only applied {applied} certified perturbations"
    assert not violations, f"section 4.4 falsified: {violations[:3]}"


@pytest.mark.slow
def test_the_radius_is_tight_not_merely_conservative() -> None:
    """Exactly *at* the radius the theorem does not apply, and pairs do flip.

    Without this, ``test_no_certified_perturbation_ever_changes_the_top_k``
    would also pass against an absurdly small radius. This is what shows the
    bound is the real one.
    """
    flipped = total = 0
    for n, k in itertools.product((6, 12), (1, 2, 5)):
        if k >= n:
            continue
        table = _adversarial_table(n)
        for trial in range(60):
            rng = random.Random(trial * 104729 + n + k)
            scores = [rng.randrange(0, 64) / 64.0 for _ in range(n)]
            cert = certified_radius(sorted(scores, reverse=True), k)
            if not cert.defined or math.isnan(cert.set_radius) or cert.set_radius == 0.0:
                continue
            base = frozenset(rank_top_k(scores, table, k=k).order[:k])
            order = sorted(range(n), key=lambda i: (-scores[i], i))
            perturbed = list(scores)
            for position, index in enumerate(order):
                perturbed[index] = scores[index] + (
                    -cert.set_radius if position < k else cert.set_radius
                )
            total += 1
            if frozenset(rank_top_k(perturbed, table, k=k).order[:k]) != base:
                flipped += 1

    assert total > 100
    assert flipped > total // 4, (
        f"only {flipped}/{total} flipped at exactly the radius; if the bound were "
        f"tight this should be common, so the radius may be far too small"
    )
