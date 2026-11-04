"""Score-separation margins and flip radii (README sections 2.3.2 and 4.4).

Both directions of the ``eps < m_k / 2`` bound are covered: below the radius
the top-k set does not move, and a witness just above it flips. The witness is
dyadic, so each of its values is exact in binary64 and it carries no slack.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tfidf_stability.perturbation.score_bounds import (
    StabilityCertificate,
    certified_radius,
    flip_witness,
    is_order_stable,
    is_top_k_stable,
)
from tfidf_stability.ranking.attributes import AttributeSpec, AttributeTable
from tfidf_stability.ranking.margins import (
    Margin,
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
from tfidf_stability.utils.numerics import same_bits, ulps_between
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
# Tie-break independence: why A1 and A2 are separate questions
# ---------------------------------------------------------------------------
def test_margins_are_identical_under_all_three_operators(
    mini_attributes: AttributeTable,
) -> None:
    """``m_k`` depends only on the score multiset, never on the tie-break.

    The paper does not state this; it is what keeps A1 (margins) and A2
    (tie-breaking) independent rather than confounded.
    """
    scores = [0.9, 0.9, 0.5, 0.5, 0.0, 0.2]
    rankings = rank_all_operators(scores, mini_attributes)

    # Structural: every operator returns the same array object.
    arrays = [r.sorted_scores for r in rankings.values()]
    assert all(a is arrays[0] for a in arrays)

    # And bitwise, recomputed without that sharing.
    independent = sorted_scores_desc(scores)
    for k in range(1, 6):
        reference = boundary_margin(independent, k).value
        for r in rankings.values():
            assert same_bits(boundary_margin(r.sorted_scores, k).value, reference)


def test_no_negative_zero_in_scores(mini_model) -> None:  # type: ignore[no-untyped-def]
    """Precondition for ``sorted_scores`` being bit-determined by the multiset.

    ``+0.0`` and ``-0.0`` compare equal but differ in bits, so with both present
    the sorted array's bits would depend on placement. Cosine returns a literal
    ``0.0`` for the zero convention and a quotient of non-negatives is ``+0.0``.
    """
    from tfidf_stability.similarity.cosine import cosine_against_corpus
    from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

    q = TfidfVectoriser.transform_query(["numerical", "stability"], mini_model)
    docs = [mini_model.document(i) for i in range(mini_model.n_documents)]
    for s in cosine_against_corpus(q, docs, mini_model.norms):
        assert not (s == 0.0 and math.copysign(1.0, s) < 0), "a -0.0 escaped"


# ---------------------------------------------------------------------------
# G3 edge cases: NaN plus a validity flag rather than a coerced number
# ---------------------------------------------------------------------------
def test_k_equals_n_gives_nan_and_a_validity_flag() -> None:
    s = (1.0, 0.5, 0.25)
    m = boundary_margin(s, 3)
    assert math.isnan(m.value)
    assert m.defined is False
    assert m.reason
    # Uncoerced: 0 reads as an exact tie and infinity as perfect stability,
    # and either would corrupt a summary distribution.
    assert m.value != 0.0
    assert not math.isinf(m.value)
    assert m.is_exact_tie is False


def test_k_greater_than_n_lenient_clamps_and_records() -> None:
    m = boundary_margin((1.0, 0.5), 9, mode=LENIENT)
    assert m.k == 9
    assert m.k_effective == 2
    assert m.defined is False


def test_exact_tie_gives_a_defined_zero_margin() -> None:
    """A tie is a defined margin of zero, so summaries count it rather than drop it."""
    m = boundary_margin((0.5, 0.5, 0.1), 1)
    assert m.defined is True
    assert m.value == 0.0
    assert m.is_exact_tie is True
    assert m.flip_radius == 0.0


def test_min_adjacent_margin_at_k_equals_one_is_nan() -> None:
    """The order margin at ``k = 1`` is an empty minimum, which G16 makes NaN.

    ``+inf`` would claim "no constraint" and pollute every percentile summary
    it entered; G3 does not cover the case and addendum G16 adopts NaN.
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
# Section 4.4: sufficiency
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(
    st.lists(st.sampled_from([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), min_size=6, max_size=25),
    st.integers(min_value=1, max_value=5),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_perturbation_below_half_the_margin_never_changes_the_top_k_set(
    scores: list[float], k: int, fraction: float
) -> None:
    """Section 4.4: ``|ds_i| <= eps`` with ``eps < m_k / 2`` keeps the top-k set.

    The proof is over the reals, but ``fl(s_i + d_i)`` rounds, so the realised
    perturbation can exceed the drawn ``eps`` by up to half an ulp. The assume()
    is therefore on the realised deltas, not on the drawn ``eps``.
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
# Section 4.4: necessity, which the paper does not address
# ---------------------------------------------------------------------------
def test_the_flip_radius_bound_is_tight() -> None:
    """A witness just above ``m_k / 2`` that flips the top-k.

    Every value is dyadic (scores 0.5 and 0.25, ``eps = 0.125 + 2^-30``), so
    each step is exact in binary64 and the test carries no floating-point slack.
    Rationale: docs/engineering_notes.md#dyadic-flip-witness
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

    # The construction is exact: nothing rounds.
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
    """At ``eps = m / 2`` membership passes from the margin to the tie-break.

    The two boundary scores become bit-identical; this is where research
    question A1 (margins) hands over to A2 (tie-breaking). The values are
    dyadic, so the equality is exact.
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

    ``m_k = 0`` leaves no radius, so membership already rests on the tie-break;
    swapping the attribute priority moves the top-k with the scores held
    bit-identical.
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


@pytest.mark.property
@given(scores_strategy)
def test_every_defined_margin_is_non_negative(scores: list[float]) -> None:
    s = sorted_scores_desc(scores)
    for k in range(1, len(s) + 1):
        m = boundary_margin(s, k, mode=LENIENT)
        assert not m.defined or m.value >= 0.0


@pytest.mark.property
@given(scores_strategy, st.integers(min_value=2, max_value=10))
def test_min_adjacent_margin_lower_bounds_every_top_k_gap(scores: list[float], k: int) -> None:
    s = sorted_scores_desc(scores)
    assume(k <= len(s))
    m = min_adjacent_margin_top(s, k)
    if m.defined:
        for j in range(k - 1):
            assert s[j] - s[j + 1] >= m.value


@pytest.mark.property
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

    The worst case (rank-k score down, rank-(k+1) up, each by the largest amount
    strictly inside the radius) is a measure-zero corner of the perturbation
    cube that random sampling does not reach, so it is built at every ``(n, k)``.
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
            # nextafter(radius, 0) added to a dyadic score rounds back up and
            # puts the realised movement on the boundary the theorem excludes.
            # A 2^-30 backoff stays strictly inside and is still adversarial.
            eps = cert.set_radius * (1.0 - 2.0**-30)

            for direction in (+1, -1):
                perturbed = list(scores)
                for position, index in enumerate(order):
                    perturbed[index] = scores[index] + (-eps if position < k else eps) * direction
                # Realised movement rather than intended: fl(s + d) rounds.
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
    """At the radius the theorem does not apply, and pairs do flip.

    ``test_no_certified_perturbation_ever_changes_the_top_k`` would also pass
    against an arbitrarily small radius, so this pins the radius from below.
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


def test_a_summary_of_only_undefined_margins_reports_no_percentiles() -> None:
    """A summary of only undefined margins carries counts but no percentiles.

    The counts still reach the manifest; the percentiles would not be a
    measured distribution, and an empty tuple says so where a row of NaNs
    would not.
    """
    s = (1.0, 0.5)
    margins = [boundary_margin(s, 2, mode=LENIENT), boundary_margin(s, 99, mode=LENIENT)]
    assert all(not m.defined for m in margins), "the premise: nothing is defined"

    summary = summarise(margins)
    assert summary.percentiles == ()
    assert summary.n == 2
    assert summary.n_defined == 0
    assert summary.n_undefined == 2
    assert math.isnan(summary.p_exact_tie), "a rate over nothing is not zero"


def test_the_percentile_index_is_nearest_rank_on_a_known_distribution() -> None:
    """Ten margins of 1.0 to 10.0, so every quantile has one right answer.

    Nearest rank is `defined[ceil(q * n) - 1]`; the expected values pin the
    product `q * n`, the `- 1` and the lower clamp. Section 7 reports these.
    """
    margins = [Margin("boundary", 1, 1, float(i), True) for i in range(1, 11)]
    summary = summarise(margins)
    got = dict(summary.percentiles)

    assert summary.n_defined == 10
    # ceil(q*10) - 1, clamped: 0.01 -> 0, 0.25 -> 2, 0.5 -> 4, 0.75 -> 7, 0.99 -> 9.
    assert got[0.01] == 1.0
    assert got[0.05] == 1.0
    assert got[0.25] == 3.0
    assert got[0.5] == 5.0, "the median is the 5th of ten, not the 6th and not the 1st"
    assert got[0.75] == 8.0
    assert got[0.95] == 10.0
    assert got[0.99] == 10.0


def test_the_default_quantiles_are_the_seven_the_paper_reports() -> None:
    """The set is a default argument, so nothing else pins it."""
    margins = [Margin("boundary", 1, 1, float(i), True) for i in range(1, 11)]
    assert [q for q, _ in summarise(margins).percentiles] == [
        0.01,
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
        0.99,
    ]


def test_a_single_defined_margin_is_every_percentile_of_itself() -> None:
    """n = 1 maps every default quantile to index 0: `ceil(q * 1) - 1` is 0."""
    got = dict(summarise([Margin("boundary", 1, 1, 7.5, True)]).percentiles)
    assert set(got.values()) == {7.5}


def test_an_undefined_boundary_margin_says_which_case_it_hit() -> None:
    """The reason text distinguishes `k == N` from `k > N`.

    One is the top of the ranking, the other a request the corpus could not
    serve, and the reason is what a run report shows a reader.
    """
    s = (1.0, 0.5, 0.25)
    assert boundary_margin(s, 3, mode=LENIENT).reason == "k == N: r_{k+1} does not exist"
    assert boundary_margin(s, 99, mode=LENIENT).reason == "k clamped to N"


# ---------------------------------------------------------------------------
# Erroneous: the two refusals, with the message that tells them apart
# ---------------------------------------------------------------------------


# Both arrive as `KOutOfRangeError`, so a test asserting only the type passes
# whichever guard fired. They are different situations: one is a request the
# corpus cannot serve, the other a non-positive k that no clamp can repair.
@pytest.mark.parametrize("n", [0, 1, 2, 5])
@pytest.mark.parametrize("mode", [StrictMode.STRICT, LENIENT])
@pytest.mark.parametrize("k", [0, -1, -(2**40)])
def test_a_non_positive_k_is_refused_whatever_the_corpus_or_the_mode(
    k: int, mode: StrictMode, n: int
) -> None:
    """A non-positive k is refused regardless of corpus size and mode.

    Lenient mode clamps an over-large k because a sweep point past the end of
    a small corpus is legitimate; there is nothing to clamp a non-positive k to.
    """
    scores = tuple(1.0 - 0.1 * i for i in range(n))
    with pytest.raises(KOutOfRangeError, match=f"k must be positive, got {k}"):
        boundary_margin(scores, k, mode=mode)


@pytest.mark.parametrize("k", [3, 9, 2**40])
def test_an_over_large_k_names_the_corpus_it_exceeded(k: int) -> None:
    """The message carries both numbers and the way out.

    A caller sweeping k over a corpus smaller than the grid wants lenient mode,
    not a smaller grid.
    """
    with pytest.raises(KOutOfRangeError, match=f"k={k} exceeds the 2 rankable documents"):
        boundary_margin((1.0, 0.5), k, mode=StrictMode.STRICT)


def test_the_refusal_points_at_the_mode_that_would_have_clamped() -> None:
    with pytest.raises(KOutOfRangeError, match=r"Use StrictMode\.LENIENT to clamp\."):
        boundary_margin((1.0, 0.5), 9, mode=StrictMode.STRICT)


# ---------------------------------------------------------------------------
# Boundary: the two ways a margin comes out undefined
# ---------------------------------------------------------------------------
def test_the_two_undefined_reasons_are_told_apart_by_their_text() -> None:
    """The two undefined reasons are told apart only by their text.

    At `k == N` only `r_{k+1}` is missing; `k > N` clamped is a request the
    corpus could not serve. Both carry `defined=False` and a NaN, so the reason
    is the only field separating them.
    """
    at_the_end = boundary_margin((1.0, 0.5), 2, mode=LENIENT)
    clamped = boundary_margin((1.0, 0.5), 9, mode=LENIENT)

    assert at_the_end.reason == "k == N: r_{k+1} does not exist"
    assert clamped.reason == "k clamped to N"
    assert at_the_end.k_effective == clamped.k_effective == 2
    assert not at_the_end.defined
    assert not clamped.defined


def test_a_clamped_margin_still_reports_the_k_that_was_asked_for() -> None:
    """`k` is the requested value and `k_effective` the computed one.

    Section 7.1 reports rates per requested k, so losing the request would
    merge grid points that all clamped to the same effective k.
    """
    margin = boundary_margin((1.0, 0.5), 9, mode=LENIENT)
    assert (margin.k, margin.k_effective) == (9, 2)


def test_an_empty_score_array_gives_an_undefined_margin_rather_than_refusing() -> None:
    """An empty score array gives an undefined margin, unlike ranking.

    G17 makes ranking an empty corpus an error (`EmptyCorpusError`) because
    there is no ranking to return. An undefined margin is a value a sweep can
    carry, so a degenerate grid point does not abort the run.
    """
    margin = boundary_margin((), 1, mode=LENIENT)
    assert margin.k_effective == 0
    assert not margin.defined
    assert math.isnan(margin.value)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_a_margin_is_defined_exactly_when_the_boundary_has_a_rank_below_it(n: int) -> None:
    """A margin is defined for `k < N` and undefined for `k >= N`.

    The sweep covers every k a corpus of this size admits, so an off-by-one at
    either end shows up as a defined margin that should not exist, or the
    reverse.
    """
    scores = tuple(1.0 - 0.1 * i for i in range(n))
    for k in range(1, n + 1):
        margin = boundary_margin(scores, k, mode=LENIENT)
        assert margin.defined is (k < n), f"n={n} k={k}"
    assert n >= 1, "the sweep ran over at least one k"


# ---------------------------------------------------------------------------
# Boundary: the value, bit for bit
# ---------------------------------------------------------------------------
def test_a_defined_margin_is_the_subtraction_and_nothing_else() -> None:
    """No rounding, no absolute value, no clamping: `S[k-1] - S[k]` exactly."""
    assert same_bits(boundary_margin((1.0, 0.5), 1).value, 0.5)
    assert same_bits(boundary_margin((1.0, 0.75, 0.25), 2).value, 0.5)


def test_a_margin_of_one_ulp_is_carried_at_full_precision() -> None:
    """A one-ulp margin is carried at full precision.

    A margin computed at reduced precision, or through a rounded intermediate,
    would collapse to zero and report an exact tie that is not one.
    """
    below = math.nextafter(1.0, 0.0)
    margin = boundary_margin((1.0, below), 1)

    assert margin.defined
    assert same_bits(margin.value, math.ulp(below))
    assert margin.value > 0.0
    assert not margin.is_exact_tie


# ---------------------------------------------------------------------------
# flip_radius: where the docstring's exactness claim stops holding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [1.0, 0.5, 0.1, 2.0**-1021, 2.0**-1074 * 2])
def test_halving_a_margin_and_doubling_it_back_recovers_it_exactly(value: float) -> None:
    """Division by two only shifts the exponent, so the round trip is exact.

    This holds for every margin whose exponent has room to shift.
    """
    margin = Margin("boundary", 1, 1, value, True)
    assert same_bits(2.0 * margin.flip_radius, value)


def test_the_exactness_claim_fails_for_every_odd_subnormal_margin() -> None:
    """Halving a subnormal margin is not a shift.

    A subnormal has no exponent left to shift, so halving rounds to nearest
    even: an odd multiple of `5e-324` loses a bit and `5e-324` itself halves to
    zero. Rationale: docs/engineering_notes.md#subnormal-flip-radius
    """
    smallest = 5e-324

    margin = Margin("boundary", 1, 1, smallest, True)
    assert margin.flip_radius == 0.0
    assert 2.0 * margin.flip_radius != smallest
    assert not margin.is_exact_tie, "the margin itself is still non-zero"

    def round_trips(multiple: int) -> bool:
        value = multiple * smallest
        return same_bits(2.0 * Margin("boundary", 1, 1, value, True).flip_radius, value)

    assert [m for m in range(1, 40, 2) if round_trips(m)] == [], "odd multiples lose a bit"
    assert [m for m in range(2, 40, 2) if not round_trips(m)] == [], "even multiples survive"


def test_the_flip_radius_of_a_negative_zero_margin_keeps_its_sign() -> None:
    """`-0.0 / 2.0` is `-0.0`, and the sign survives into the flip radius.

    The value compares equal to zero, so nothing downstream changes, but the
    bits differ and this project compares bits.
    """
    assert same_bits(Margin("boundary", 1, 1, -0.0, True).flip_radius, -0.0)


# ---------------------------------------------------------------------------
# is_exact_tie: a two-field truth table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("defined", "value", "expected"),
    [
        (True, 0.0, True),
        (True, -0.0, True),
        (True, 5e-324, False),
        (True, 1.0, False),
        (False, math.nan, False),
        (False, 0.0, False),
    ],
)
def test_an_exact_tie_is_a_defined_margin_of_zero(
    defined: bool, value: float, expected: bool
) -> None:
    """An exact tie requires both `defined` and a zero value.

    `P(m_k = 0)` is a headline statistic, so an undefined margin counted as a
    tie would inflate it, and `-0.0` not counted would deflate it.
    """
    assert Margin("boundary", 1, 1, value, defined).is_exact_tie is expected


# ---------------------------------------------------------------------------
# Degenerate inputs, and the preconditions that are not checked
# ---------------------------------------------------------------------------
def test_an_infinite_score_gives_an_infinite_margin_rather_than_refusing() -> None:
    """Margins do not re-validate finiteness.

    `rank` rejects a non-finite score before sorting; this pins that division
    of responsibility, so a guard added here changes a tested contract.
    """
    margin = boundary_margin((math.inf, 1.0), 1)
    assert margin.defined
    assert margin.value == math.inf


def test_a_non_increasing_input_yields_a_negative_margin_rather_than_an_error() -> None:
    """A non-increasing input yields a negative margin rather than an error.

    The argument is named `sorted_scores` and every caller satisfies the order
    by construction. Nothing checks it, so an ascending array gives a negative
    margin and flip radius; a guard here would run on every margin in a sweep.
    """
    margin = boundary_margin((0.0, 1.0), 1)
    assert margin.defined
    assert margin.value == -1.0
    assert margin.flip_radius == -0.5


# ---------------------------------------------------------------------------
# adjacent_gaps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7])
def test_there_is_one_fewer_gap_than_there_are_scores(n: int) -> None:
    """There are `n - 1` gaps for `n` scores.

    `tie_groups` reads these as the runs of gaps below tau, so an extra or
    missing entry shifts every chain boundary.
    """
    scores = tuple(1.0 - 0.1 * i for i in range(n))
    assert len(adjacent_gaps(scores)) == max(0, n - 1)


def test_the_gap_between_the_two_zeros_keeps_the_sign_of_the_subtraction() -> None:
    """`-0.0 - 0.0` is `-0.0`, not `+0.0`, and the gap keeps that sign.

    It is an exact tie either way, and the bits differ.
    """
    (gap,) = adjacent_gaps((-0.0, 0.0))
    assert gap == 0.0
    assert same_bits(gap, -0.0)


def test_two_infinite_scores_have_no_gap_between_them() -> None:
    """`inf - inf` is NaN, so two infinite scores have no usable gap.

    The gap is neither zero nor positive, which is one reason `rank` refuses
    non-finite scores upstream rather than here.
    """
    (gap,) = adjacent_gaps((math.inf, math.inf))
    assert math.isnan(gap)


# ---------------------------------------------------------------------------
# min_adjacent_margin_top: G16
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5])
def test_the_order_margin_at_k_one_is_undefined_and_not_infinite(n: int) -> None:
    """The order margin at `k = 1` is a minimum over an empty set.

    G16 adopts NaN because `+inf` would claim "no constraint at all" and enter
    a percentile summary as its largest value, the opposite of what an empty
    constraint set means.
    """
    scores = tuple(1.0 - 0.1 * i for i in range(n))
    margin = min_adjacent_margin_top(scores, 1, mode=LENIENT)

    assert not margin.defined
    assert math.isnan(margin.value)
    assert not math.isinf(margin.value)
    assert margin.reason == "k == 1: vacuous minimum"


def test_the_order_margin_is_the_smallest_gap_strictly_inside_the_top_k() -> None:
    """The order margin covers gaps 1->2 through (k-1)->k, not k->k+1.

    Those two gap sets are disjoint, which is why neither radius dominates the
    other.
    """
    # Dyadic throughout, so every gap is exact and the assertion is the
    # arithmetic rather than a rounding of it.
    scores = (1.0, 0.75, 0.5, 0.0)
    assert same_bits(min_adjacent_margin_top(scores, 3).value, 0.25), "min(1->2, 2->3)"
    assert same_bits(boundary_margin(scores, 3).value, 0.5), "the 3->4 gap, which it excludes"


# ---------------------------------------------------------------------------
# margin_profile
# ---------------------------------------------------------------------------
def test_an_empty_k_set_produces_no_margins() -> None:
    """An empty k set is a legitimate configuration rather than one to refuse."""
    assert margin_profile((1.0, 0.5), ks=()) == ()


def test_a_non_positive_k_in_the_grid_is_refused_even_in_the_lenient_default() -> None:
    """The lenient default does not extend to a non-positive k.

    `margin_profile` defaults to lenient so a k past the corpus size is a grid
    point rather than a misconfiguration; a k of zero has nothing to clamp to.
    """
    with pytest.raises(KOutOfRangeError, match="k must be positive, got 0"):
        margin_profile((1.0, 0.5), ks=(1, 0))


def test_the_profile_keeps_the_order_and_the_repeats_of_its_k_set() -> None:
    """The profile is reported positionally against the requested grid.

    Neither sorting nor de-duplicating the k set would be safe.
    """
    profile = margin_profile((1.0, 0.5, 0.25), ks=(2, 1, 2))
    assert [m.k for m in profile] == [2, 1, 2]


# ---------------------------------------------------------------------------
# summarise: quantiles at and beyond the ends
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("q", "expected"), [(0.0, 1.0), (1.0, 4.0), (-1.0, 1.0), (2.0, 4.0)])
def test_a_quantile_outside_the_unit_interval_lands_on_an_end(q: float, expected: float) -> None:
    """An out-of-range quantile reports the smallest or largest observation.

    The index is clamped into range rather than raising; the same clamp maps
    `q = 0.0` to index 0, not to `defined[-1]`, which is the maximum.
    """
    margins = [Margin("boundary", 1, 1, float(i), True) for i in range(1, 5)]
    assert dict(summarise(margins, quantiles=[q]).percentiles)[q] == expected


@pytest.mark.parametrize(
    ("q", "error"), [(math.nan, ValueError), (math.inf, OverflowError), (-math.inf, OverflowError)]
)
def test_a_non_finite_quantile_is_refused_by_the_index_arithmetic(
    q: float, error: type[Exception]
) -> None:
    """`math.ceil` cannot convert a non-finite quantile to an integer.

    The refusal comes from the index arithmetic rather than a guard in this
    module, and a quantile grid built from a computed value could carry one.
    """
    margins = [Margin("boundary", 1, 1, 1.0, True)]
    with pytest.raises(error, match="cannot convert float"):
        summarise(margins, quantiles=[q])


def test_the_exact_tie_rate_is_taken_over_the_defined_margins_only() -> None:
    """The denominator of the exact-tie rate is `n_defined`, not `n`.

    An undefined margin is no evidence either way, and counting it would drag
    the headline statistic towards zero on corpora where most margins are
    undefined.
    """
    margins = [
        Margin("boundary", 1, 1, 0.0, True),
        Margin("boundary", 2, 2, 1.0, True),
        Margin("boundary", 3, 3, math.nan, False),
        Margin("boundary", 4, 4, math.nan, False),
    ]
    summary = summarise(margins)

    assert (summary.n, summary.n_defined, summary.n_undefined) == (4, 2, 2)
    assert summary.p_exact_tie == 0.5


# ---------------------------------------------------------------------------
# The section 4.4 certificates
# ---------------------------------------------------------------------------
def test_a_certificate_carries_both_radii_because_neither_dominates() -> None:
    """Neither radius dominates the other, so the certificate carries both.

    A tight cluster at the top with a wide boundary makes the order radius
    bind; a well-spread top with a near-tied boundary makes the set radius bind.
    """
    order_binds = certified_radius((1.0, 0.99, 0.5, 0.0), 3)
    assert order_binds.order_radius < order_binds.set_radius
    assert order_binds.order_radius_is_binding

    set_binds = certified_radius((1.0, 0.5, 0.25, 0.24), 3)
    assert set_binds.set_radius < set_binds.order_radius
    assert not set_binds.order_radius_is_binding


def test_the_joint_radius_is_the_smaller_of_the_two() -> None:
    """The joint radius is the minimum of the set and order radii.

    Section 4.4's two conditions constrain disjoint gap sets, so a caller
    wanting both guarantees gets the smaller.
    """
    cert = certified_radius((1.0, 0.99, 0.5, 0.0), 3)
    assert cert.joint_radius == min(cert.set_radius, cert.order_radius)


def test_a_certificate_with_no_order_radius_falls_back_to_the_set_radius() -> None:
    """At `k = 1` the joint radius is the set radius alone.

    The order margin is undefined there (G16) because there is no ordering to
    preserve inside the top-1; NaN would read as "nothing is certified".
    """
    cert = certified_radius((1.0, 0.5, 0.25), 1)
    assert math.isnan(cert.order_radius)
    assert cert.joint_radius == cert.set_radius
    assert not cert.order_radius_is_binding


def test_an_undefined_certificate_certifies_nothing_at_all() -> None:
    """At `k == N` the joint radius is NaN rather than zero.

    There is no rank below the boundary, so no perturbation can be certified.
    """
    cert = certified_radius((1.0, 0.5), 2)
    assert not cert.defined
    assert math.isnan(cert.joint_radius)
    assert not cert.order_radius_is_binding


def test_an_exact_tie_certifies_a_radius_of_zero_rather_than_nothing() -> None:
    """An exact tie is a defined certificate with a radius of zero.

    The margin exists and is zero, so membership is already decided by the
    tie-break and a change of operator flips it at `ds = 0`.
    """
    cert = certified_radius((0.5, 0.5, 0.1), 1)
    assert cert.defined
    assert cert.exact_tie
    assert cert.set_radius == 0.0


@pytest.mark.parametrize("checker", [is_top_k_stable, is_order_stable])
def test_stability_is_not_guaranteed_at_exactly_the_radius(checker: object) -> None:
    """The stability inequality is strict.

    At `eps == m/2` the two boundary scores can be driven to equality and
    membership passes to the tie-break; one ulp below, the guarantee holds.
    """
    scores = (1.0, 0.9, 0.5, 0.0)
    cert = certified_radius(scores, 3)
    radius = cert.order_radius if checker is is_order_stable else cert.set_radius

    assert not checker(scores, 3, radius)  # type: ignore[operator]
    assert checker(scores, 3, math.nextafter(radius, 0.0))  # type: ignore[operator]


@pytest.mark.parametrize("checker", [is_top_k_stable, is_order_stable])
def test_an_undefined_certificate_guarantees_nothing_at_any_epsilon(checker: object) -> None:
    """An undefined certificate answers False at every epsilon, even zero.

    "Not guaranteed" rather than "will change": the condition is sufficient,
    not necessary, and an undefined certificate has nothing to appeal to.
    """
    assert not checker((1.0, 0.5), 2, 0.0)  # type: ignore[operator]
    assert not checker((1.0, 0.5), 2, -1.0)  # type: ignore[operator]


def test_an_undefined_certificate_is_never_binding_even_with_a_radius_present() -> None:
    """An undefined certificate is never binding, whatever its order radius.

    The fields are independent and `certified_radius` is not the only
    constructor, so a certificate can carry a numeric `order_radius` and still
    be undefined; the guard tests `defined` before the comparison.
    """
    undefined_but_numeric = StabilityCertificate(
        k=3, set_radius=1.0, order_radius=0.5, exact_tie=False, defined=False
    )
    assert not undefined_but_numeric.order_radius_is_binding
    assert math.isnan(undefined_but_numeric.joint_radius)


def test_equal_radii_leave_neither_one_binding() -> None:
    """Equal radii leave neither one binding.

    "Binding" means strictly tighter, and when the two coincide there is
    nothing to disambiguate; reporting the order radius as binding would
    misstate which constraint limited the certificate.
    """
    balanced = StabilityCertificate(
        k=2, set_radius=0.25, order_radius=0.25, exact_tie=False, defined=True
    )
    assert not balanced.order_radius_is_binding
    assert balanced.joint_radius == 0.25


# ---------------------------------------------------------------------------
# flip_witness: the boundary rank, and how far past the radius it goes
# ---------------------------------------------------------------------------


# Several tests below take the default `delta`, which every caller that omits
# it gets, and two reach the rank guard from below.
@pytest.mark.parametrize("k", [0, -1, -(2**30)])
def test_a_boundary_rank_below_one_has_no_witness(k: int) -> None:
    """Ranks are 1-indexed, so `k = 0` names no boundary.

    Without the lower bound the function would index `order[-1]`, the last
    document, and construct a witness for the worst-ranked pair while claiming
    to describe the top of the ranking.
    """
    assert flip_witness([1.0, 0.5, 0.25, 0.0], [0, 1, 2, 3], k) is None


@pytest.mark.parametrize("k", [4, 5, 2**30])
def test_a_boundary_rank_at_or_past_the_end_has_no_witness(k: int) -> None:
    """At `k == N` there is no rank below the boundary to swap with.

    This is the same condition that makes the margin undefined.
    """
    assert flip_witness([1.0, 0.5, 0.25, 0.0], [0, 1, 2, 3], k) is None


def test_an_exact_tie_has_no_radius_to_exceed() -> None:
    """An exact tie has no radius for a witness to exceed.

    The pair is already tied, so `None` is returned rather than a zero-epsilon
    witness, which would claim a flip the tie-break already owns.
    """
    assert flip_witness([0.5, 0.5, 0.5, 0.5], [0, 1, 2, 3], 1) is None


def test_the_default_witness_only_just_exceeds_the_radius() -> None:
    """The default witness only just exceeds the radius.

    The default delta is a millionth of the radius, floored at a few ulp so the
    excess survives rounding. The floor makes the exact figure scale-dependent,
    so the test bounds the ratio rather than asserting a value.
    """
    scores = [1.0, 0.5, 0.25, 0.0]
    witness = flip_witness(scores, [0, 1, 2, 3], 2)
    assert witness is not None
    _, eps = witness

    radius = (scores[1] - scores[2]) / 2.0
    assert eps > radius, "it has to exceed the radius, or it is not a witness"
    # A millionth of the radius. Bounded at two millionths rather than loosely,
    # because "only just past" is the whole claim: a witness a percent past the
    # radius would flip the pair and say nothing about the radius being tight.
    assert eps <= radius * (1.0 + 2e-6)


def test_the_witness_moves_only_the_two_documents_at_the_boundary() -> None:
    """The witness moves only the two documents at the boundary.

    `|ds_i| <= eps` for every document is what makes the witness admissible
    under section 4.4. The realised movement is compared to `eps` in ulps:
    with the default non-dyadic delta, `fl(s - eps)` differs from `s - eps`.
    """
    scores = [1.0, 0.5, 0.25, 0.0]
    witness = flip_witness(scores, [0, 1, 2, 3], 2)
    assert witness is not None
    perturbed, eps = witness

    moved = [i for i, (p, s) in enumerate(zip(perturbed, scores, strict=True)) if p != s]
    assert moved == [1, 2]

    realised = max(abs(p - s) for p, s in zip(perturbed, scores, strict=True))
    assert realised != eps, "the premise: a non-dyadic delta does not land exactly"
    assert abs(ulps_between(eps, realised)) <= 2.0, "and it lands within a rounding of it"


def test_a_rank_of_zero_is_refused_by_the_rank_guard_and_not_incidentally() -> None:
    """The rank guard, not the margin guard, owns `k = 0`.

    Without the rank guard, `k = 0` on a descending corpus pairs `order[-1]`
    with `order[0]` and the margin guard rejects the negative margin. The order
    here disagrees with the scores, so that margin is positive.
    """
    assert flip_witness([0.0, 1.0], [0, 1], 0) is None


def test_the_delta_floor_is_measured_at_one_rather_than_at_the_score() -> None:
    """The delta floor is measured at `max(|s|, 1.0)`, not at the score.

    Below a margin of about `1e-9` the ulp floor replaces the relative term. An
    ulp of a score below one is smaller than an ulp of one, so a floor at the
    score would shrink where the excess most needs to survive the addition.
    """
    tiny_margin = 2.0**-40
    scores = [0.5, 0.5 - tiny_margin, 0.0]
    witness = flip_witness(scores, [0, 1, 2], 1)
    assert witness is not None
    _, eps = witness

    excess = eps - tiny_margin / 2.0
    assert excess >= 4.0 * math.ulp(1.0), "the floor is taken at one"
    assert excess > 4.0 * math.ulp(0.5), "and not at the score, which is half as coarse"


def test_a_perturbation_landing_exactly_on_a_tie_is_not_a_witness() -> None:
    """A perturbation that lands exactly on a tie is not a witness.

    The comparison is strictly greater; with `delta = 0` a dyadic margin halves
    exactly and the two scores meet. A tie is not a reversal, and a flip radius
    derived from one would be a half-step short.
    """
    scores = [1.0, 0.5, 0.25, 0.0]
    order = [0, 1, 2, 3]

    assert flip_witness(scores, order, 1, delta=0.0) is None

    exactly_tied = 1.0 - 0.25, 0.5 + 0.25
    assert exactly_tied[0] == exactly_tied[1] == 0.75, "the two scores really do meet"


def test_the_smallest_delta_that_does_reverse_the_pair_is_a_witness() -> None:
    """The smallest reversing delta is accepted as a witness.

    This is the other side of the same comparison, so the refusal above is
    about the tie rather than about `delta` being small.
    """
    scores = [1.0, 0.5, 0.25, 0.0]
    order = [0, 1, 2, 3]

    witness = flip_witness(scores, order, 1, delta=0.125)
    assert witness is not None

    perturbed, eps = witness
    assert eps == 0.375, "margin / 2 plus the delta asked for"
    assert perturbed[1] > perturbed[0], "the pair is genuinely reversed"
    assert (perturbed[0], perturbed[1]) == (0.625, 0.875)
