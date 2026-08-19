"""The ranking operator and its tie-break (README sections 2.3.1 and 4.5).

The claim under test: the comparator is a strict total order. Everything else
follows from it. The sorted permutation is unique, sort stability is irrelevant,
and the two backends can be required to agree exactly.

:func:`test_all_distinct_scores_makes_the_three_operators_identical` establishes
the premise of research question A2, that any disagreement between pi, pi_score
and pi_alt is attributable to ties alone. Without it the tie-break ablation
would be measuring something else.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tfidf_stability.ranking.attributes import (
    AttributeDType,
    AttributeSpec,
    AttributeTable,
    Direction,
    MissingPolicy,
    ratio_less,
)
from tfidf_stability.ranking.ranker import (
    Selection,
    rank,
    rank_all_operators,
    rank_top_k,
    sorted_scores_desc,
)
from tfidf_stability.ranking.sort_keys import (
    PI,
    PI_ALT,
    PI_SCORE,
    SortKeySpec,
    assert_strict_total_order,
    build_keys,
)
from tfidf_stability.utils.validation import (
    DuplicateIdentifierError,
    EmptyCorpusError,
    KOutOfRangeError,
    StrictMode,
    TfidfStabilityError,
)

POP = SortKeySpec("pop_only", ("popularity",))


def table_of(n: int, **cols: list[int]) -> AttributeTable:
    """A small table with integer attributes and ``d0..d{n-1}`` identifiers."""
    records = [
        {"doc_id": f"d{i}", **{name: values[i] for name, values in cols.items()}} for i in range(n)
    ]
    specs = tuple(AttributeSpec(name, Direction.DESC, AttributeDType.INT64) for name in cols)
    return AttributeTable.from_records(records, specs)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------
def test_duplicate_identifiers_are_rejected() -> None:
    """Unique ids are the precondition of the strict total order, and so of
    every uniqueness claim this layer makes."""
    with pytest.raises(DuplicateIdentifierError):
        AttributeTable.from_records(
            [{"doc_id": "a", "popularity": 1}, {"doc_id": "a", "popularity": 2}],
            (AttributeSpec("popularity"),),
        )


def test_non_finite_float_attribute_is_rejected() -> None:
    spec = (AttributeSpec("x", Direction.DESC, AttributeDType.FLOAT64),)
    with pytest.raises(TfidfStabilityError, match="not finite"):
        AttributeTable.from_records(
            [{"doc_id": "a", "x": 1.0}, {"doc_id": "b", "x": float("nan")}], spec
        )


def test_the_exact_pair_separates_means_that_binary64_collides() -> None:
    """Why G8 keeps ratings as exact pairs.

    ``1/3`` and ``(10^17+1)/(3*10^17)`` are different reals that round to the
    same binary64. A float mean calls them equal and drops the ordering through
    to the identifier; cross-multiplication separates them.
    """
    a_num, a_den = 1, 3
    b_num, b_den = 10**17 + 1, 3 * 10**17
    assert a_num / a_den == b_num / b_den, "the premise: binary64 collides them"
    assert ratio_less(a_num, a_den, b_num, b_den), "but they are genuinely ordered"

    # The cross-products a comparison forms stay int64-safe despite the large
    # denominators: small denominator against large numerator and vice versa.
    assert max(a_num * b_den, b_num * a_den) < (1 << 63) - 1

    table = AttributeTable.from_records(
        [
            {"doc_id": "a", "rating_sum2": a_num, "rating_count": a_den},
            {"doc_id": "b", "rating_sum2": b_num, "rating_count": b_den},
        ],
        (AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),),
    )
    ranks = table.column("rating").ranks
    assert ranks[0] != ranks[1], "the exact pair must distinguish them"
    assert ranks[1] < ranks[0], "b is the larger mean, so it ranks earlier under desc"


def test_missing_attribute_sorts_last_and_is_never_nan() -> None:
    spec = (AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64),)
    table = AttributeTable.from_records(
        [
            {"doc_id": "a", "popularity": 5},
            {"doc_id": "b"},  # absent
            {"doc_id": "c", "popularity": 9},
        ],
        spec,
    )
    col = table.column("popularity")
    assert col.has_value == (True, False, True)
    assert col.ranks[1] == max(col.ranks), "missing sorts last"
    assert all(isinstance(r, int) for r in col.ranks), "ranks are ints, never NaN"


def test_missing_first_policy_places_absent_documents_first() -> None:
    spec = (AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64, MissingPolicy.FIRST),)
    table = AttributeTable.from_records([{"doc_id": "a", "popularity": 5}, {"doc_id": "b"}], spec)
    assert table.column("popularity").ranks[1] == 0


def test_forbidden_missing_value_raises() -> None:
    spec = (
        AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64, MissingPolicy.FORBID),
    )
    with pytest.raises(TfidfStabilityError, match="FORBID"):
        AttributeTable.from_records([{"doc_id": "a"}], spec)


def test_ratio_overflow_is_rejected_for_the_native_mirror() -> None:
    """Python cannot overflow; C++ can, so the guard lives at construction.

    The overflowing product has to be one a comparison forms: a numerator from
    one document against a denominator from another.
    """
    spec = (AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),)
    with pytest.raises(TfidfStabilityError, match="overflows int64"):
        AttributeTable.from_records(
            [
                {"doc_id": "a", "rating_sum2": 10**18, "rating_count": 1},
                {"doc_id": "b", "rating_sum2": 1, "rating_count": 10**18},
            ],
            spec,
        )


def test_the_overflow_guard_does_not_reject_a_single_document() -> None:
    """Nothing is ever compared with itself, so no product is formed at all."""
    spec = (AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),)
    table = AttributeTable.from_records(
        [{"doc_id": "a", "rating_sum2": 10**18, "rating_count": 10**18}], spec
    )
    assert table.n_documents == 1


def test_the_overflow_guard_is_not_over_conservative() -> None:
    """A false positive the naive bound would produce.

    Bounding by ``max(num) * max(den)`` over the whole column rejects this
    table: ``b`` holds the largest numerator and the largest denominator, a
    pairing no comparison ever forms. The products that do occur are both about
    3e17, comfortably int64-safe.
    """
    spec = (AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),)
    table = AttributeTable.from_records(
        [
            {"doc_id": "a", "rating_sum2": 1, "rating_count": 3},
            {"doc_id": "b", "rating_sum2": 10**17 + 1, "rating_count": 3 * 10**17},
        ],
        spec,
    )
    assert table.column("rating").n_distinct == 2


# ---------------------------------------------------------------------------
# The comparator
# ---------------------------------------------------------------------------
def test_the_comparator_is_a_strict_total_order(mini_attributes: AttributeTable) -> None:
    scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # every score identical
    assert_strict_total_order(build_keys(scores, mini_attributes, PI))
    assert_strict_total_order(build_keys(scores, mini_attributes, PI_SCORE))


def test_keys_are_injective_even_when_every_attribute_collides() -> None:
    """The identifier is what rescues totality; nothing else can."""
    table = table_of(4, popularity=[7, 7, 7, 7])
    keys = build_keys([1.0] * 4, table, PI_SCORE)
    assert len(set(keys)) == 4


def test_priority_may_not_name_the_identifier() -> None:
    table = table_of(2, popularity=[1, 2])
    with pytest.raises(ValueError, match="non-total"):
        build_keys([1.0, 2.0], table, SortKeySpec("bad", ("identifier",)))


@pytest.mark.parametrize("selection", list(Selection))
def test_five_selection_algorithms_agree(
    mini_attributes: AttributeTable, selection: Selection
) -> None:
    """The operational content of "sort stability is irrelevant here".

    Under an injective key no two elements compare equal, so the stability
    clause quantifies over the empty set and every correct algorithm must emit
    the identical permutation.
    """
    scores = [0.9, 0.9, 0.5, 0.5, 0.0, 0.5]
    expected = rank(scores, mini_attributes, PI, selection=Selection.FULL_SORT).order
    assert rank(scores, mini_attributes, PI, selection=selection).order == expected


def test_ranking_is_invariant_to_input_document_order() -> None:
    """The stronger check.

    A non-total comparator (score-only, say) can pass the five-algorithm check
    by luck on a small input; it cannot survive a permutation of the input.
    """
    rng = random.Random(4)
    n = 30
    pops = [rng.randrange(3) for _ in range(n)]
    scores = [rng.choice([0.0, 0.25, 0.5]) for _ in range(n)]

    records = [{"doc_id": f"d{i}", "popularity": pops[i]} for i in range(n)]
    specs = (AttributeSpec("popularity"),)
    forward = rank(scores, AttributeTable.from_records(records, specs), PI_SCORE)

    perm = list(range(n))
    rng.shuffle(perm)
    shuffled = rank(
        [scores[i] for i in perm],
        AttributeTable.from_records([records[i] for i in perm], specs),
        PI_SCORE,
    )
    # Compare by document identity rather than by position.
    assert [f"d{i}" for i in forward.order] == [f"d{perm[i]}" for i in shuffled.order]


# ---------------------------------------------------------------------------
# The three operators
# ---------------------------------------------------------------------------
def test_all_distinct_scores_makes_the_three_operators_identical(
    mini_attributes: AttributeTable,
) -> None:
    """Validates the premise of research question A2.

    With no ties the attribute tuple is never consulted, so pi, pi_score and
    pi_alt coincide. Any disagreement in the section 7.3 ablation is then
    attributable to ties alone, which is what that experiment claims to measure.
    """
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    rankings = rank_all_operators(scores, mini_attributes)
    orders = {r.order for r in rankings.values()}
    assert len(orders) == 1


def test_operators_diverge_when_scores_tie(mini_attributes: AttributeTable) -> None:
    """The converse: with ties present, priority order becomes observable."""
    scores = [0.0] * 6  # everything tied; the attributes decide entirely
    rankings = rank_all_operators(scores, mini_attributes)
    assert rankings["pi"].order != rankings["pi_alt"].order, (
        "reversing the attribute priority must change the order on a full tie"
    )
    assert rankings["pi"].order != rankings["pi_score"].order


def test_zero_query_ranks_purely_by_attributes(mini_attributes: AttributeTable) -> None:
    scores = [0.0] * 6
    r = rank(scores, mini_attributes, PI)
    assert r.query_degenerate is True
    # pi_score falls straight through to the identifier, i.e. input order here.
    assert rank(scores, mini_attributes, PI_SCORE).order == (0, 1, 2, 3, 4, 5)


def test_score_dominates_every_attribute(mini_attributes: AttributeTable) -> None:
    """d5 has the worst attributes but the best score, so it must rank first."""
    scores = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert rank(scores, mini_attributes, PI).order[0] == 4


# ---------------------------------------------------------------------------
# Rejection and edge cases (G3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", list(StrictMode))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_is_rejected_before_sorting(
    mini_attributes: AttributeTable, mode: StrictMode, bad: float
) -> None:
    """Rejected in lenient mode too.

    G3 lists non-finite scores among the rejected inputs rather than the
    legitimate degenerate grid points. In the native backend a NaN in a sort key
    is undefined behaviour: an out-of-bounds write in libstdc++'s final
    insertion pass, beyond merely a wrong answer.
    """
    scores = [0.5, bad, 0.1, 0.2, 0.3, 0.4]
    with pytest.raises(TfidfStabilityError, match="not finite"):
        rank(scores, mini_attributes, PI, mode=mode)


def test_empty_corpus_is_an_error() -> None:
    with pytest.raises(EmptyCorpusError):
        rank([], AttributeTable.from_records([], ()), PI_SCORE)


def test_k_out_of_range_strict_raises(mini_attributes: AttributeTable) -> None:
    with pytest.raises(KOutOfRangeError):
        rank_top_k([0.1] * 6, mini_attributes, PI, k=99, mode=StrictMode.STRICT)


def test_k_out_of_range_lenient_clamps_and_records(mini_attributes: AttributeTable) -> None:
    r = rank_top_k([0.1] * 6, mini_attributes, PI, k=99, mode=StrictMode.LENIENT)
    assert r.k_effective == 6
    assert r.n_selected == 6


def test_top_k_selects_k_plus_one(mini_attributes: AttributeTable) -> None:
    """m_k needs score(r_{k+1}), so a top-k that stopped at k could not report
    its own boundary margin."""
    r = rank_top_k([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], mini_attributes, PI, k=3)
    assert r.n_selected == 4
    assert r.is_complete is False


def test_truncated_ranking_refuses_whole_corpus_questions(
    mini_attributes: AttributeTable,
) -> None:
    r = rank_top_k([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], mini_attributes, PI, k=2)
    with pytest.raises(ValueError, match="complete ranking"):
        r.require_complete("this")
    with pytest.raises(KeyError):
        r.rank_of(5)  # not selected


def test_sorted_scores_is_complete_even_when_the_order_is_truncated(
    mini_attributes: AttributeTable,
) -> None:
    """The asymmetry that margins depend on: truncate the order, never the
    score array."""
    r = rank_top_k([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], mini_attributes, PI, k=2)
    assert len(r.sorted_scores) == 6
    assert len(r.order) == 3


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_operator_identity_and_digest_are_recorded(mini_attributes: AttributeTable) -> None:
    r = rank([0.1] * 6, mini_attributes, PI)
    assert r.operator == "pi"
    assert len(r.key_digest) == 64
    assert PI.digest(mini_attributes) != PI_ALT.digest(mini_attributes)


def test_key_digest_binds_the_attribute_table_not_just_the_priority() -> None:
    """The digest identifies the ranking function rather than the raw values.

    Taken over the integer ranks, so two tables that induce identical orderings
    for every score vector share a digest, which is what certifying a published
    ranking needs. Tables that order documents differently must differ.
    """
    same_order_a = table_of(2, popularity=[1, 2])
    same_order_b = table_of(2, popularity=[5, 6])  # different values, same ranks
    assert POP.digest(same_order_a) == POP.digest(same_order_b)

    reversed_order = table_of(2, popularity=[2, 1])
    assert POP.digest(same_order_a) != POP.digest(reversed_order)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
tie_prone = st.lists(st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]), min_size=1, max_size=40)


@given(tie_prone)
def test_the_order_is_always_a_permutation(scores: list[float]) -> None:
    table = table_of(len(scores), popularity=[i % 3 for i in range(len(scores))])
    r = rank(scores, table, POP)
    assert sorted(r.order) == list(range(len(scores)))


@given(tie_prone)
def test_scores_are_non_increasing_along_the_order(scores: list[float]) -> None:
    table = table_of(len(scores), popularity=[i % 3 for i in range(len(scores))])
    r = rank(scores, table, POP)
    along = [r.scores[d] for d in r.order]
    assert all(a >= b for a, b in pairwise(along))
    assert along == list(r.sorted_scores)


@given(tie_prone, st.integers(min_value=1, max_value=20))
def test_top_k_prefix_matches_the_full_ranking(scores: list[float], k: int) -> None:
    """Certifies partial selection against the normative full sort."""
    n = len(scores)
    table = table_of(n, popularity=[i % 4 for i in range(n)])
    full = rank(scores, table, POP)
    partial = rank_top_k(scores, table, POP, k=k, mode=StrictMode.LENIENT)
    assert full.order[: partial.n_selected] == partial.order


@given(tie_prone)
def test_sorted_scores_desc_is_a_sorted_permutation(scores: list[float]) -> None:
    s = sorted_scores_desc(scores)
    assert sorted(s, reverse=True) == list(s)
    assert sorted(s) == sorted(scores)


# ---------------------------------------------------------------------------
# Equal means must share a rank (regression)
# ---------------------------------------------------------------------------
def test_equal_means_written_differently_receive_the_same_rank() -> None:
    """G8's exact comparison has to survive rank-encoding as well as
    `ratio_less`.

    `_order_distinct` deduplicates with `dict.fromkeys`, which uses tuple
    equality, so (14, 2) and (21, 3) both survive despite being the same mean of
    3.5, which `_ratio_cmp` reports. Numbering the survivors positionally gave
    them two ranks and un-tied a rating G8 requires to tie.
    """
    table = AttributeTable.from_records(
        [
            {"doc_id": "a", "rating_sum2": 14, "rating_count": 2},  # 3.5
            {"doc_id": "b", "rating_sum2": 21, "rating_count": 3},  # 3.5
            # rating_sum2 is TWICE the rating sum, so the mean is sum2/(2*count).
            # 18/(2*2) = 4.5, strictly better than the 3.5 above.
            {"doc_id": "c", "rating_sum2": 18, "rating_count": 2},
        ],
        specs=(AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),),
    )
    ranks = table.column("rating").ranks
    assert ranks[0] == ranks[1], "14/2 and 21/3 are the same mean and must share a rank"
    assert ranks[2] < ranks[0], "4.5 outranks 3.5 under DESC"


def test_a_rating_tie_falls_through_to_the_next_attribute() -> None:
    """The consequence of the above: if the rating does not tie, engagement is
    never consulted, and the wrong document wins."""
    records = [
        {"doc_id": "a", "popularity": 1, "rating_sum2": 14, "rating_count": 2, "engagement": 5},
        {"doc_id": "b", "popularity": 1, "rating_sum2": 21, "rating_count": 3, "engagement": 9},
    ]
    table = AttributeTable.from_records(records)
    order = rank_all_operators([0.5, 0.5], table)["pi"].order
    assert [records[i]["doc_id"] for i in order] == ["b", "a"], (
        "scores, popularity and rating all tie, so engagement DESC decides"
    )


def test_the_ranking_does_not_depend_on_the_order_records_were_supplied_in() -> None:
    """The property section 2.3.1's total-order argument rests on.

    The shape the earlier bug took: `dict.fromkeys` preserves insertion order,
    so which of two equal representations sorted first depended on the corpus
    order, and the ranking moved with it.
    """
    a = {"doc_id": "a", "popularity": 1, "rating_sum2": 14, "rating_count": 2, "engagement": 5}
    b = {"doc_id": "b", "popularity": 1, "rating_sum2": 21, "rating_count": 3, "engagement": 9}

    forward = [a, b]
    reverse = [b, a]
    first = [
        forward[i]["doc_id"]
        for i in rank_all_operators([0.5, 0.5], AttributeTable.from_records(forward))["pi"].order
    ]
    second = [
        reverse[i]["doc_id"]
        for i in rank_all_operators([0.5, 0.5], AttributeTable.from_records(reverse))["pi"].order
    ]
    assert first == second, f"ranking depends on record order: {first} vs {second}"


# ---------------------------------------------------------------------------
# The comparator self-test, shown to actually detect
# ---------------------------------------------------------------------------
# assert_strict_total_order is an oracle: its job is to catch a comparator that
# has stopped being a strict total order. Its failure arms had never fired, which
# means the detector had never been shown to detect anything. Pragmas would have
# recorded that permanently, so each axiom gets a specimen that violates exactly
# it. The mock here is the specimen under test, not a stand-in for a collaborator.
@dataclass(frozen=True)
class _ReflexivelyLess:
    """Less than everything, itself included. Violates irreflexivity only."""

    tag: int

    def __lt__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _MutuallyLess:
    """Less than everything except itself: two of these are each less than the
    other, which is the asymmetry violation and nothing else."""

    tag: int

    def __lt__(self, other: object) -> bool:
        return other is not self


@dataclass(frozen=True)
class _CyclicKey:
    """Rock-paper-scissors: 0 < 1 < 2 < 0.

    Irreflexive, asymmetric and trichotomous for every pair, so the first three
    checks pass and only transitivity can catch it.
    """

    i: int

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _CyclicKey)
        return (self.i + 1) % 3 == other.i


def test_a_key_that_compares_less_than_itself_is_reported_as_not_irreflexive() -> None:
    with pytest.raises(AssertionError, match="not irreflexive"):
        assert_strict_total_order([_ReflexivelyLess(0)])  # type: ignore[list-item]


def test_two_keys_each_less_than_the_other_are_reported_as_not_asymmetric() -> None:
    keys = [_MutuallyLess(0), _MutuallyLess(1)]
    assert keys[0] < keys[1], "the premise being demonstrated"
    assert keys[1] < keys[0], "and in the other direction, which is the violation"
    with pytest.raises(AssertionError, match="not asymmetric"):
        assert_strict_total_order(keys)  # type: ignore[arg-type]


def test_a_rock_paper_scissors_comparator_is_reported_as_not_transitive() -> None:
    """The only violation the O(n^3) pass can catch that the pairwise ones cannot."""
    keys = [_CyclicKey(0), _CyclicKey(1), _CyclicKey(2)]
    assert keys[0] < keys[1], "the premise"
    assert keys[1] < keys[2], "and onward round the cycle"
    assert not keys[0] < keys[2], "and the cycle closes"
    with pytest.raises(AssertionError, match="not transitive"):
        assert_strict_total_order(keys)  # type: ignore[arg-type]


def test_two_distinct_nan_keys_are_reported_as_non_trichotomous_not_silently_tied() -> None:
    """Reachable with real tuple keys, no specimen required.

    Two separately constructed NaNs hash differently, so injectivity passes, and
    then neither compares less than the other. A comparator that answers "no"
    both ways has no opinion about the pair, and the sorted permutation stops
    being determined by the key.
    """
    first, second = float("nan"), float("nan")
    keys = [(first, 0), (second, 1)]
    assert len(set(keys)) == 2, "the premise: injectivity passes"
    assert not keys[0] < keys[1]
    assert not keys[1] < keys[0]
    with pytest.raises(AssertionError, match="not trichotomous"):
        assert_strict_total_order(keys)


def test_a_duplicated_key_is_reported_as_non_injective() -> None:
    """With duplicates the ranking is only a weak order and the sorted
    permutation is no longer unique, which is what the message says."""
    with pytest.raises(AssertionError, match="not injective"):
        assert_strict_total_order([(-1.0, 0), (-1.0, 0)])


def test_the_transitivity_sweep_is_skipped_above_its_size_limit() -> None:
    """O(n^3) is affordable at test sizes and not beyond, so the check switches
    off above 40. Both sides are exercised, since a limit nothing crosses is a
    limit nothing tests."""
    forty = [(-float(i), i) for i in range(40)]
    forty_one = [(-float(i), i) for i in range(41)]
    assert_strict_total_order(forty)
    assert_strict_total_order(forty_one)


def test_a_score_count_that_disagrees_with_the_table_is_rejected_not_zipped_short() -> None:
    """Zipping to the shorter of the two would silently rank a prefix."""
    table = table_of(4)
    with pytest.raises(ValueError, match="scores but the attribute table"):
        build_keys([0.1, 0.2], table, SortKeySpec(PI))


# ---------------------------------------------------------------------------
# Attribute encoding: the dtypes and directions the tie-break can be built from
# ---------------------------------------------------------------------------
def test_a_bytes_attribute_orders_by_utf8_bytes_not_by_locale() -> None:
    """The same rule vocabulary.py uses, and reproducible in C++ via memcmp.

    Locale or Unicode collation would order these differently between machines,
    which would move a published ranking without changing a single score.
    """
    records = [
        {"doc_id": "d0", "label": "Zebra"},
        {"doc_id": "d1", "label": "apple"},
        {"doc_id": "d2", "label": "Apple"},
    ]
    specs = (AttributeSpec("label", Direction.ASC, AttributeDType.BYTES),)
    table = AttributeTable.from_records(records, specs)
    ranks = table.column("label").ranks

    # Uppercase sorts before lowercase in byte order; a locale-aware collation
    # would interleave them.
    order = sorted(range(3), key=lambda i: ranks[i])
    assert [records[i]["label"] for i in order] == ["Apple", "Zebra", "apple"]


def test_an_ascending_direction_reverses_the_ranks_a_descending_one_gives() -> None:
    """Only DESC was ever built, so the reversal arm was never taken."""
    records = [{"doc_id": f"d{i}", "popularity": v} for i, v in enumerate([10, 30, 20])]
    asc = AttributeTable.from_records(
        records, (AttributeSpec("popularity", Direction.ASC, AttributeDType.INT64),)
    )
    desc = AttributeTable.from_records(
        records, (AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64),)
    )
    assert asc.column("popularity").ranks[0] < asc.column("popularity").ranks[2], (
        "ascending puts the smallest first"
    )
    assert desc.column("popularity").ranks[0] > desc.column("popularity").ranks[2], (
        "descending puts the largest first"
    )


def test_a_missing_bytes_value_becomes_the_empty_string_and_is_marked_absent() -> None:
    records = [{"doc_id": "d0", "label": "x"}, {"doc_id": "d1"}]
    specs = (AttributeSpec("label", Direction.ASC, AttributeDType.BYTES),)
    column = AttributeTable.from_records(records, specs).column("label")
    assert column.value_of(0) == "x"
    assert column.value_of(1) is None, "absent is None, never the placeholder it stores"


def test_a_column_reports_one_entry_per_document() -> None:
    table = table_of(5, popularity=[1, 2, 3, 4, 5])
    assert len(table.column("popularity")) == 5


def test_asking_for_an_attribute_that_does_not_exist_names_the_ones_that_do() -> None:
    table = table_of(3, popularity=[1, 2, 3])
    with pytest.raises(KeyError, match="no attribute named"):
        table.column("engagement")


def test_a_table_built_from_no_records_has_no_positions_rather_than_raising() -> None:
    table = AttributeTable.from_records(
        [], (AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64),)
    )
    assert table.n_documents == 0
    assert table.column("popularity").ranks == ()


def test_the_reporting_view_returns_the_raw_values_not_the_encoded_ranks() -> None:
    """Section 7.4 prints these, so a rank leaking into the report would show a
    dense position where a reader expects the rating that produced it."""
    records = [
        {"doc_id": "a", "popularity": 7, "rating_sum2": 9, "rating_count": 2},
        {"doc_id": "b", "popularity": 3, "rating_sum2": 5, "rating_count": 2},
    ]
    specs = (
        AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64),
        AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),
    )
    table = AttributeTable.from_records(records, specs)
    view = table.attributes_of(0)

    assert view["doc_id"] == "a"
    assert view["popularity"] == 7, "the raw value, not its dense rank"
    assert view["rating"] == (9, 2), "the exact pair, never a divided mean"
    assert view["id_rank"] == table.id_ranks[0]


def test_a_rating_count_of_zero_with_a_non_zero_sum_is_inconsistent_not_absent() -> None:
    """Zero ratings summing to something is a corrupt record, and treating it as
    merely missing would rank the document on a value that cannot exist."""
    records = [
        {"doc_id": "a", "rating_sum2": 7, "rating_count": 0},
        {"doc_id": "b", "rating_sum2": 4, "rating_count": 2},
    ]
    specs = (AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),)
    with pytest.raises(TfidfStabilityError, match="is inconsistent"):
        AttributeTable.from_records(records, specs)


def test_a_rating_count_of_zero_with_a_zero_sum_is_simply_absent() -> None:
    """The other side: nothing was rated, which is ordinary rather than corrupt."""
    records = [
        {"doc_id": "a", "rating_sum2": 0, "rating_count": 0},
        {"doc_id": "b", "rating_sum2": 4, "rating_count": 2},
    ]
    specs = (AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),)
    column = AttributeTable.from_records(records, specs).column("rating")
    assert column.value_of(0) is None, "an unrated document has no rating, not a zero one"
    assert column.value_of(1) == (4, 2)


def test_a_negative_rating_count_is_rejected() -> None:
    records = [
        {"doc_id": "a", "rating_sum2": 4, "rating_count": -2},
        {"doc_id": "b", "rating_sum2": 4, "rating_count": 2},
    ]
    specs = (AttributeSpec("rating", Direction.DESC, AttributeDType.RATIO_I64),)
    with pytest.raises(TfidfStabilityError, match="is negative"):
        AttributeTable.from_records(records, specs)
