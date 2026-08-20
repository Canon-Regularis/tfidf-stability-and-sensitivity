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

import math
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
    _ratio_cmp,
    check_ratio_fits_int64,
    ratio_less,
)
from tfidf_stability.ranking.ranker import (
    Ranking,
    Selection,
    _select,
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


#: A priority naming only `popularity`, for tests about key shape rather than
#: about the normative operator.
_POP_ONLY = SortKeySpec(name="pop_only", priority=("popularity",))


def _two_document_table() -> AttributeTable:
    """The smallest table carrying every attribute `PI` names.

    All three of `popularity`, `rating` and `engagement` are present because
    `build_keys` resolves the whole priority; a table with only one of them
    raises a `KeyError` that is about the fixture rather than about the code.

    Local to this file by house convention: a shared builder would let a change
    made for one suite silently alter another's fixtures.
    """
    return AttributeTable.from_records(
        [
            {"doc_id": "a", "popularity": 3, "rating_sum2": 9, "rating_count": 2, "engagement": 5},
            {"doc_id": "b", "popularity": 1, "rating_sum2": 7, "rating_count": 2, "engagement": 2},
        ]
    )


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


@pytest.mark.property
@given(tie_prone)
def test_the_order_is_always_a_permutation(scores: list[float]) -> None:
    table = table_of(len(scores), popularity=[i % 3 for i in range(len(scores))])
    r = rank(scores, table, POP)
    assert sorted(r.order) == list(range(len(scores)))


@pytest.mark.property
@given(tie_prone)
def test_scores_are_non_increasing_along_the_order(scores: list[float]) -> None:
    table = table_of(len(scores), popularity=[i % 3 for i in range(len(scores))])
    r = rank(scores, table, POP)
    along = [r.scores[d] for d in r.order]
    assert all(a >= b for a, b in pairwise(along))
    assert along == list(r.sorted_scores)


@pytest.mark.property
@given(tie_prone, st.integers(min_value=1, max_value=20))
def test_top_k_prefix_matches_the_full_ranking(scores: list[float], k: int) -> None:
    """Certifies partial selection against the normative full sort."""
    n = len(scores)
    table = table_of(n, popularity=[i % 4 for i in range(n)])
    full = rank(scores, table, POP)
    partial = rank_top_k(scores, table, POP, k=k, mode=StrictMode.LENIENT)
    assert full.order[: partial.n_selected] == partial.order


@pytest.mark.property
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


# ---------------------------------------------------------------------------
# A truncated ranking answers only what it can
# ---------------------------------------------------------------------------
# rank_top_k selects a prefix, so the object it returns knows the whole score
# vector but only part of the order. Every accessor below has to say which of the
# two it is answering from, because a question about the corpus answered from the
# prefix would be wrong rather than merely unavailable.
def test_a_complete_ranking_answers_whole_corpus_questions() -> None:
    table = table_of(5, popularity=[5, 4, 3, 2, 1])
    ranking = rank([0.5, 0.4, 0.3, 0.2, 0.1], table, POP)
    assert ranking.is_complete
    ranking.require_complete("a test")  # must not raise
    assert ranking.top_k(5) == ranking.order


def test_a_truncated_ranking_refuses_a_whole_corpus_question() -> None:
    table = table_of(6, popularity=[6, 5, 4, 3, 2, 1])
    ranking = rank_top_k([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], table, POP, k=2)
    assert not ranking.is_complete
    with pytest.raises(ValueError, match="needs the complete ranking"):
        ranking.require_complete("an ordering distance")


def test_asking_a_truncated_ranking_for_more_than_it_selected_is_refused() -> None:
    """Returning a short prefix instead would silently answer a different
    question from the one asked."""
    table = table_of(6, popularity=[6, 5, 4, 3, 2, 1])
    ranking = rank_top_k([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], table, POP, k=2)
    assert len(ranking.top_k(2)) == 2
    with pytest.raises(ValueError, match="but only"):
        ranking.top_k(5)


def test_the_score_at_a_rank_is_available_even_where_the_order_is_truncated() -> None:
    """sorted_scores is always the whole corpus; only the order is cut. That
    asymmetry is what lets margins be read at every k after a top-k selection."""
    scores = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    table = table_of(6, popularity=[6, 5, 4, 3, 2, 1])
    ranking = rank_top_k(scores, table, POP, k=2)

    assert ranking.score_at_rank(1) == 0.6
    assert ranking.score_at_rank(6) == 0.1, "readable past the truncation point"


@pytest.mark.parametrize("bad", [0, 7, -1])
def test_a_rank_outside_one_to_n_is_refused(bad: int) -> None:
    """Ranks are 1-indexed in the public API and 0-indexed underneath, so both
    ends are checked rather than trusted to the array."""
    table = table_of(6, popularity=[6, 5, 4, 3, 2, 1])
    ranking = rank([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], table, POP)
    with pytest.raises(IndexError, match="out of range"):
        ranking.score_at_rank(bad)


def test_the_rank_of_an_unselected_document_says_the_ranking_was_truncated() -> None:
    """Two different absences: not in the corpus, and not in the selected
    prefix. The message distinguishes them so a caller can tell which."""
    table = table_of(6, popularity=[6, 5, 4, 3, 2, 1])
    truncated = rank_top_k([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], table, POP, k=2)
    with pytest.raises(KeyError, match="truncated"):
        truncated.rank_of(5)

    complete = rank([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], table, POP)
    assert complete.rank_of(5) == 6, "the same document is answerable when nothing was cut"


def test_restricting_to_a_subset_keeps_the_ranking_order_not_the_argument_order() -> None:
    """Used to compare two operators over a tie group, so the result has to be
    ordered by the ranking rather than by however the caller listed them."""
    table = table_of(5, popularity=[5, 4, 3, 2, 1])
    ranking = rank([0.5, 0.4, 0.3, 0.2, 0.1], table, POP)
    assert ranking.order_within([3, 0, 2]) == (0, 2, 3)
    assert ranking.order_within([]) == ()
    assert ranking.order_within([99]) == (), "a document outside the ranking contributes nothing"


def test_ranking_a_top_k_of_an_empty_corpus_is_refused_like_ranking_one() -> None:
    """Both entry points guard it: the empty case must not reach resolve_k and
    come back as a k error, which would name the wrong cause."""
    table = AttributeTable.from_records(
        [], (AttributeSpec("popularity", Direction.DESC, AttributeDType.INT64),)
    )
    with pytest.raises(EmptyCorpusError, match="empty corpus"):
        rank_top_k([], table, POP, k=1)


# ---------------------------------------------------------------------------
# ratio_less: the preconditions the column invariant guarantees
# ---------------------------------------------------------------------------
# Both denominators must be strictly positive. Nothing checks it, because a zero
# denominator means "no rating", is carried by the `has_value` bit, and never
# reaches here. Pinned rather than guarded: this is the innermost comparison of
# every tie-break, and a check would run on every pair of every sort.
def test_a_zero_denominator_gives_a_defined_but_meaningless_answer() -> None:
    """No `ZeroDivisionError` -- cross-multiplication never divides. It returns
    a plain `False`, which is why the presence bit has to keep such rows out
    rather than relying on this to complain."""
    assert ratio_less(1, 0, 1, 1) is False
    assert ratio_less(0, 0, 0, 0) is False


def test_a_negative_denominator_inverts_the_comparison() -> None:
    """Cross-multiplying by a negative flips the inequality, so the answer is
    not merely undefined but confidently wrong. `1/-2` really is less than
    `1/1`, and this reports otherwise."""
    assert (1 / -2) < (1 / 1), "the true ordering"
    assert ratio_less(1, -2, 1, 1) is False, "and the one cross-multiplication gives"


def test_the_comparison_is_exact_where_binary64_would_collide() -> None:
    """The reason G8 stores `(2*sum, count)` rather than a mean: two ratios a
    single ulp apart as floats are ordered exactly here."""
    assert ratio_less(1, 3, 100_000_000_000_000_001, 300_000_000_000_000_000)
    assert (1 / 3) == (100_000_000_000_000_001 / 300_000_000_000_000_000), "collide as floats"


# ---------------------------------------------------------------------------
# check_ratio_fits_int64: exactly where it starts refusing
# ---------------------------------------------------------------------------
def test_a_product_of_exactly_int64_max_is_admissible() -> None:
    """The bound is `> _INT64_MAX`, so the largest representable product is
    still representable. One past it is not, and the two differ by one."""
    largest = (1 << 63) - 1
    check_ratio_fits_int64([(largest, 1), (1, 1)], "rating")

    with pytest.raises(TfidfStabilityError, match="overflows int64"):
        check_ratio_fits_int64([(largest, 1), (1, 2)], "rating")


def test_the_overflow_message_names_both_factors_and_their_product() -> None:
    """A reader hitting this has to decide whether to rescale the numerators or
    the denominators, and cannot without knowing which pair collided."""
    largest = (1 << 63) - 1
    with pytest.raises(
        TfidfStabilityError, match=f"a numerator of {largest} against a denominator of 2"
    ):
        check_ratio_fits_int64([(largest, 1), (1, 2)], "rating")


def test_the_guard_says_which_attribute_it_was_checking() -> None:
    """`what` is threaded through so a corpus with several ratio columns names
    the one that overflowed."""
    with pytest.raises(TfidfStabilityError, match=r"^engagement:"):
        check_ratio_fits_int64([((1 << 63), 1), (1, 2)], "engagement")


@pytest.mark.parametrize("pairs", [[], [(2**62, 2**62)]])
def test_a_column_too_short_to_compare_is_never_refused(pairs: list[tuple[int, int]]) -> None:
    """The products that occur are `num_i * den_j` for `i != j`, and with fewer
    than two rows there are none. A single document whose own product would
    overflow is therefore admissible -- it is never compared with itself."""
    check_ratio_fits_int64(pairs, "rating")


# ---------------------------------------------------------------------------
# AttributeTable accessors
# ---------------------------------------------------------------------------
def test_asking_for_no_attributes_at_all_gives_no_rank_rows() -> None:
    """A ranking on the identifier alone is legitimate -- it is what `pi_score`
    reduces to once the scores are distinct."""
    table = _two_document_table()
    assert table.rank_matrix(()) == ()


def test_naming_an_attribute_twice_returns_its_row_twice() -> None:
    """The matrix is positional against the priority, so a repeated name is a
    repeated tie-break level. Degenerate -- the second can never break a tie the
    first did not -- but well-defined, and de-duplicating would silently change
    the priority a manifest recorded."""
    table = _two_document_table()
    rows = table.rank_matrix(("popularity", "popularity"))
    assert len(rows) == 2
    assert rows[0] == rows[1]


def test_the_reporting_view_of_a_negative_index_is_the_document_from_the_end() -> None:
    """Python indexing, not a bounds check. `attributes_of(-1)` returns the last
    document rather than raising, so a caller that computed an index badly gets
    a plausible row for the wrong document.

    Pinned as the trap it is: every other rank-taking entry point in the package
    (`ball`, `chain_of`, `score_at_rank`) refuses a negative index, and this one
    does not.
    """
    table = _two_document_table()
    assert table.attributes_of(-1)["doc_id"] == table.doc_ids[-1]
    assert table.attributes_of(0)["doc_id"] == table.doc_ids[0]


def test_a_column_value_past_the_end_is_an_index_error() -> None:
    """Unlike `attributes_of`, the column indexes a tuple directly, so it does
    raise. The inconsistency is the point of stating both."""
    with pytest.raises(IndexError, match="tuple index out of range"):
        _two_document_table().column("popularity").value_of(99)


def test_an_empty_table_reports_no_documents_rather_than_refusing() -> None:
    """A corpus can be filtered to nothing by a query protocol, and the table is
    built before anyone asks whether it is rankable. `rank` is where an empty
    corpus becomes an error (G17)."""
    empty = AttributeTable.from_records([], specs=(AttributeSpec("popularity"),))

    assert empty.n_documents == 0
    assert empty.id_ranks == ()
    assert empty.rank_matrix(("popularity",)) == ((),)
    assert len(empty.digest()) == 64


# ---------------------------------------------------------------------------
# from_records: identifiers
# ---------------------------------------------------------------------------
def test_a_record_without_the_identifier_field_is_a_key_error() -> None:
    """Not a `DuplicateIdentifierError`: the identifier is absent, not repeated,
    and conflating the two would send a reader looking for a collision that does
    not exist."""
    with pytest.raises(KeyError, match="doc_id"):
        AttributeTable.from_records([{"popularity": 1}], specs=(AttributeSpec("popularity"),))


def test_two_records_with_a_null_identifier_collide_on_its_rendering() -> None:
    """Identifiers are stringified, so two `None`s both become `'None'` and are
    a duplicate. Better than silently ranking two documents under one key, which
    is what the strict total order cannot survive."""
    with pytest.raises(DuplicateIdentifierError, match="'None' appears at positions 0 and 1"):
        AttributeTable.from_records(
            [{"doc_id": None}, {"doc_id": None}], specs=(AttributeSpec("popularity"),)
        )


def test_the_identifier_ranks_are_a_bijection_onto_the_positions() -> None:
    """What makes the sort key injective, and therefore the sorted permutation
    unique and sort stability irrelevant."""
    table = AttributeTable.from_records(
        [{"doc_id": name, "popularity": 1} for name in ("delta", "alpha", "charlie", "bravo")],
        specs=(AttributeSpec("popularity"),),
    )
    assert sorted(table.id_ranks) == list(range(4))


def test_the_identifier_ranks_follow_utf8_byte_order_not_the_input_order() -> None:
    """Byte order rather than a locale collation, so the ranking is the same on
    every machine."""
    names = ["b", "A", "a", "B"]
    table = AttributeTable.from_records(
        [{"doc_id": n, "popularity": 1} for n in names], specs=(AttributeSpec("popularity"),)
    )
    by_rank = [n for _, n in sorted(zip(table.id_ranks, names, strict=True))]
    assert by_rank == sorted(names, key=lambda s: s.encode("utf-8"))


# ---------------------------------------------------------------------------
# _extract: the type coercions, and where they leak
# ---------------------------------------------------------------------------
def test_a_value_that_cannot_be_coerced_leaks_a_bare_value_error() -> None:
    """`float("abc")` raises `ValueError`, not a package error, and nothing
    catches it. Pinned rather than wrapped: a non-numeric value in a numeric
    column is a defect in the corpus builder, not a corpus shape this package
    should describe.
    """
    with pytest.raises(ValueError, match="could not convert string to float"):
        AttributeTable.from_records(
            [{"doc_id": "a", "f": "abc"}, {"doc_id": "b", "f": 1.0}],
            specs=(AttributeSpec("f", dtype=AttributeDType.FLOAT64),),
        )


@pytest.mark.parametrize(
    ("dtype", "raw", "expected"),
    [
        (AttributeDType.INT64, "7", 7),
        (AttributeDType.INT64, 7.9, 7),
        (AttributeDType.FLOAT64, "1.5", 1.5),
        (AttributeDType.FLOAT64, 2, 2.0),
        (AttributeDType.BYTES, 7, "7"),
    ],
)
def test_a_present_value_is_coerced_to_the_declared_type(
    dtype: AttributeDType, raw: object, expected: object
) -> None:
    """The declared type wins over what the record happened to carry. Note the
    INT64 case truncates rather than rounding, so `7.9` is 7 -- which is what a
    rank-encoded column needs, and worth stating."""
    table = AttributeTable.from_records(
        [{"doc_id": "a", "v": raw}, {"doc_id": "b", "v": raw}],
        specs=(AttributeSpec("v", dtype=dtype),),
    )
    assert table.column("v").value_of(0) == expected


def test_a_falsy_present_value_is_not_an_absent_one() -> None:
    """`0` and `""` are values. Testing presence by truthiness would mark the
    least popular document as unrated, which the missing policy then sorts
    somewhere else entirely."""
    table = AttributeTable.from_records(
        [{"doc_id": "a", "popularity": 0}, {"doc_id": "b"}],
        specs=(AttributeSpec("popularity"),),
    )
    column = table.column("popularity")

    assert column.has_value == (True, False)
    assert column.value_of(0) == 0


def test_a_missing_first_and_a_missing_last_column_agree_when_nothing_is_present() -> None:
    """With no distinct present values there is no ordering for the absent ones
    to sit before or after, so the two policies coincide. A rank scheme that
    reserved a slot per policy would disagree here."""
    records = [{"doc_id": "a"}, {"doc_id": "b"}]
    first = AttributeTable.from_records(
        records, specs=(AttributeSpec("p", missing_policy=MissingPolicy.FIRST),)
    )
    last = AttributeTable.from_records(
        records, specs=(AttributeSpec("p", missing_policy=MissingPolicy.LAST),)
    )
    assert first.column("p").ranks == last.column("p").ranks


# ---------------------------------------------------------------------------
# digest: what it is and is not sensitive to
# ---------------------------------------------------------------------------
def _digest_with(**overrides: object) -> str:
    spec = AttributeSpec(
        name=str(overrides.get("name", "popularity")),
        direction=overrides.get("direction", Direction.DESC),  # type: ignore[arg-type]
        dtype=overrides.get("dtype", AttributeDType.INT64),  # type: ignore[arg-type]
        missing_policy=overrides.get("policy", MissingPolicy.LAST),  # type: ignore[arg-type]
    )
    records = overrides.get(
        "records", [{"doc_id": "a", "popularity": 3}, {"doc_id": "b", "popularity": 1}]
    )
    return AttributeTable.from_records(records, specs=(spec,)).digest()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("label", "override"),
    [
        ("the direction", {"direction": Direction.ASC}),
        ("the declared type", {"dtype": AttributeDType.FLOAT64}),
        ("the missing policy", {"policy": MissingPolicy.FIRST}),
        ("the attribute name", {"name": "engagement"}),
    ],
)
def test_anything_that_changes_the_ordering_changes_the_digest(
    label: str, override: dict[str, object]
) -> None:
    """The digest goes into the run manifest as the identity of the tie-break
    data. Two tables with the same digest must induce the same ranking, so every
    field that can move a rank has to move it."""
    assert _digest_with(**override) != _digest_with(), label


def test_raw_values_that_leave_the_ranks_alone_leave_the_digest_alone() -> None:
    """Taken over the ranks rather than the raw values, deliberately: the ranks
    determine the ordering. Rescaling a popularity column by a constant is not a
    different tie-break, and a digest that changed would force a rerun for
    nothing.
    """
    scaled = [{"doc_id": "a", "popularity": 300}, {"doc_id": "b", "popularity": 100}]
    assert _digest_with(records=scaled) == _digest_with()


def test_reordering_the_records_changes_the_digest_but_not_the_ranking() -> None:
    """The digest is positional; the ranking is not. Both are deliberate.

    `ranks` is indexed by document position, so supplying the same corpus in a
    different order permutes the tuple and the digest changes -- even though the
    per-document mapping, and therefore every ranking built from it, is
    identical. The digest identifies *this table*, not the corpus in the
    abstract, which is what makes it usable as a manifest key for a run that
    also recorded the corpus order.

    Stated in both directions because the invariance is easy to assume from the
    neighbouring test that reordering does not change the ranking.
    """
    forwards = [{"doc_id": "a", "popularity": 3}, {"doc_id": "b", "popularity": 1}]
    backwards = list(reversed(forwards))
    specs = (AttributeSpec("popularity"),)

    first = AttributeTable.from_records(forwards, specs=specs)
    second = AttributeTable.from_records(backwards, specs=specs)

    assert first.digest() != second.digest()
    assert dict(zip(first.doc_ids, first.column("popularity").ranks, strict=True)) == dict(
        zip(second.doc_ids, second.column("popularity").ranks, strict=True)
    ), "the same document keeps the same rank"


# ---------------------------------------------------------------------------
# build_keys: the two refusals, and what the key is made of
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("n_scores", "n_docs"), [(1, 2), (3, 2), (0, 2)])
def test_a_score_count_that_disagrees_with_the_table_names_both_counts(
    n_scores: int, n_docs: int
) -> None:
    """Zipping the shorter of the two would silently rank a prefix of the corpus
    and report it as the whole thing, so the counts are compared first and the
    message carries both."""
    table = AttributeTable.from_records(
        [{"doc_id": chr(97 + i), "popularity": i} for i in range(n_docs)],
        specs=(AttributeSpec("popularity"),),
    )
    with pytest.raises(
        ValueError, match=f"{n_scores} scores but the attribute table has {n_docs} documents"
    ):
        build_keys([0.5] * n_scores, table, PI)


@pytest.mark.parametrize("name", ["identifier", "doc_id"])
def test_the_identifier_may_not_be_named_in_a_priority(name: str) -> None:
    """Both spellings. It terminates every key implicitly, so naming it would
    either duplicate it or move it -- and moving it makes the ordering non-total,
    which is the one property the whole tie-break rests on."""
    table = _two_document_table()
    spec = SortKeySpec(name="broken", priority=("popularity", name))

    with pytest.raises(ValueError, match="the identifier terminates every key implicitly"):
        build_keys([0.5, 0.4], table, spec)


def test_the_key_leads_with_the_negated_score_and_ends_with_the_identifier() -> None:
    """The shape is `(-score, *attribute ranks, id_rank)`. Negated because the
    sort is ascending while scores rank descending, and the identifier last
    because it is the only field guaranteed to break every remaining tie."""
    table = _two_document_table()
    keys = build_keys([0.25, 0.75], table, _POP_ONLY)

    assert [k[0] for k in keys] == [-0.25, -0.75]
    assert len(keys[0]) == 3, "score, one attribute, identifier"
    assert [k[-1] for k in keys] == list(table.id_ranks)


def test_a_priority_of_no_attributes_still_ends_with_the_identifier() -> None:
    """`pi_score` reduces to this once scores are distinct: the key is the score
    and the identifier, and it is still injective."""
    table = _two_document_table()
    keys = build_keys([0.5, 0.5], table, SortKeySpec(name="score_only", priority=()))

    assert all(len(k) == 2 for k in keys)
    assert len(set(keys)) == 2, "still injective on a total tie"


def test_a_non_finite_score_is_not_rejected_when_the_key_is_built() -> None:
    """`rank` rejects it before sorting; `build_keys` does not. Pinning the
    division of responsibility, so the guard is not accidentally duplicated or
    accidentally removed from the one place that has it."""
    table = _two_document_table()
    keys = build_keys([math.nan, 0.5], table, PI)
    assert math.isnan(keys[0][0])


# ---------------------------------------------------------------------------
# assert_strict_total_order: the size gate
# ---------------------------------------------------------------------------
def test_the_axioms_hold_vacuously_on_nothing_and_on_one() -> None:
    """No pairs and no triples. A guard that indexed before checking the size
    would fail here rather than passing trivially."""
    assert_strict_total_order([])
    assert_strict_total_order([(1.0, 0)])


def test_the_duplicate_report_counts_how_many_collided() -> None:
    """The count tells a reader whether one document was duplicated or the key
    collapsed entirely -- which are different bugs."""
    with pytest.raises(AssertionError, match=r"not injective: 2 duplicate key\(s\)"):
        assert_strict_total_order([(1.0, 0), (1.0, 0), (1.0, 0)])


@pytest.mark.parametrize("n", [40, 41])
def test_the_transitivity_sweep_runs_at_forty_and_stops_above_it(n: int) -> None:
    """The check is O(n^3), so it is gated at 40. Both sides of the gate are
    exercised here: at 41 the sweep is skipped and the call still returns, which
    is what keeps a debug mode usable on a real corpus.
    """
    keys = [(float(-i), i) for i in range(n)]
    assert_strict_total_order(keys)


# ---------------------------------------------------------------------------
# Ranking accessors: the index domain
# ---------------------------------------------------------------------------
def _ranked(n: int = 4, k: int | None = None) -> Ranking:
    """A ranking over `n` distinctly-scored documents, optionally truncated."""
    table = table_of(
        n,
        popularity=[n - i for i in range(n)],
        rating=[n - i for i in range(n)],
        engagement=[n - i for i in range(n)],
    )
    scores = [1.0 - 0.1 * i for i in range(n)]
    if k is None:
        return rank(scores, table, PI)
    return rank_top_k(scores, table, PI, k=k)


def test_asking_for_no_documents_at_all_returns_nothing() -> None:
    """`top_k(0)` is a legitimate question with an empty answer, not an error."""
    assert _ranked().top_k(0) == ()


def test_a_negative_top_k_silently_drops_from_the_end() -> None:
    """`order[:-1]` is a legal slice, so a `k` that arrived negative returns
    almost the whole ranking rather than raising. The upper bound is checked and
    the lower one is not.

    Pinned as the asymmetry it is -- the same slicing trap as `short(digest, -1)`
    in the hashing module, and reached the same way.
    """
    ranking = _ranked(4)
    assert len(ranking.top_k(-1)) == 3
    assert ranking.top_k(-1) == ranking.order[:-1]

    with pytest.raises(ValueError, match=r"top_k\(5\) but only 4 documents were selected"):
        ranking.top_k(5)


@pytest.mark.parametrize("j", [0, -1, 5, 99])
def test_a_rank_outside_one_to_n_is_refused_with_the_range(j: int) -> None:
    """Ranks are 1-indexed in the public API to match `r_1 ... r_n`, so 0 is out
    of range at the bottom exactly as `n + 1` is at the top."""
    with pytest.raises(IndexError, match=f"rank {j} out of range 1..4"):
        _ranked(4).score_at_rank(j)


def test_the_first_and_last_ranks_are_both_reachable() -> None:
    """The companion to the refusals: an inclusive range means both ends work."""
    ranking = _ranked(4)
    assert ranking.score_at_rank(1) == max(ranking.sorted_scores)
    assert ranking.score_at_rank(4) == min(ranking.sorted_scores)


def test_a_truncated_ranking_says_so_when_asked_for_an_unselected_document() -> None:
    """The two `KeyError` messages differ by one clause, and that clause is the
    whole diagnosis: on a complete ranking the document does not exist, on a
    truncated one it merely was not selected."""
    with pytest.raises(KeyError, match="the ranking is truncated"):
        _ranked(4, k=2).rank_of(3)

    with pytest.raises(KeyError) as caught:
        _ranked(4).rank_of(99)
    assert "truncated" not in str(caught.value)


def test_a_whole_corpus_question_names_what_was_asked_and_what_was_selected() -> None:
    """The message routes the reader to `rank()` rather than leaving them to
    work out why a ranking they hold cannot answer."""
    with pytest.raises(ValueError, match="needs the complete ranking, but only 2 of 4"):
        _ranked(4, k=1).require_complete("kendall_tau_distance")


def test_a_complete_ranking_answers_whole_corpus_questions_silently() -> None:
    assert _ranked(4).require_complete("anything") is None
    assert _ranked(4).is_complete


# ---------------------------------------------------------------------------
# rank_top_k: how many documents get selected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2])
def test_one_more_than_k_is_selected_so_the_boundary_gap_is_available(k: int) -> None:
    """`m = min(k + 1, n)`. The extra document is `r_{k+1}`, without which the
    boundary margin at `k` could not be computed from the ranking."""
    ranking = _ranked(4, k=k)
    assert ranking.n_selected == k + 1
    assert not ranking.is_complete


@pytest.mark.parametrize("k", [3, 4])
def test_once_k_plus_one_reaches_the_corpus_the_ranking_is_complete(k: int) -> None:
    """`m = min(k + 1, n)`, so the clamp bites one k *below* the corpus size:
    on four documents a top-3 already selects all four, and there is no further
    document for the boundary gap to point at."""
    ranking = _ranked(4, k=k)
    assert ranking.n_selected == 4
    assert ranking.is_complete


def test_the_sorted_scores_are_complete_even_where_the_order_is_truncated() -> None:
    """Margins are computed from the score multiset, which does not depend on
    the selection, so a truncated ranking can still answer a margin question at
    any k."""
    ranking = _ranked(6, k=2)
    assert len(ranking.sorted_scores) == 6
    assert len(ranking.order) == 3


# ---------------------------------------------------------------------------
# The selection algorithms
# ---------------------------------------------------------------------------
def test_an_unrecognised_selection_falls_through_to_the_insertion_sort() -> None:
    """The dispatch is an `if` chain with no `else: raise`, so a value outside
    the enumeration takes the final branch rather than being refused.

    It cannot be reached through the public API -- `rank` types the parameter as
    `Selection` -- and the fallback is a correct independent implementation, so
    the consequence is a slow ranking rather than a wrong one. Pinned because
    the shape invites a fifth algorithm being added above the fallback and
    silently never running.
    """
    table = _two_document_table()
    keys = build_keys([0.25, 0.75], table, PI)

    assert _select(keys, 2, "not_a_selection") == _select(keys, 2, Selection.INSERTION)  # type: ignore[arg-type]


@pytest.mark.parametrize("how", list(Selection))
def test_every_selection_algorithm_answers_the_degenerate_sizes_alike(how: Selection) -> None:
    """`m = 0` and `m = n` are where a top-k heap and a full sort are most
    likely to disagree, and they must not."""
    table = _two_document_table()
    keys = build_keys([0.25, 0.75], table, PI)

    assert _select(keys, 0, how) == ()
    assert _select(keys, 2, how) == _select(keys, 2, Selection.FULL_SORT)


# ---------------------------------------------------------------------------
# rank_all_operators
# ---------------------------------------------------------------------------
def test_every_operator_sees_one_and_the_same_score_array() -> None:
    """Asserted by identity, not equality. A disagreement between operators is
    only attributable to the tie-break if the scores they saw were the same
    object; two equal arrays computed twice would leave a second explanation.
    """
    table = _two_document_table()
    rankings = rank_all_operators([0.5, 0.5], table)

    shared = {id(r.sorted_scores) for r in rankings.values()}
    assert len(shared) == 1


def test_asking_for_no_operators_returns_no_rankings() -> None:
    assert rank_all_operators([0.5, 0.5], _two_document_table(), specs=()) == {}


def test_two_specs_of_one_name_collapse_to_a_single_entry() -> None:
    """The result is keyed by operator name, so a repeated name overwrites
    rather than accumulating. Pinned because an ablation sweeping a list of
    specs would silently measure fewer operators than it listed."""
    table = _two_document_table()
    duplicated = (PI, SortKeySpec(name=PI.name, priority=("engagement",)))

    assert len(rank_all_operators([0.5, 0.5], table, specs=duplicated)) == 1


def test_the_overflow_bound_is_the_largest_int64_and_not_one_past_it() -> None:
    """Straddled exactly. The C++ mirror multiplies into `__int128` and this
    guard is what lets its inner loop skip a checked multiply, so the bound has
    to be the largest value that mirror can hold and not the first it cannot.
    """
    largest = (1 << 63) - 1
    check_ratio_fits_int64([(largest, 1), (1, 1)], "rating")

    with pytest.raises(TfidfStabilityError, match="overflows int64"):
        check_ratio_fits_int64([(1 << 63, 1), (1, 1)], "rating")


def test_the_guard_looks_past_the_single_largest_numerator_and_denominator() -> None:
    """Why it takes the *two* largest of each rather than the one.

    Here the largest numerator and the largest denominator belong to the same
    document, so their product never arises -- nothing is compared with itself.
    The maximising pair is the largest numerator against the *second* largest
    denominator, and a guard that kept only the top one would find no admissible
    combination at all and pass a column the mirror cannot represent.
    """
    same_document = [(2**62, 2**62), (2**40, 2**40)]

    assert 2**62 * 2**40 > (1 << 63) - 1, "the cross pair really does overflow"
    with pytest.raises(TfidfStabilityError, match="overflows int64"):
        check_ratio_fits_int64(same_document, "rating")


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((1, 2), (2, 3), -1),
        ((2, 3), (1, 2), 1),
        ((1, 2), (2, 4), 0),
        ((0, 1), (1, 1), -1),
        ((3, 1), (3, 1), 0),
    ],
)
def test_the_ratio_comparator_reports_all_three_outcomes(
    a: tuple[int, int], b: tuple[int, int], expected: int
) -> None:
    """A three-way comparison, and each value has to be distinct: collapsing
    "less" onto "equal" would make the sort treat an ordered pair as tied and
    fall through to the next attribute that should never have been reached."""
    assert _ratio_cmp(a, b) == expected


def test_a_missing_first_column_shifts_every_present_value_up_by_one() -> None:
    """The absent documents take rank 0, so the present ones start at 1.

    Without the shift a present value would share rank 0 with every absent one,
    and the tie-break would treat "least popular" and "unrated" as the same
    thing -- which is exactly the conflation the presence bit exists to prevent.
    """
    records = [{"doc_id": "a", "p": 5}, {"doc_id": "b"}, {"doc_id": "c", "p": 3}]

    first = AttributeTable.from_records(
        records, specs=(AttributeSpec("p", missing_policy=MissingPolicy.FIRST),)
    )
    column = first.column("p")
    assert column.ranks == (1, 0, 2)
    present_ranks = {r for r, ok in zip(column.ranks, column.has_value, strict=True) if ok}
    assert 0 not in present_ranks, "rank 0 belongs to the absent documents alone"

    last = AttributeTable.from_records(
        records, specs=(AttributeSpec("p", missing_policy=MissingPolicy.LAST),)
    )
    assert last.column("p").ranks == (0, 2, 1), "and no shift is needed the other way"
