"""Document frequency and smoothed IDF (README section 2.1).

The golden values here are *derived*, not recorded: each is computed
independently in exact arithmetic (:class:`~decimal.Decimal` at 80 digits) from
the formula as written in the paper, then compared bit-for-bit against the
implementation. A recorded snapshot would only catch regressions; a derivation
also catches an error that has been present since the first commit.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from itertools import pairwise

import pytest

from tfidf_stability.utils.numerics import correctly_rounded_log_ratio, same_bits
from tfidf_stability.utils.validation import EmptyVocabularyError
from tfidf_stability.vectorisation.idf import (
    LogImpl,
    delta_idf,
    smoothed_idf_one,
)
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser
from tfidf_stability.vectorisation.vocabulary import (
    MaxFeaturesPolicy,
    VocabularyConfig,
    build_vocabulary,
)


def reference_idf(df: int, n: int) -> float:
    """``log((1+N)/(1+df)) + 1``, derived independently in exact arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 80
        return float((Decimal(1 + n) / Decimal(1 + df)).ln()) + 1.0


# ---------------------------------------------------------------------------
# Document frequency
# ---------------------------------------------------------------------------
def test_df_counts_documents_not_occurrences() -> None:
    # 'a' occurs three times but in only two documents.
    vocab = build_vocabulary([["a", "a", "a"], ["a", "b"], ["b"]])
    assert vocab.df_of("a") == 2
    assert vocab.df_of("b") == 2
    assert vocab.cf[vocab.id_of("a")] == 4  # type: ignore[index]


def test_df_of_absent_token_is_zero() -> None:
    vocab = build_vocabulary([["a"], ["b"]])
    assert vocab.df_of("zzz") == 0
    assert vocab.id_of("zzz") is None


def test_vocabulary_is_byte_sorted_and_ids_follow_that_order() -> None:
    vocab = build_vocabulary([["zebra", "apple", "Mango".lower(), "banana"]])
    assert vocab.tokens == ("apple", "banana", "mango", "zebra")
    assert vocab.is_sorted()
    assert [vocab.id_of(t) for t in vocab.tokens] == [0, 1, 2, 3]


def test_vocabulary_is_invariant_to_document_order() -> None:
    """The determinism guarantee: identifiers depend on the token set alone."""
    docs = [["c", "a"], ["b"], ["a", "b", "d"]]
    a = build_vocabulary(docs)
    b = build_vocabulary(list(reversed(docs)))
    assert a.tokens == b.tokens
    assert a.df == b.df
    assert a.digest() == b.digest()


# ---------------------------------------------------------------------------
# IDF -- against the exact derivation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 10, 610, 9742])
def test_idf_matches_exact_derivation_bitwise(n: int) -> None:
    for df in range(0, n + 1):
        got = smoothed_idf_one(df, n)
        exp = reference_idf(df, n)
        assert same_bits(got, exp), f"df={df} N={n}: {got!r} != {exp!r}"


@pytest.mark.parametrize("n", [5, 50, 500])
def test_idf_is_strictly_decreasing_in_df(n: int) -> None:
    """Section 2.1: "monotonic decay of idf(t) as df(t) increases"."""
    vals = [smoothed_idf_one(df, n) for df in range(0, n + 1)]
    assert all(a > b for a, b in pairwise(vals))


@pytest.mark.parametrize("n", [1, 7, 100, 9742])
def test_idf_is_at_least_one(n: int) -> None:
    """Strict positivity, and in fact ``idf >= 1``.

    Section 2.1 claims only positivity, but ``df <= N`` forces the ratio to be
    at least 1 and hence the logarithm non-negative. The stronger bound is what
    makes the corpus-level Lipschitz constant of spec_addenda G4 computable, so
    it is asserted rather than assumed.
    """
    for df in range(0, n + 1):
        assert smoothed_idf_one(df, n) >= 1.0


def test_idf_at_df_equals_n_is_exactly_one() -> None:
    """The limiting case section 2.1 calls out: log(1) + 1 == 1, exactly."""
    for n in (1, 2, 10, 1000):
        assert smoothed_idf_one(n, n) == 1.0


def test_idf_rejects_out_of_range_df() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        smoothed_idf_one(-1, 10)
    with pytest.raises(ValueError, match="exceeds the corpus size"):
        smoothed_idf_one(11, 10)


# ---------------------------------------------------------------------------
# G13 -- the platform logarithm is not correctly rounded
# ---------------------------------------------------------------------------
def test_platform_log_differs_from_correctly_rounded() -> None:
    """Pins the finding that motivates the exact-logarithm default.

    If this test ever *fails* -- that is, if the platform libm becomes correctly
    rounded -- the exact-log machinery could be retired. Until then it is load
    bearing, and this test keeps the reason visible.
    """
    n = 9742
    differing = sum(
        1
        for df in range(1, n + 1)
        if not same_bits(
            smoothed_idf_one(df, n, LogImpl.CORRECTLY_ROUNDED),
            smoothed_idf_one(df, n, LogImpl.PLATFORM),
        )
    )
    # Measured at ~15% on the reference machine; the bound is deliberately loose
    # so the test asserts "materially different", not a machine-specific figure.
    assert differing > 0.05 * n, f"only {differing}/{n} entries differ"


def test_division_before_log_is_not_the_same_as_difference_of_logs() -> None:
    """Section 2.1 writes ``log((1+N)/(1+df))``; the form is load-bearing."""
    n = 9742
    differing = sum(
        1
        for df in range(1, n + 1)
        if math.log((1 + n) / (1 + df)) != math.log(1 + n) - math.log(1 + df)
    )
    assert differing > 0.5 * n


def test_correctly_rounded_log_is_precision_stable() -> None:
    """60 digits is comfortably enough to round binary64 correctly."""
    from decimal import getcontext

    for a, b in ((9743, 5), (5, 4), (100001, 2), (2, 1)):
        lo = correctly_rounded_log_ratio(a, b)
        prev = getcontext().prec
        try:
            getcontext().prec = 200
            hi = float((Decimal(a) / Decimal(b)).ln())
        finally:
            getcontext().prec = prev
        assert same_bits(lo, hi)


# ---------------------------------------------------------------------------
# delta_idf (section 4.1)
# ---------------------------------------------------------------------------
def test_delta_idf_is_zero_for_an_unchanged_corpus() -> None:
    assert delta_idf(3, 3, 10, 10) == 0.0


def test_delta_idf_matches_the_difference_of_idf_values() -> None:
    """The ``+1`` cancels, so delta must equal the difference of the idf values."""
    got = delta_idf(2, 3, 10, 11)
    exp = smoothed_idf_one(3, 11) - smoothed_idf_one(2, 10)
    assert abs(got - exp) <= 4 * math.ulp(max(abs(got), abs(exp), 1.0))


def test_delta_idf_sign_follows_document_frequency() -> None:
    """Adding a document that contains t lowers its idf; one that does not raises it."""
    assert delta_idf(3, 4, 10, 11) < 0  # t gained a document
    assert delta_idf(3, 3, 10, 11) > 0  # corpus grew, t did not


# ---------------------------------------------------------------------------
# Vocabulary filtering
# ---------------------------------------------------------------------------
def test_min_df_filters_rare_tokens() -> None:
    docs = [["a", "b"], ["a", "c"], ["a", "b"]]
    vocab = build_vocabulary(docs, VocabularyConfig(min_df=2))
    assert vocab.tokens == ("a", "b")
    assert vocab.n_discarded == 1


def test_min_df_as_a_proportion_is_resolved_exactly() -> None:
    """``Fraction(0.1) * 30`` exceeds 3 and would ceil to 4, excluding b.

    It is ``limit_denominator`` that snaps 0.1 to one tenth and returns 3, so
    this test fails if that call is dropped as cosmetic. (A previous version of
    this docstring said ``0.1 * 30`` is 3.0000000000000004; it is exactly 3.0.)
    """
    docs = [["a"] for _ in range(27)] + [["a", "b"] for _ in range(3)]
    vocab = build_vocabulary(docs, VocabularyConfig(min_df=0.1))
    assert "b" in vocab, "b has df=3 and the threshold is exactly 3"


def test_max_df_filters_ubiquitous_tokens() -> None:
    docs = [["a", "b"], ["a", "c"], ["a", "d"]]
    vocab = build_vocabulary(docs, VocabularyConfig(max_df=2))
    assert "a" not in vocab


def test_a_proportional_max_df_never_admits_more_than_the_proportion() -> None:
    """An upper bound must round *down*, unlike ``min_df``.

    Both thresholds once shared one ceiling, which is right for the lower bound
    and inverts the upper one: at ``p=0.5, n=3`` the cap resolved to 2 and kept
    a token present in 2 of 3 documents, and at ``p=0.95, n=7`` it resolved to 7
    and filtered nothing at all.
    """
    docs = [["a", "b"], ["a", "c"], ["d"]]
    vocab = build_vocabulary(docs, VocabularyConfig(max_df=0.5))
    assert "a" not in vocab, "df 2/3 = 66.7% must not survive a 50% cap"
    for token in ("b", "c", "d"):
        assert token in vocab, f"{token} has df 1/3 and must survive"

    # The degenerate end: a cap below 1/n admits nothing, which is the honest
    # answer rather than silently keeping everything.
    ubiquitous = [["a", "b"] for _ in range(7)]
    with pytest.raises(EmptyVocabularyError):
        build_vocabulary(ubiquitous, VocabularyConfig(max_df=0.95))


def test_empty_vocabulary_raises() -> None:
    with pytest.raises(EmptyVocabularyError):
        build_vocabulary([["a"], ["b"]], VocabularyConfig(min_df=3))
    with pytest.raises(EmptyVocabularyError, match="empty corpus"):
        build_vocabulary([])


# ---------------------------------------------------------------------------
# max_features -- TFIDF-SPEC-01 (spec_addenda G6)
# ---------------------------------------------------------------------------
def test_max_features_keeps_the_highest_df_tokens() -> None:
    docs = [["a", "b", "c"], ["a", "b"], ["a"]]  # df: a=3, b=2, c=1
    vocab = build_vocabulary(docs, VocabularyConfig(max_features=2))
    assert set(vocab.tokens) == {"a", "b"}


def test_max_features_tie_break_is_total_and_order_invariant() -> None:
    """The rule must not depend on which document happened to arrive first.

    Every token here has df=1 and cf=1, so the outcome rests entirely on the
    final byte-order key. Without it the result would be arbitrary.
    """
    docs = [["z"], ["y"], ["x"], ["w"]]
    a = build_vocabulary(docs, VocabularyConfig(max_features=2))
    b = build_vocabulary(list(reversed(docs)), VocabularyConfig(max_features=2))
    assert a.tokens == b.tokens == ("w", "x")


def test_max_features_cf_breaks_df_ties_before_byte_order() -> None:
    # 'a' and 'b' both have df=2, but 'a' occurs more often overall.
    docs = [["a", "a", "b"], ["a", "b"]]
    vocab = build_vocabulary(docs, VocabularyConfig(max_features=1))
    assert vocab.tokens == ("a",)


def test_max_features_policies_can_disagree() -> None:
    """df-ranking and cf-ranking are genuinely different criteria."""
    # 'rare' appears many times but in one document; 'common' once per document.
    docs = [["rare"] * 10 + ["common"], ["common"], ["common"]]
    by_df = build_vocabulary(docs, VocabularyConfig(max_features=1))
    by_cf = build_vocabulary(
        docs,
        VocabularyConfig(max_features=1, max_features_policy=MaxFeaturesPolicy.CF_DESC),
    )
    assert by_df.tokens == ("common",)
    assert by_cf.tokens == ("rare",)


# ---------------------------------------------------------------------------
# End to end on the mini corpus
# ---------------------------------------------------------------------------
def test_mini_corpus_idf_is_exact(mini_model) -> None:  # type: ignore[no-untyped-def]
    n = mini_model.vocabulary.n_documents
    for term_id, df in enumerate(mini_model.vocabulary.df):
        assert same_bits(mini_model.idf[term_id], reference_idf(df, n))


def test_mini_corpus_has_the_expected_shape(mini_model) -> None:  # type: ignore[no-untyped-def]
    assert mini_model.n_documents == 6
    assert mini_model.vocabulary.is_sorted()
    assert mini_model.matrix.is_canonical()
    # d5 is entirely stopwords, so it must embed to the zero vector (section 2.2).
    assert mini_model.doc_ids[4] == "d5"
    assert mini_model.norms[4] == 0.0
    assert mini_model.zero_norm_documents == (4,)
    # d3 and d4 are identical text, so their rows must be bit-identical.
    assert mini_model.matrix.row(2).values == mini_model.matrix.row(3).values


def test_fitting_is_deterministic_across_repeats(mini_features, mini_corpus) -> None:  # type: ignore[no-untyped-def]
    ids = [str(d["doc_id"]) for d in mini_corpus]
    a = TfidfVectoriser().fit(list(mini_features), ids)
    b = TfidfVectoriser().fit(list(mini_features), ids)
    assert a.digest() == b.digest()
