"""Document frequency and smoothed IDF (README section 2.1).

Every golden value is computed in exact arithmetic (:class:`~decimal.Decimal`
at 80 digits) from the formula as written in the paper, then compared bit for
bit against the implementation, which catches an error a snapshot would keep.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from itertools import pairwise

import pytest

from tfidf_stability.utils.numerics import correctly_rounded_log_ratio, same_bits, ulps_between
from tfidf_stability.utils.validation import EmptyVocabularyError, TfidfStabilityError
from tfidf_stability.vectorisation.idf import (
    IdfVector,
    LogImpl,
    delta_idf,
    idf_linf,
    smoothed_idf,
    smoothed_idf_one,
)
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser
from tfidf_stability.vectorisation.vocabulary import (
    MaxFeaturesPolicy,
    VocabularyConfig,
    _resolve_threshold,
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
# IDF: against the exact derivation
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
    """``df <= N`` gives ``idf >= 1``, a stronger bound than section 2.1 states.

    The ratio is at least 1, so the logarithm is non-negative. The corpus-level
    Lipschitz constant of spec_addenda G4 needs this stronger bound.
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
# G13: the platform logarithm is not correctly rounded
# ---------------------------------------------------------------------------
def test_platform_log_differs_from_correctly_rounded() -> None:
    """The platform logarithm differs from the correctly rounded one.

    This difference motivates the exact-logarithm default. A failure here means
    the platform libm is correctly rounded and the exact-log path is redundant.
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
    # About 15% of entries differ on the reference machine. The 5% bound asserts
    # a material difference rather than a machine-specific figure.
    assert differing > 0.05 * n, f"only {differing}/{n} entries differ"


def test_the_string_spelling_of_a_log_impl_selects_the_same_logarithm() -> None:
    """The string and the member spellings select the same logarithm.

    `LogImpl` is a `str` enum, so `"platform"` equals the member without being
    it; a selection on `is` would send the string to the correctly-rounded
    branch. Both spellings must agree and differ from the correctly-rounded value.
    """
    n = 9742
    # The two logarithms agree on most df, so the case is chosen rather than
    # assumed: a df they agree on passes whichever branch the string takes.
    df = next(
        d
        for d in range(1, n + 1)
        if not same_bits(
            smoothed_idf_one(d, n, LogImpl.CORRECTLY_ROUNDED),
            smoothed_idf_one(d, n, LogImpl.PLATFORM),
        )
    )
    exact = smoothed_idf_one(df, n, LogImpl.CORRECTLY_ROUNDED)
    member = smoothed_idf_one(df, n, LogImpl.PLATFORM)
    assert not same_bits(member, exact), "the df chosen must be one that differs"

    assert same_bits(smoothed_idf_one(df, n, "platform"), member)  # type: ignore[arg-type]
    assert same_bits(smoothed_idf_one(df, n, "correctly_rounded"), exact)  # type: ignore[arg-type]

    # delta_idf selects the same way, and so does the vector's recorded flag.
    assert same_bits(
        delta_idf(df, df + 1, n, n, "platform"),  # type: ignore[arg-type]
        delta_idf(df, df + 1, n, n, LogImpl.PLATFORM),
    )
    assert smoothed_idf([df], n, "platform").log_impl is LogImpl.PLATFORM  # type: ignore[arg-type]


def test_an_unrecognised_log_impl_is_refused_rather_than_defaulted() -> None:
    """An unrecognised `LogImpl` raises `ValueError`.

    A misspelling therefore cannot select the default and then be recorded as
    the value requested.
    """
    for bad in ("Platform", "exact", ""):
        with pytest.raises(ValueError, match="is not a valid LogImpl"):
            smoothed_idf_one(3, 10, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is not a valid LogImpl"):
        IdfVector(values=(1.0,), n_documents=1, log_impl="platfrom")  # type: ignore[arg-type]


def test_division_before_log_is_not_the_same_as_difference_of_logs() -> None:
    """The two forms of the formula disagree on over half of df in 1..9742.

    Section 2.1 writes ``log((1+N)/(1+df))``, not a difference of logarithms.
    """
    n = 9742
    differing = sum(
        1
        for df in range(1, n + 1)
        if math.log((1 + n) / (1 + df)) != math.log(1 + n) - math.log(1 + df)
    )
    assert differing > 0.5 * n


def test_correctly_rounded_log_is_precision_stable() -> None:
    """A precision of 60 digits is enough to round binary64 correctly."""
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
    """The ``+1`` cancels, so delta equals the difference of the idf values."""
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

    ``limit_denominator`` snaps 0.1 to one tenth, so the threshold resolves to 3;
    without that call this test fails.
    """
    docs = [["a"] for _ in range(27)] + [["a", "b"] for _ in range(3)]
    vocab = build_vocabulary(docs, VocabularyConfig(min_df=0.1))
    assert "b" in vocab, "b has df=3 and the threshold is exactly 3"


def test_max_df_filters_ubiquitous_tokens() -> None:
    docs = [["a", "b"], ["a", "c"], ["a", "d"]]
    vocab = build_vocabulary(docs, VocabularyConfig(max_df=2))
    assert "a" not in vocab


def test_a_proportional_max_df_never_admits_more_than_the_proportion() -> None:
    """An upper bound must round down, unlike ``min_df``.

    A ceiling inverts the upper bound: at ``p=0.5, n=3`` it gives 2 and keeps a
    token present in 2 of 3 documents; at ``p=0.95, n=7`` it gives 7 and filters
    nothing.
    """
    docs = [["a", "b"], ["a", "c"], ["d"]]
    vocab = build_vocabulary(docs, VocabularyConfig(max_df=0.5))
    assert "a" not in vocab, "df 2/3 = 66.7% must not survive a 50% cap"
    for token in ("b", "c", "d"):
        assert token in vocab, f"{token} has df 1/3 and must survive"

    # The degenerate end: a cap below 1/n admits nothing rather than keeping
    # everything.
    ubiquitous = [["a", "b"] for _ in range(7)]
    with pytest.raises(EmptyVocabularyError, match="over 7 documents with 2 distinct features"):
        build_vocabulary(ubiquitous, VocabularyConfig(max_df=0.95))


def test_empty_vocabulary_raises() -> None:
    with pytest.raises(EmptyVocabularyError, match="min_df=3, max_df=2"):
        build_vocabulary([["a"], ["b"]], VocabularyConfig(min_df=3))
    with pytest.raises(EmptyVocabularyError, match="empty corpus"):
        build_vocabulary([])


# ---------------------------------------------------------------------------
# max_features: TFIDF-SPEC-01 (spec_addenda G6)
# ---------------------------------------------------------------------------
def test_max_features_keeps_the_highest_df_tokens() -> None:
    docs = [["a", "b", "c"], ["a", "b"], ["a"]]  # df: a=3, b=2, c=1
    vocab = build_vocabulary(docs, VocabularyConfig(max_features=2))
    assert set(vocab.tokens) == {"a", "b"}


def test_max_features_tie_break_is_total_and_order_invariant() -> None:
    """The rule must not depend on which document arrived first.

    Every token here has df=1 and cf=1, so the outcome rests on the final
    byte-order key; without it the result is arbitrary.
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
    """df-ranking and cf-ranking are different criteria."""
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


def test_a_negative_max_features_is_rejected_not_silently_applied() -> None:
    """A negative ``max_features`` raises rather than slicing ``survivors[:-1]``.

    ``-1`` is the plausible typo for "unlimited", which this codebase spells
    ``null``; ``min_df`` and ``max_df`` reject a negative value in the same way.
    """
    docs = [["a", "b", "c"], ["a", "b"], ["a"], ["b", "d"]]
    assert build_vocabulary(docs, VocabularyConfig()).tokens == ("a", "b", "c", "d")

    for bad in (-1, -2):
        with pytest.raises(ValueError, match="max_features"):
            build_vocabulary(docs, VocabularyConfig(max_features=bad))

    # The sibling thresholds reject a negative value in the same way.
    for name in ("min_df", "max_df"):
        with pytest.raises(ValueError, match=name):
            build_vocabulary(docs, VocabularyConfig(**{name: -1}))


# ---------------------------------------------------------------------------
# The IDF vector's own accessors, and the bounds section 4.2 reads off them
# ---------------------------------------------------------------------------
def test_an_idf_vector_reports_one_entry_per_vocabulary_term() -> None:
    idf = smoothed_idf([1, 2, 3], 10)
    assert len(idf) == 3
    assert idf[0] == smoothed_idf_one(1, 10)


def test_the_infinity_norm_is_the_largest_idf_and_the_minimum_the_smallest() -> None:
    """The vector's ``linf`` and ``minimum`` are its largest and smallest entries.

    ||idf||_inf appears in the section 4.2 perturbation bound, so it is read from
    the vector rather than recomputed at the call site.
    """
    idf = smoothed_idf([1, 5, 10], 10)
    assert idf.linf == max(idf.values)
    assert idf.minimum == min(idf.values)
    assert idf.minimum >= 1.0, "every df <= N gives an idf of at least 1"


def test_an_empty_idf_vector_reports_zero_rather_than_raising() -> None:
    """An empty vector reports zero for both extremes.

    ``max()`` over nothing raises, and a report assembled from a filtered corpus
    must not fail because a vocabulary is empty.
    """
    empty = smoothed_idf([], 10)
    assert len(empty) == 0
    assert empty.linf == 0.0
    assert empty.minimum == 0.0


def test_the_free_function_accepts_either_a_vector_or_a_raw_sequence() -> None:
    """Both argument forms give the same norm.

    The bound is computed in places that hold one or the other, and a
    disagreement between the two forms would move a published bound.
    """
    idf = smoothed_idf([1, 5, 10], 10)
    assert idf_linf(idf) == idf.linf
    assert idf_linf(list(idf.values)) == idf.linf


def test_the_free_function_over_an_empty_sequence_is_zero() -> None:
    assert idf_linf([]) == 0.0
    assert idf_linf(smoothed_idf([], 4)) == 0.0


# ---------------------------------------------------------------------------
# Vocabulary thresholds as proportions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, 2.0])
def test_a_proportion_outside_zero_to_one_is_rejected(bad: float) -> None:
    """A float threshold outside (0, 1] is rejected.

    A float is a proportion of the corpus, not a very large absolute count.
    """
    with pytest.raises(ValueError, match="as a proportion must be in"):
        build_vocabulary([["a"], ["b"]], VocabularyConfig(min_df=bad))


def test_a_proportion_of_exactly_one_is_accepted_as_the_whole_corpus() -> None:
    """A proportion of 1.0 is every document and filters nothing as an upper bound."""
    vocab = build_vocabulary([["a", "b"], ["a"]], VocabularyConfig(max_df=1.0))
    assert "a" in vocab.tokens, "a term in every document survives max_df = 1.0"


def test_the_collection_frequency_policy_orders_by_total_occurrences() -> None:
    """CF_DESC and DF_DESC disagree when a term is concentrated in one document.

    That is the case the policy exists to distinguish.
    """
    corpus = [["rare", "rare", "rare"], ["common"], ["common"]]
    by_cf = build_vocabulary(
        corpus, VocabularyConfig(max_features=1, max_features_policy=MaxFeaturesPolicy.CF_DESC)
    )
    by_df = build_vocabulary(
        corpus, VocabularyConfig(max_features=1, max_features_policy=MaxFeaturesPolicy.DF_DESC)
    )
    assert by_cf.tokens == ("rare",), "three occurrences in one document wins on cf"
    assert by_df.tokens == ("common",), "two documents wins on df"


def test_the_sklearn_compatible_policy_ignores_document_frequency_entirely() -> None:
    corpus = [["rare", "rare", "rare"], ["common"], ["common"]]
    vocab = build_vocabulary(
        corpus,
        VocabularyConfig(max_features=1, max_features_policy=MaxFeaturesPolicy.SKLEARN_COMPAT),
    )
    assert vocab.tokens == ("rare",)


def test_the_string_spelling_of_a_policy_selects_the_same_vocabulary() -> None:
    """The string and the member spellings select the same vocabulary.

    `MaxFeaturesPolicy` is a `str` enum, so `"cf_desc"` equals the member
    without being it, and `build_vocabulary` selects the ranking on `is`. `rare`
    wins on cf and `common` on df, so the two policies cannot agree by accident.
    """
    corpus = [["rare", "rare", "rare"], ["common"], ["common"]]

    def survivor(policy: object) -> tuple[str, ...]:
        cfg = VocabularyConfig(max_features=1, max_features_policy=policy)  # type: ignore[arg-type]
        return build_vocabulary(corpus, cfg).tokens

    assert survivor(MaxFeaturesPolicy.CF_DESC) == ("rare",)
    assert survivor(MaxFeaturesPolicy.DF_DESC) == ("common",)
    assert survivor("cf_desc") == ("rare",), "the string must not fall through to df_desc"
    assert survivor("df_desc") == ("common",)
    assert survivor("sklearn_compat") == ("rare",)

    assert VocabularyConfig(max_features_policy="cf_desc").max_features_policy is (  # type: ignore[arg-type]
        MaxFeaturesPolicy.CF_DESC
    )


def test_an_unrecognised_max_features_policy_is_refused_rather_than_defaulted() -> None:
    """An unrecognised policy raises `ValueError`.

    A misspelling therefore cannot select the default and then reach the
    manifest as the policy requested.
    """
    for bad in ("cf-desc", "CF_DESC", "frequency", ""):
        with pytest.raises(ValueError, match="is not a valid MaxFeaturesPolicy"):
            VocabularyConfig(max_features_policy=bad)  # type: ignore[arg-type]


def test_a_term_id_maps_back_to_the_token_it_was_assigned_to() -> None:
    """``token_of`` is the inverse of ``id_of``, used where a report names a term."""
    vocab = build_vocabulary([["beta", "alpha"], ["alpha"]])
    for token in vocab.tokens:
        term_id = vocab.id_of(token)
        assert term_id is not None
        assert vocab.token_of(term_id) == token
    assert vocab.id_of("absent") is None, "an out-of-vocabulary token has no identifier"


def test_the_byte_order_invariant_is_raised_rather_than_asserted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sort guard raises an exception, so it survives `python -O`.

    The binary searches in tf.py and the merge in align_models are correct only
    on a byte-sorted vocabulary, so the check is not a bare `assert`. No input
    reaches it, since build_vocabulary sorts first, so `is_sorted` is patched.
    """
    from tfidf_stability.vectorisation import vocabulary as vocabulary_module

    monkeypatch.setattr(vocabulary_module.Vocabulary, "is_sorted", lambda self: False, raising=True)
    with pytest.raises(TfidfStabilityError, match="UTF-8 byte order"):
        build_vocabulary([["alpha", "beta"]])


def test_a_correctly_sorted_vocabulary_passes_that_same_guard() -> None:
    """A sorted vocabulary passes the guard."""
    vocab = build_vocabulary([["beta", "alpha", "gamma"]])
    assert vocab.is_sorted()
    assert list(vocab.tokens) == sorted(vocab.tokens, key=lambda t: t.encode("utf-8"))


# ---------------------------------------------------------------------------
# _resolve_threshold: the two ends round in opposite directions
# ---------------------------------------------------------------------------


# `min_df` keeps `df >= p*n` and rounds up; `max_df` keeps `df <= p*n` and must
# round down. Rounding an upper bound up admits what the caller asked to
# exclude; the first two cases below show it.
@pytest.mark.parametrize(
    ("proportion", "n_docs", "lower", "upper"),
    [
        (0.5, 3, 2, 1),
        (0.95, 7, 7, 6),
        (0.3, 10, 3, 3),
        (0.25, 4, 1, 1),
    ],
)
def test_the_two_ends_of_a_proportion_resolve_in_opposite_directions(
    proportion: float, n_docs: int, lower: int, upper: int
) -> None:
    """The lower bound rounds up and the upper bound rounds down.

    At `p=0.5, n=3` the exact threshold is 1.5: the lower bound is 2 and the upper
    is 1. Where `p*n` is an integer the two coincide, so only the fractional cases
    show the asymmetry.
    """
    assert _resolve_threshold(proportion, n_docs, name="min_df") == lower
    assert _resolve_threshold(proportion, n_docs, name="max_df", bound="upper") == upper


def test_an_upper_bound_rounded_up_would_filter_nothing_at_all() -> None:
    """At `p=0.95, n=7` an upper bound rounded up is 7 and filters nothing.

    Rounding down gives 6, so a `max_df` meant to drop near-ubiquitous terms
    drops them.
    """
    assert _resolve_threshold(0.95, 7, name="max_df", bound="upper") == 6
    assert _resolve_threshold(0.95, 7, name="min_df") == 7, "the other end really does give 7"


@pytest.mark.parametrize("proportion", [0.0, -0.5, 1.0000001, 2.0, float("inf")])
def test_a_proportion_outside_its_half_open_interval_is_refused(proportion: float) -> None:
    """A proportion outside `(0, 1]` is refused.

    Zero names no threshold, and anything above one asks for more documents than
    exist.
    """
    with pytest.raises(ValueError, match="as a proportion must be in"):
        _resolve_threshold(proportion, 10, name="min_df")


def test_a_proportion_of_exactly_one_is_the_whole_corpus() -> None:
    """A proportion of 1.0 requires a token to appear in every document."""
    assert _resolve_threshold(1.0, 10, name="min_df") == 10


def test_a_vanishing_proportion_still_requires_one_document() -> None:
    """The ceiling is floored at 1, so a tiny proportion cannot resolve to zero.

    A threshold of zero would admit tokens no document contains.
    """
    assert _resolve_threshold(1e-9, 10, name="min_df") == 1


@pytest.mark.parametrize("value", [-1, -(2**40)])
def test_a_negative_integer_threshold_is_refused(value: int) -> None:
    """Integers take the other branch entirely, so they need their own guard."""
    with pytest.raises(ValueError, match="must be non-negative"):
        _resolve_threshold(value, 10, name="min_df")


def test_an_integer_threshold_of_zero_is_admissible() -> None:
    """An integer `min_df` of 0 means "no lower bound" and is accepted.

    A proportion of zero is refused; the integer branch differs.
    """
    assert _resolve_threshold(0, 10, name="min_df") == 0


# ---------------------------------------------------------------------------
# build_vocabulary: the ways it comes out empty
# ---------------------------------------------------------------------------
def test_a_corpus_with_no_documents_is_distinguished_from_one_with_no_tokens() -> None:
    """An empty corpus and an over-filtered one raise different messages.

    The second message carries the thresholds so the reader can loosen them.
    """
    with pytest.raises(
        EmptyVocabularyError, match="cannot build a vocabulary from an empty corpus"
    ):
        build_vocabulary([])

    with pytest.raises(EmptyVocabularyError, match="no token survived filtering"):
        build_vocabulary([[], []])


def test_the_filtering_message_carries_every_threshold_that_produced_it() -> None:
    """The filtering message names all three thresholds and the corpus size.

    It also names the number of distinct tokens before filtering.
    """
    documents = [["a", "b"], ["a"], ["b", "c"]]
    with pytest.raises(
        EmptyVocabularyError,
        match=r"min_df=99, max_df=3, max_features=None\) over 3 documents with 3 distinct",
    ):
        build_vocabulary(documents, VocabularyConfig(min_df=99))


def test_a_negative_feature_cap_is_refused_rather_than_dropping_the_last_token() -> None:
    """A negative feature cap raises rather than slicing `survivors[:-1]`.

    `-1` is the plausible typo for "unlimited", which this codebase spells `null`.
    """
    documents = [["a", "b"], ["a"], ["b", "c"]]
    with pytest.raises(ValueError, match="max_features must be non-negative or None, got -1"):
        build_vocabulary(documents, VocabularyConfig(max_features=-1))


def test_a_feature_cap_of_zero_empties_the_vocabulary_by_the_other_route() -> None:
    """A cap of zero fails as an empty vocabulary, not as a `ValueError`.

    Zero is a non-negative cap, so it passes the guard and fails later. `-1` and
    `0` raise different errors.
    """
    documents = [["a", "b"], ["a"], ["b", "c"]]
    with pytest.raises(EmptyVocabularyError, match="max_features=0"):
        build_vocabulary(documents, VocabularyConfig(max_features=0))


def test_the_identifiers_are_byte_sorted_whatever_the_truncation_kept() -> None:
    """Survivors of `max_features` are re-sorted by byte order.

    The ranking decides which tokens survive and never which identifier each one
    gets.
    """
    documents = [["z", "a"], ["z"], ["z", "a"], ["m"]]
    vocab = build_vocabulary(documents, VocabularyConfig(max_features=2))

    assert vocab.tokens == tuple(sorted(vocab.tokens, key=lambda t: t.encode("utf-8")))
    assert vocab.is_sorted()


def test_the_discarded_count_is_what_filtering_removed() -> None:
    """`n_discarded` counts the tokens filtering removed.

    A run can then say how much of the corpus its vocabulary covers without
    recomputing the unfiltered one.
    """
    documents = [["a", "b", "c"], ["a"], ["a"]]
    vocab = build_vocabulary(documents, VocabularyConfig(min_df=2))

    assert vocab.tokens == ("a",)
    assert vocab.n_discarded == 2, "b and c fell below min_df"
    assert vocab.n_documents == 3


# ---------------------------------------------------------------------------
# smoothed_idf_one: the ends of the admissible df range
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_documents", [0, 1, 10, 9742])
def test_a_token_in_every_document_sits_exactly_on_the_smoothing_floor(
    n_documents: int,
) -> None:
    """A ubiquitous token has an idf of exactly one at every corpus size.

    `log((1+N)/(1+N)) + 1` is `log(1) + 1`. That floor keeps `w = tf * idf` from
    shrinking a term below its raw frequency, and an idf of one means "carried by
    every document", not "absent".
    """
    assert smoothed_idf_one(n_documents, n_documents) == 1.0


@pytest.mark.parametrize("n_documents", [0, 1, 10])
def test_a_token_in_no_document_takes_the_largest_idf_the_corpus_admits(
    n_documents: int,
) -> None:
    """`df = 0` is admissible and gives the largest idf.

    A token can survive into the vocabulary and then be absent from a perturbed
    corpus, which is the case section 4.1 measures.
    """
    assert smoothed_idf_one(0, n_documents) == pytest.approx(math.log(1 + n_documents) + 1.0)


@pytest.mark.parametrize(("df", "n_documents"), [(11, 10), (1, 0), (2**40, 10)])
def test_a_frequency_above_the_corpus_size_is_refused(df: int, n_documents: int) -> None:
    """A df above N is refused.

    Such a df cannot arise from counting, so the two arguments come from
    different corpora; unguarded, it would produce a plausible negative logarithm.
    """
    with pytest.raises(ValueError, match=f"df={df} exceeds the corpus size N={n_documents}"):
        smoothed_idf_one(df, n_documents)


@pytest.mark.parametrize("df", [-1, -(2**40)])
def test_a_negative_frequency_is_refused_before_the_range_check(df: int) -> None:
    """Negativity is checked before the range.

    `df=-1, N=-5` reports the negative df rather than a range comparison between
    two nonsense numbers.
    """
    with pytest.raises(ValueError, match=f"df must be non-negative, got {df}"):
        smoothed_idf_one(df, 10)


def test_the_idf_is_monotone_decreasing_in_the_document_frequency() -> None:
    """A rarer token weighs more, across the whole df range.

    The range is swept rather than sampled, since a single pair would pass for a
    formula that is monotone only locally.
    """
    values = [smoothed_idf_one(df, 100) for df in range(101)]
    assert values == sorted(values, reverse=True)
    assert values[-1] == 1.0, "and it lands exactly on the floor"


# ---------------------------------------------------------------------------
# idf_linf: an exported function with no caller in the package
# ---------------------------------------------------------------------------
def test_the_infinity_norm_of_no_idf_values_is_zero() -> None:
    """The infinity norm of no values is zero, not an error.

    Zero is the identity for the bound it feeds.
    """
    assert idf_linf([]) == 0.0


def test_the_infinity_norm_takes_the_maximum_rather_than_the_largest_magnitude() -> None:
    """`idf_linf` computes `max(x_i)`, not `max |x_i|`, and this pins that.

    On genuine idf values the two agree, because smoothing floors every entry at
    1.0; a delta vector, which section 4.1 makes freely negative, exposes the gap.
    Rationale: docs/engineering_notes.md#idf-linf-pinned-not-repaired
    """
    with_negative = [1.0, -3.0, 2.0]

    assert idf_linf(with_negative) == 2.0
    assert max(abs(v) for v in with_negative) == 3.0, "what an infinity norm would give"


def test_every_genuine_idf_vector_makes_the_two_definitions_agree() -> None:
    """No genuine idf vector can distinguish the two spellings of the norm.

    Smoothing puts every entry at or above one.
    """
    idf = [smoothed_idf_one(df, 50) for df in range(51)]

    assert min(idf) >= 1.0
    assert idf_linf(idf) == max(abs(v) for v in idf)


# ---------------------------------------------------------------------------
# delta_idf
# ---------------------------------------------------------------------------
def test_an_unchanged_corpus_moves_no_idf_at_all() -> None:
    """An unchanged corpus gives a delta of positive zero.

    The two logarithms are the same call on the same arguments, so the
    subtraction is exact.
    """
    assert same_bits(delta_idf(3, 3, 10, 10), 0.0)


def test_a_token_becoming_rarer_raises_its_idf() -> None:
    """A rarer token gains idf, the direction section 4.1 depends on.

    The delta is a difference of two exact logarithms rather than one logarithm
    of a ratio of ratios, matching the expression as written.
    """
    assert delta_idf(5, 2, 10, 10) > 0.0
    assert delta_idf(2, 5, 10, 10) < 0.0


def test_the_smoothing_constant_cancels_in_the_difference() -> None:
    """The `+1` smoothing term cancels in the delta.

    It appears in both idf terms, so `delta_idf` does not add it and its result
    can be negative where an idf cannot.
    """
    by_hand = smoothed_idf_one(2, 10) - smoothed_idf_one(5, 10)
    assert delta_idf(5, 2, 10, 10) == pytest.approx(by_hand, rel=1e-15)


def test_the_default_keeps_a_term_seen_in_a_single_document() -> None:
    """The default `min_df` of 1 filters nothing.

    Every term appears in at least one document, so a term in exactly one
    document, which has the highest idf in the corpus, survives.
    Rationale: docs/engineering_notes.md#min-df-default-keeps-hapaxes
    """
    docs = [["shared", "once_only"], ["shared", "elsewhere"]]

    vocab = build_vocabulary(docs)
    assert set(vocab.tokens) == {"shared", "once_only", "elsewhere"}
    assert vocab.df[vocab.tokens.index("once_only")] == 1
    assert vocab.n_discarded == 0, "the default threshold discards nothing"


def test_the_default_and_an_explicit_threshold_of_one_agree() -> None:
    """Naming the default `min_df` gives the same vocabulary, digest included.

    The default is a value, not a separate code path.
    """
    docs = [["shared", "once_only"], ["shared", "elsewhere"]]

    assert (
        build_vocabulary(docs).digest()
        == build_vocabulary(docs, VocabularyConfig(min_df=1)).digest()
    )


def test_a_threshold_of_two_does_discard_the_hapax() -> None:
    """At min_df = 2 the same corpus loses the terms the default keeps.

    The default therefore does work on this corpus rather than describing one
    with nothing to filter.
    """
    docs = [["shared", "once_only"], ["shared", "elsewhere"]]

    filtered = build_vocabulary(docs, VocabularyConfig(min_df=2))
    assert set(filtered.tokens) == {"shared"}
    assert filtered.n_discarded == 2


@pytest.mark.parametrize(("df", "n"), [(1, 10), (5, 100), (1, 1), (9741, 9742), (0, 4)])
def test_the_platform_branch_computes_section_two_ones_formula(df: int, n: int) -> None:
    """The platform branch evaluates `log((1 + N) / (1 + df)) + 1` through libm.

    The G13 test asserts only that the two implementations differ, which a branch
    computing a different expression would also satisfy. An independent
    evaluation pins both smoothing terms, the ratio before the log, and the +1.
    """
    assert smoothed_idf_one(df, n, LogImpl.PLATFORM) == math.log((1 + n) / (1 + df)) + 1.0


@pytest.mark.parametrize(("df", "n"), [(1, 10), (5, 100), (9741, 9742), (3, 9742)])
def test_the_two_log_implementations_disagree_only_in_the_last_bits(df: int, n: int) -> None:
    """The two implementations differ by at most two ulps.

    G13 is a statement about rounding, not arithmetic: a wider gap would mean the
    two branches compute different quantities.
    """
    exact = smoothed_idf_one(df, n, LogImpl.CORRECTLY_ROUNDED)
    platform = smoothed_idf_one(df, n, LogImpl.PLATFORM)

    assert abs(ulps_between(exact, platform)) <= 2.0, f"{exact!r} vs {platform!r}"


def test_a_term_in_every_document_still_carries_the_additive_one() -> None:
    """A term in every document has an idf of exactly 1.0 on both branches.

    `df == N` makes the ratio 1 and its logarithm zero. Without the additive `+ 1`
    such a term would weigh nothing; without the two `+ 1` smoothings `df = N = 0`
    would divide by zero instead of being defined.
    """
    for impl in (LogImpl.CORRECTLY_ROUNDED, LogImpl.PLATFORM):
        assert smoothed_idf_one(9742, 9742, impl) == 1.0, impl
        assert smoothed_idf_one(0, 0, impl) == 1.0, "the empty-corpus edge is defined, not a div0"
