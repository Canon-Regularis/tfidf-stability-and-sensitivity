"""TF-IDF construction, sparse primitives, and the scikit-learn cross-check.

scikit-learn is the external oracle: an independent implementation of the same
weighting scheme, which rules out the shared-misreading errors that comparing
our own two backends against each other cannot.

The algebra the paper leaves out. scikit-learn uses raw counts where section 2.2
uses ``count / L``, and L2-normalises rows where section 2.2 does not normalise
at all. Both differences are positive per-vector scalings, and cosine similarity
is invariant under those. So:

* ``idf`` must agree exactly (``smooth_idf=True`` is literally section 2.1);
* the **vectors must differ**, by the scalar ``1 / L_i``;
* the **similarities must agree**.

The middle point is asserted as a guard: norms equal to sklearn's would mean we
had implemented sklearn rather than the paper.
"""

from __future__ import annotations

import math
import random
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tfidf_stability.similarity.cosine import cosine
from tfidf_stability.utils.numerics import Reduction, same_bits, sqrt, ulps_between
from tfidf_stability.vectorisation.idf import LogImpl, smoothed_idf_one
from tfidf_stability.vectorisation.sparse import CsrMatrix, SparseVector, cosine_of, dot, l2_norm
from tfidf_stability.vectorisation.tf import term_frequencies
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

sklearn = pytest.importorskip("sklearn", reason="external oracle")
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import linear_kernel  # noqa: E402

_IDENTITY_ANALYSER = list  # bypass sklearn's own tokeniser; we supply features


def _two_token_vocabulary():  # type: ignore[no-untyped-def]
    """A vocabulary of exactly `a` and `b`, built rather than stubbed.

    Local to this file by house convention; two tokens is the smallest size at
    which a frequency can be something other than one.
    """
    from tfidf_stability.vectorisation.vocabulary import build_vocabulary

    return build_vocabulary([["a", "b"], ["a"]])


def _two_document_model():  # type: ignore[no-untyped-def]
    """A fitted model over the same two documents."""
    return TfidfVectoriser().fit([["a", "b"], ["a"]], ["d0", "d1"])


@pytest.fixture(scope="module")
def random_corpus() -> list[list[str]]:
    rng = random.Random(11)
    alpha = [f"t{i}" for i in range(60)]
    return [[rng.choice(alpha) for _ in range(rng.randint(1, 25))] for _ in range(40)]


# ---------------------------------------------------------------------------
# Sparse primitives
# ---------------------------------------------------------------------------
def test_sparse_vector_is_canonical_from_a_mapping() -> None:
    """Sorting at construction is where dict iteration order stops mattering."""
    v = SparseVector.from_mapping({7: 1.0, 2: 2.0, 5: 3.0}, dim=10)
    assert v.indices == (2, 5, 7)
    assert v.values == (2.0, 3.0, 1.0)
    assert v.is_canonical()


def test_sparse_vector_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        SparseVector(indices=(0, 1), values=(1.0,), dim=4)


def test_dense_round_trip() -> None:
    v = SparseVector.from_mapping({1: 2.5, 3: -1.0}, dim=5)
    assert v.to_dense() == [0.0, 2.5, 0.0, -1.0, 0.0]


def test_dot_ignores_non_overlapping_support() -> None:
    a = SparseVector.from_mapping({0: 1.0, 1: 2.0, 5: 3.0}, 8)
    b = SparseVector.from_mapping({1: 4.0, 2: 9.0, 5: 1.0}, 8)
    assert dot(a, b) == 2.0 * 4.0 + 3.0 * 1.0


def test_l2_norm_of_exact_values_is_exact() -> None:
    assert l2_norm(SparseVector.from_mapping({0: 3.0, 1: 4.0}, 4)) == 5.0
    assert l2_norm(SparseVector.zero(4)) == 0.0


def test_csr_round_trips_its_rows() -> None:
    rows = [
        SparseVector.from_mapping({0: 1.0, 2: 2.0}, 4),
        SparseVector.zero(4),
        SparseVector.from_mapping({3: 5.0}, 4),
    ]
    m = CsrMatrix.from_rows(rows, 4)
    assert m.is_canonical()
    assert m.nnz == 3
    assert [m.row(i).values for i in range(3)] == [r.values for r in rows]
    assert m.row_norms()[1] == 0.0


def test_csr_rejects_a_row_of_the_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="does not match n_cols"):
        CsrMatrix.from_rows([SparseVector.zero(3)], 4)


# ---------------------------------------------------------------------------
# TF-IDF construction
# ---------------------------------------------------------------------------
def test_tf_sums_to_one_over_in_vocabulary_tokens(mini_model) -> None:  # type: ignore[no-untyped-def]
    """Exactly 1 as a rational; ``L`` is an integer so no rounding enters it."""
    from fractions import Fraction

    checked = 0
    for i in range(mini_model.n_documents):
        row = mini_model.matrix.row(i)
        if row.nnz == 0:
            continue
        checked += 1
        total = Fraction(0)
        for tid, w in zip(row.indices, row.values, strict=True):
            # w = tf * idf; dividing back recovers tf to within the rational
            # limit imposed below, which is all this total needs.
            total += Fraction(w / mini_model.idf[tid]).limit_denominator(10**9)
        assert abs(total - 1) < Fraction(1, 10**9)
    # Every document being the zero vector would skip the loop body entirely.
    assert checked > 0, "no document had an in-vocabulary token"


def test_zero_in_vocabulary_document_is_the_zero_vector(mini_model) -> None:  # type: ignore[no-untyped-def]
    """Section 2.2: "documents whose in-vocabulary token count is zero are
    mapped to the zero vector"."""
    i = mini_model.doc_ids.index("d5")
    assert mini_model.matrix.row(i).nnz == 0
    assert mini_model.norms[i] == 0.0
    assert mini_model.lengths[i] == 0


def test_identical_documents_produce_bit_identical_rows(mini_model) -> None:  # type: ignore[no-untyped-def]
    """d3 and d4 have identical text, so they must tie bitwise."""
    a = mini_model.matrix.row(mini_model.doc_ids.index("d3"))
    b = mini_model.matrix.row(mini_model.doc_ids.index("d4"))
    assert a.indices == b.indices
    assert a.values == b.values
    assert same_bits(
        mini_model.norms[mini_model.doc_ids.index("d3")],
        mini_model.norms[mini_model.doc_ids.index("d4")],
    )


def test_query_is_embedded_with_the_corpus_idf(mini_model) -> None:
    """Section 3 and spec_addenda G12: no IDF recomputation, no vocabulary growth."""
    q = TfidfVectoriser.transform_query(["quick", "brown", "zzz_unseen"], mini_model)
    assert q.dim == len(mini_model.vocabulary)
    assert all(0 <= i < len(mini_model.vocabulary) for i in q.indices)
    assert mini_model.vocabulary.id_of("zzz_unseen") is None


def test_out_of_vocabulary_query_embeds_to_zero(mini_model) -> None:
    q = TfidfVectoriser.transform_query(["zzz", "yyy"], mini_model)
    assert q.nnz == 0
    assert cosine(q, mini_model.document(0), u_norm=0.0, v_norm=mini_model.norms[0]) == 0.0


def test_doc_ids_length_is_validated(mini_features) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="doc_ids has length"):
        TfidfVectoriser().fit(list(mini_features), ["only-one"])


# ---------------------------------------------------------------------------
# Metamorphic relations
# ---------------------------------------------------------------------------
def test_duplicating_a_document_leaves_tf_exactly_unchanged() -> None:
    """``(2c)/(2L)`` and ``c/L`` are the same rational, so the same double.

    ``L`` is an integer sum and the division is correctly rounded, so the
    equality is exact and a tolerance here would be hiding a defect.
    """
    doc = ["a", "b", "b", "c"]
    m1 = TfidfVectoriser().fit([doc, ["a"], ["b"]])
    m2 = TfidfVectoriser().fit([doc + doc, ["a"], ["b"]])
    assert m1.vocabulary.tokens == m2.vocabulary.tokens
    assert m1.vocabulary.df == m2.vocabulary.df  # df counts documents, not occurrences
    assert m1.matrix.row(0).values == m2.matrix.row(0).values


def test_renaming_tokens_order_preservingly_leaves_scores_unchanged() -> None:
    """An order-preserving bijection leaves the identifiers' relative order
    alone, so every summation order, and hence every bit, is preserved."""
    docs = [["aa", "bb"], ["bb", "cc"], ["aa", "cc", "cc"]]
    renamed = [[{"aa": "am", "bb": "bm", "cc": "cm"}[t] for t in d] for d in docs]
    a = TfidfVectoriser().fit(docs)
    b = TfidfVectoriser().fit(renamed)
    assert a.matrix.values == b.matrix.values
    assert a.norms == b.norms


def test_corpus_reordering_leaves_every_document_bit_identical() -> None:
    docs = [["a", "b"], ["b", "c"], ["a", "c", "c"], ["d"]]
    fwd = TfidfVectoriser().fit(docs, ["0", "1", "2", "3"])
    rev = TfidfVectoriser().fit(list(reversed(docs)), ["3", "2", "1", "0"])
    for i, doc_id in enumerate(fwd.doc_ids):
        j = rev.doc_ids.index(doc_id)
        assert fwd.matrix.row(i).values == rev.matrix.row(j).values
        assert same_bits(fwd.norms[i], rev.norms[j])


# ---------------------------------------------------------------------------
# scikit-learn: the external oracle
# ---------------------------------------------------------------------------
@pytest.mark.sklearn
def test_vocabulary_matches_sklearn(random_corpus: list[list[str]]) -> None:
    sk = TfidfVectorizer(analyzer=_IDENTITY_ANALYSER, smooth_idf=True).fit(random_corpus)
    ours = TfidfVectoriser().fit(random_corpus)
    assert tuple(sk.get_feature_names_out()) == ours.vocabulary.tokens


@pytest.mark.sklearn
def test_idf_matches_sklearn_exactly_under_the_platform_log(
    random_corpus: list[list[str]],
) -> None:
    """Our formula is sklearn's.

    Under ``LogImpl.PLATFORM`` the agreement is bitwise, so any difference under
    the default comes from the correctly-rounded logarithm (spec_addenda G13)
    rather than from a different formula.
    """
    sk = TfidfVectorizer(analyzer=_IDENTITY_ANALYSER, smooth_idf=True).fit(random_corpus)
    ours = TfidfVectoriser(log_impl=LogImpl.PLATFORM).fit(random_corpus)
    assert all(same_bits(a, b) for a, b in zip(sk.idf_, ours.idf.values, strict=True))


@pytest.mark.sklearn
def test_correctly_rounded_idf_differs_from_sklearn_only_by_rounding(
    random_corpus: list[list[str]],
) -> None:
    """The default differs from sklearn by at most 1 ulp, in a minority of entries."""
    sk = TfidfVectorizer(analyzer=_IDENTITY_ANALYSER, smooth_idf=True).fit(random_corpus)
    ours = TfidfVectoriser().fit(random_corpus)
    diffs = [
        abs(ulps_between(a, b))
        for a, b in zip(sk.idf_, ours.idf.values, strict=True)
        if not same_bits(a, b)
    ]
    assert diffs, "expected the correctly-rounded log to differ somewhere"
    assert max(diffs) <= 1.0
    assert len(diffs) < len(ours.idf.values) / 2


@pytest.mark.sklearn
def test_weights_equal_sklearn_unnormalised_divided_by_length(
    random_corpus: list[list[str]],
) -> None:
    """The exact algebraic relation ``w_ours = x_sklearn / L_i``."""
    raw = TfidfVectorizer(analyzer=_IDENTITY_ANALYSER, smooth_idf=True, norm=None).fit_transform(
        random_corpus
    )
    ours = TfidfVectoriser(log_impl=LogImpl.PLATFORM).fit(random_corpus)
    compared = 0
    for i in range(len(random_corpus)):
        dense = raw[i].toarray()[0]
        row = ours.matrix.row(i)
        for tid, w in zip(row.indices, row.values, strict=True):
            assert abs(ulps_between(dense[tid] / ours.lengths[i], w)) <= 2.0
            compared += 1
    # An all-empty matrix would satisfy the loop without comparing a weight.
    assert compared > 0, "no weight was compared against sklearn"


@pytest.mark.sklearn
def test_cosine_similarities_match_sklearn(random_corpus: list[list[str]]) -> None:
    """The headline cross-check: the scale factors cancel, so scores agree."""
    X = TfidfVectorizer(analyzer=_IDENTITY_ANALYSER, smooth_idf=True, norm="l2").fit_transform(
        random_corpus
    )
    K = linear_kernel(X)
    ours = TfidfVectoriser().fit(random_corpus)
    worst = 0.0
    for i in range(len(random_corpus)):
        for j in range(len(random_corpus)):
            c = cosine(
                ours.document(i),
                ours.document(j),
                u_norm=ours.norms[i],
                v_norm=ours.norms[j],
            )
            worst = max(worst, abs(c - K[i, j]))
    assert worst < 1e-14, f"max disagreement {worst:.3e}"


@pytest.mark.sklearn
def test_our_vectors_are_not_l2_normalised(random_corpus: list[list[str]]) -> None:
    """Guard against having implemented sklearn instead of the paper.

    Section 2.2 applies no vector normalisation and section 6 says none is
    introduced. Norms of 1 everywhere would mean sklearn's convention had been
    adopted and the section 4.2/4.3 bounds were measuring the wrong geometry.
    """
    ours = TfidfVectoriser().fit(random_corpus)
    assert all(abs(n - 1.0) > 1e-9 for n in ours.norms)
    assert max(ours.norms) / min(ours.norms) > 2.0


# ---------------------------------------------------------------------------
# Reduction policies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(Reduction))
def test_model_builds_under_every_reduction_policy(mini_features, policy: Reduction) -> None:  # type: ignore[no-untyped-def]
    m = TfidfVectoriser(reduction=policy).fit(list(mini_features))
    assert m.reduction is policy
    assert all(math.isfinite(n) for n in m.norms)


def test_reduction_policy_affects_norms_but_not_weights(mini_features) -> None:  # type: ignore[no-untyped-def]
    """Weights are products rather than sums, so only the norms see the policy."""
    a = TfidfVectoriser(reduction=Reduction.NAIVE).fit(list(mini_features))
    b = TfidfVectoriser(reduction=Reduction.EXACT).fit(list(mini_features))
    assert a.matrix.values == b.matrix.values


def test_intermediates_reports_the_tf_that_was_actually_used() -> None:
    """``inspect`` must not print a tf one ulp from the one behind the weight.

    The field was computed as ``weight / idf`` under a comment claiming that
    division is exact. Two roundings do not cancel: over a sweep of 184,080
    realistic ``(N, df, L, count)`` combinations the round trip missed by an ulp
    in 9.01% of them. Scores here are compared on raw bit patterns, so the
    reported intermediate was off by the quantity under study.
    """
    from tfidf_stability.vectorisation.tfidf import _exact_tf

    naive_wrong = 0
    cases = 0
    for n_docs in (100, 610, 9742):
        for df in range(1, 40):
            idf = smoothed_idf_one(df, n_docs)
            if idf == 0.0:
                continue
            for length in range(1, 30):
                for count in range(1, length + 1):
                    tf = count / length
                    weight = tf * idf
                    cases += 1
                    naive_wrong += weight / idf != tf
                    assert same_bits(_exact_tf(weight, idf, length), tf)

    assert cases > 10_000, "the sweep must be large enough to be worth anything"
    assert naive_wrong > 0, (
        "the naive weight/idf must fail somewhere, or this test proves nothing "
        "and the reconstruction it guards is unnecessary"
    )


def test_intermediates_tf_matches_the_term_frequency_vector(mini_model, mini_features) -> None:  # type: ignore[no-untyped-def]
    """End to end on the fixture, against the tf vector the fit actually built."""
    from tfidf_stability.vectorisation.tf import term_frequencies

    compared = 0
    for i, features in enumerate(mini_features):
        vector, _length = term_frequencies(features, mini_model.vocabulary)
        truth = dict(zip(vector.indices, vector.values, strict=True))
        for term in mini_model.intermediates(i)["terms"]:
            expected = truth.get(term["term_id"])
            if expected is not None:
                assert same_bits(term["tf"], expected), term["token"]
                compared += 1
    # The lookup returning None throughout would leave the comparison unreached.
    assert compared > 0, "no intermediate term matched the tf vector"


# ---------------------------------------------------------------------------
# The canonical form is a checkable invariant, so its rejections must be checked
# ---------------------------------------------------------------------------
# CsrMatrix is frozen with no validating __post_init__, and save_load.py records
# that of 2,715 mutated containers accepted, 425 carried a non-monotonic indptr.
# is_canonical is the opt-in check that catches those, and a checker whose reject
# arms have never fired is decoration: each one is constructed directly here, no
# mocking required.
def _canonical() -> CsrMatrix:
    """Two rows, ascending within each, every index in range."""
    return CsrMatrix(
        indptr=(0, 2, 3), indices=(0, 2, 1), values=(1.0, 2.0, 3.0), n_rows=2, n_cols=3
    )


def test_a_well_formed_matrix_is_canonical_so_the_rejections_below_mean_something() -> None:
    assert _canonical().is_canonical(), "the baseline must pass or every rejection is vacuous"


def test_an_indptr_of_the_wrong_length_is_not_canonical() -> None:
    bad = CsrMatrix(indptr=(0, 2), indices=(0, 2, 1), values=(1.0, 2.0, 3.0), n_rows=2, n_cols=3)
    assert not bad.is_canonical(), "indptr must hold n_rows + 1 entries"


def test_an_indptr_not_starting_at_zero_is_not_canonical() -> None:
    bad = CsrMatrix(indptr=(1, 2, 3), indices=(0, 2, 1), values=(1.0, 2.0, 3.0), n_rows=2, n_cols=3)
    assert not bad.is_canonical()


def test_an_indptr_not_ending_at_the_stored_count_is_not_canonical() -> None:
    bad = CsrMatrix(indptr=(0, 2, 9), indices=(0, 2, 1), values=(1.0, 2.0, 3.0), n_rows=2, n_cols=3)
    assert not bad.is_canonical()


def test_more_indices_than_values_is_not_canonical() -> None:
    """indptr must still agree with the value count, or an earlier arm fires
    first and this one is never reached."""
    bad = CsrMatrix(indptr=(0, 2), indices=(0, 1, 2), values=(1.0, 2.0), n_rows=1, n_cols=3)
    assert bad.indptr[-1] == len(bad.values), "the earlier arm must not be the one rejecting"
    assert not bad.is_canonical()


def test_a_decreasing_indptr_segment_is_not_canonical() -> None:
    """The 425-of-2,715 case: a row whose end precedes its start.

    The last indptr entry still matches the value count, so the monotonicity
    arm is what rejects this rather than the length arm above it.
    """
    bad = CsrMatrix(
        indptr=(0, 2, 1, 3), indices=(0, 1, 2), values=(1.0, 2.0, 3.0), n_rows=3, n_cols=3
    )
    assert bad.indptr[-1] == len(bad.values), "the earlier arm must not be the one rejecting"
    assert not bad.is_canonical()


def test_a_repeated_column_within_a_row_is_not_canonical() -> None:
    """Strictly ascending, not merely sorted: a duplicate would be summed twice
    by a merge that assumes each column appears once."""
    bad = CsrMatrix(indptr=(0, 2), indices=(1, 1), values=(1.0, 2.0), n_rows=1, n_cols=3)
    assert not bad.is_canonical()


def test_a_descending_column_pair_within_a_row_is_not_canonical() -> None:
    bad = CsrMatrix(indptr=(0, 2), indices=(2, 1), values=(1.0, 2.0), n_rows=1, n_cols=3)
    assert not bad.is_canonical()


def test_a_column_index_at_n_cols_is_out_of_range_while_one_below_is_not() -> None:
    """The off-by-one, from both sides, so a future edit cannot lose an arm."""
    inside = CsrMatrix(indptr=(0, 1), indices=(2,), values=(1.0,), n_rows=1, n_cols=3)
    outside = CsrMatrix(indptr=(0, 1), indices=(3,), values=(1.0,), n_rows=1, n_cols=3)
    assert inside.is_canonical(), "n_cols - 1 is the last legal column"
    assert not outside.is_canonical(), "n_cols itself is one past the end"


def test_a_negative_column_index_is_not_canonical() -> None:
    bad = CsrMatrix(indptr=(0, 1), indices=(-1,), values=(1.0,), n_rows=1, n_cols=3)
    assert not bad.is_canonical()


def test_an_empty_matrix_with_no_rows_is_canonical_rather_than_malformed() -> None:
    assert CsrMatrix(indptr=(0,), indices=(), values=(), n_rows=0, n_cols=0).is_canonical()


def test_a_row_of_length_zero_between_two_populated_rows_is_canonical() -> None:
    """An empty row is a document with no in-vocabulary tokens, which section 2.2
    maps to the zero vector: common, not a corruption."""
    matrix = CsrMatrix(indptr=(0, 1, 1, 2), indices=(0, 1), values=(1.0, 2.0), n_rows=3, n_cols=2)
    assert matrix.is_canonical()
    assert matrix.row(1).nnz == 0


# ---------------------------------------------------------------------------
# The module-level cosine, and the sparse vector protocol
# ---------------------------------------------------------------------------
def test_cosine_of_is_pinned_to_dot_over_the_product_of_norms() -> None:
    """The docstring forbids the algebraically equal reassociations, so the two
    are shown to actually differ on a chosen input rather than assumed to."""
    # Chosen by search: most inputs give the same double either way, so an
    # arbitrary pair would assert the pinning without being able to detect its
    # loss. These two differ by exactly one ulp.
    u = SparseVector(
        indices=(0, 1, 2),
        values=(0.42451918914251396, 0.8268521246720381, 0.12380196114964559),
        dim=3,
    )
    v = SparseVector(
        indices=(0, 1, 2),
        values=(0.22323896460701453, 0.6274332224055893, 0.9477089424570057),
        dim=3,
    )
    nu, nv = l2_norm(u), l2_norm(v)

    assert same_bits(cosine_of(u, v), dot(u, v) / (nu * nv))
    alternative = (dot(u, v) / nu) / nv
    assert not same_bits(cosine_of(u, v), alternative), (
        "the two forms agreed here, so this input does not separate them and the "
        "pinning is untested"
    )


def test_cosine_of_accepts_precomputed_norms_without_changing_the_value() -> None:
    u = SparseVector(indices=(0, 2), values=(1.0, 2.0), dim=4)
    v = SparseVector(indices=(0, 3), values=(3.0, 4.0), dim=4)
    assert same_bits(cosine_of(u, v), cosine_of(u, v, u_norm=l2_norm(u), v_norm=l2_norm(v)))


def test_cosine_of_a_zero_vector_is_zero_rather_than_a_division() -> None:
    zero = SparseVector(indices=(), values=(), dim=3)
    other = SparseVector(indices=(1,), values=(1.0,), dim=3)
    assert cosine_of(zero, other) == 0.0
    assert cosine_of(other, zero) == 0.0
    assert cosine_of(zero, zero) == 0.0


def test_a_dimension_mismatch_in_dot_is_rejected_not_silently_truncated() -> None:
    u = SparseVector(indices=(0,), values=(1.0,), dim=2)
    v = SparseVector(indices=(0,), values=(1.0,), dim=3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        dot(u, v)


def test_len_reports_the_ambient_dimension_not_the_stored_count() -> None:
    """A sparse vector is a vector of `dim`, most of whose entries are absent."""
    vector = SparseVector(indices=(1,), values=(5.0,), dim=100)
    assert len(vector) == 100
    assert vector.nnz == 1


def test_iterating_yields_index_value_pairs_in_ascending_index_order() -> None:
    vector = SparseVector(indices=(0, 3, 7), values=(1.0, 2.0, 3.0), dim=8)
    assert list(vector) == [(0, 1.0), (3, 2.0), (7, 3.0)]
    assert list(SparseVector(indices=(), values=(), dim=4)) == []


def test_iterating_the_rows_yields_every_row_in_order() -> None:
    matrix = CsrMatrix(
        indptr=(0, 1, 1, 3), indices=(0, 1, 2), values=(1.0, 2.0, 3.0), n_rows=3, n_cols=3
    )
    rows = list(matrix.rows())
    assert len(rows) == 3
    assert [r.nnz for r in rows] == [1, 0, 2], "the empty middle row must still be yielded"
    assert [list(r) for r in rows] == [[(0, 1.0)], [], [(1, 2.0), (2, 3.0)]]


def test_the_tf_reconstruction_refuses_to_divide_by_a_degenerate_denominator() -> None:
    """Neither input can occur in a fitted model -- the smoothed IDF is at least
    1, and a document of zero in-vocabulary length has no row entries to inspect
    -- so the guard exists for a caller reaching the helper directly. Zero
    rather than an exception: `intermediates` is a diagnostic, and one that
    raised while reporting a degenerate document would be useless exactly where
    it is needed.
    """
    from tfidf_stability.vectorisation.tfidf import _exact_tf

    assert _exact_tf(0.0, 0.0, 6) == 0.0
    assert _exact_tf(0.5, 1.5, 0) == 0.0
    assert _exact_tf(0.0, 0.0, 0) == 0.0

    # The guard is a boundary, not a blanket: one step off it the reconstruction
    # runs. This is the case the docstring cites, where weight / idf misses.
    idf = smoothed_idf_one(1, 100)
    tf = 5 / 6
    assert same_bits(_exact_tf(tf * idf, idf, 6), tf)
    assert (tf * idf) / idf != tf, "the premise: the naive round trip is wrong here"


def test_zero_norm_documents_are_exactly_the_ones_with_no_magnitude(mini_model) -> None:  # type: ignore[no-untyped-def]
    """A document whose every token was a stopword embeds to the zero vector.
    Its count is reported in every experiment, because it is the regime the
    section 4.5 tie-break analysis covers: those documents all score exactly
    zero and the ranking among them falls entirely to the tie-break.
    """
    reported = mini_model.zero_norm_documents
    expected = tuple(i for i, n in enumerate(mini_model.norms) if n == 0.0)

    assert reported == expected
    assert reported, "the mini corpus embeds an all-stopword document on purpose"
    assert len(reported) < mini_model.n_documents, "and it is not the whole corpus"
    assert all(mini_model.norms[i] == 0.0 for i in reported)
    assert all(
        mini_model.norms[i] > 0.0 for i in range(mini_model.n_documents) if i not in reported
    )


def test_a_query_is_weighted_by_multiplying_tf_into_idf(mini_model, mini_features) -> None:  # type: ignore[no-untyped-def]
    """`w = tf * idf`, on the query side as on the document side, using the
    fitted model's own idf (G12: nothing is refitted for a query).

    Asserted bitwise against the two factors. Dividing instead would still give
    a plausible ranking -- rarer terms would simply count for less rather than
    more -- and every cosine would still lie in [0, 1].
    """
    from tfidf_stability.vectorisation.tf import term_frequencies

    query = TfidfVectoriser.transform_query(mini_features[0], mini_model)
    tf, _ = term_frequencies(mini_features[0], mini_model.vocabulary)

    assert query.indices == tf.indices
    assert query.dim == len(mini_model.vocabulary)
    assert query.nnz > 0, "the first document has in-vocabulary tokens"
    for term_id, weight, raw in zip(query.indices, query.values, tf.values, strict=True):
        assert same_bits(weight, raw * mini_model.idf[term_id])
        assert (
            not same_bits(weight, raw / mini_model.idf[term_id]) or mini_model.idf[term_id] == 1.0
        )


# ---------------------------------------------------------------------------
# SparseVector: what construction does and does not establish
# ---------------------------------------------------------------------------
# `__post_init__` checks only that the two arrays are parallel. Ascending,
# in-range indices are established by the constructors (`from_mapping` sorts,
# `zero` is empty) and asserted by `is_canonical`, never enforced. Everything
# downstream -- the merge in `dot`, the native postings loop -- assumes them.
def test_construction_checks_only_that_the_arrays_are_parallel() -> None:
    """An index outside the ambient dimension builds without complaint. That is
    why `is_canonical` exists as a separate question rather than an invariant."""
    out_of_range = SparseVector(indices=(5,), values=(1.0,), dim=1)

    assert out_of_range.nnz == 1
    assert not out_of_range.is_canonical()


@pytest.mark.parametrize(
    ("label", "indices", "dim"),
    [
        ("a duplicated index", (1, 1), 4),
        ("descending indices", (2, 1), 4),
        ("a negative index", (-1,), 4),
        ("an index equal to the dimension", (4,), 4),
        ("any index at all in a zero-dimensional space", (0,), 0),
    ],
)
def test_each_way_of_breaking_the_canonical_form_is_detected(
    label: str, indices: tuple[int, ...], dim: int
) -> None:
    """One case per reject arm. Strict ascent makes a support a set, and the
    range check is what keeps `to_dense` and the native loop inside their
    arrays."""
    vector = SparseVector(indices=indices, values=(1.0,) * len(indices), dim=dim)
    assert not vector.is_canonical(), label


def test_a_negative_index_writes_the_wrong_slot_rather_than_raising() -> None:
    """`to_dense` assigns by index, so a negative one lands at the far end of
    the dense array. Silent, and the reason `is_canonical` checks the lower
    bound as well as the upper one."""
    wrong_slot = SparseVector(indices=(-1,), values=(9.0,), dim=3)
    assert wrong_slot.to_dense() == [0.0, 0.0, 9.0]


def test_an_index_at_the_dimension_is_an_index_error_rather_than_a_wrong_slot() -> None:
    """The upper bound is where the failure is loud, and the lower bound is
    where it is silent -- an asymmetry worth having written down."""
    with pytest.raises(IndexError, match="list assignment index out of range"):
        SparseVector(indices=(3,), values=(9.0,), dim=3).to_dense()


def test_a_mapping_is_sorted_whatever_order_it_arrived_in() -> None:
    """The point at which dictionary ordering stops being able to influence any
    downstream number."""
    scrambled = SparseVector.from_mapping({3: 0.3, 1: 0.1, 2: 0.2}, dim=4)

    assert scrambled.indices == (1, 2, 3)
    assert scrambled.values == (0.1, 0.2, 0.3)
    assert scrambled.is_canonical()


def test_a_mapping_larger_than_its_declared_dimension_is_not_checked() -> None:
    """`from_mapping` sorts but does not validate against `dim`, so a key past
    the end builds a non-canonical vector. Pinned as the precondition it is."""
    assert not SparseVector.from_mapping({9: 1.0}, dim=2).is_canonical()


@pytest.mark.parametrize("dim", [0, 1, 100])
def test_the_zero_vector_is_canonical_at_every_dimension(dim: int) -> None:
    """No indices, so nothing can be out of range. Section 2.3 gives it a
    similarity of zero rather than treating it as an error."""
    zero = SparseVector.zero(dim)
    assert zero.nnz == 0
    assert len(zero) == dim
    assert zero.is_canonical()


# ---------------------------------------------------------------------------
# dot: the merge, and the sign of nothing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("left", "right"), [({0: 1.0}, {1: 1.0}), ({}, {0: 1.0}), ({0: 1.0}, {}), ({}, {})]
)
def test_a_dot_product_with_no_shared_terms_is_positive_zero(
    left: dict[int, float], right: dict[int, float]
) -> None:
    """`+0.0` specifically. The result feeds a bitwise comparison against the
    native backend, where `-0.0` is a different value."""
    product = dot(SparseVector.from_mapping(left, 4), SparseVector.from_mapping(right, 4))
    assert same_bits(product, 0.0)


def test_a_negative_zero_term_does_not_make_the_product_negative_zero() -> None:
    """`-0.0 * 1.0` is `-0.0`, but the reduction starts from `+0.0` and
    `0.0 + -0.0` is `+0.0`. So the sign does not survive the sum -- which is
    what makes "the dot product is zero" mean one thing."""
    left = SparseVector.from_mapping({0: -0.0}, 4)
    right = SparseVector.from_mapping({0: 1.0}, 4)
    assert same_bits(dot(left, right), 0.0)


@pytest.mark.parametrize(("left_dim", "right_dim"), [(4, 5), (5, 4), (0, 1)])
def test_a_dot_product_across_dimensions_names_both(left_dim: int, right_dim: int) -> None:
    """Merging two vectors over different vocabularies would silently compare
    term ids that mean different things."""
    with pytest.raises(ValueError, match=f"dimension mismatch: {left_dim} vs {right_dim}"):
        dot(SparseVector.zero(left_dim), SparseVector.zero(right_dim))


def test_the_merge_accumulates_in_ascending_term_order() -> None:
    """The native backend's postings loop produces the same sequence, so the
    order is the specification rather than an implementation detail. Asserted by
    reproducing the fold rather than by trusting the total."""
    left = SparseVector.from_mapping({0: 1e16, 1: 1.0, 2: -1e16}, 4)
    right = SparseVector.from_mapping({0: 1.0, 1: 1.0, 2: 1.0}, 4)

    expected = (1e16 * 1.0 + 1.0 * 1.0) + (-1e16 * 1.0)
    assert same_bits(dot(left, right), expected)


def test_an_infinite_term_against_a_zero_term_gives_nan() -> None:
    """`inf * 0.0` is NaN, and nothing filters it. Another reason non-finite
    weights are refused before they reach a vector."""
    left = SparseVector.from_mapping({0: math.inf}, 4)
    right = SparseVector.from_mapping({0: 0.0}, 4)
    assert math.isnan(dot(left, right))


# ---------------------------------------------------------------------------
# l2_norm: the G18 underflow boundary, measured rather than quoted
# ---------------------------------------------------------------------------
#: `sqrt(DBL_MIN)`, defined in the numerics suite and quoted here rather than
#: re-derived. Above it, squaring a coordinate is safe; below it the square
#: drifts into the subnormals and eventually to zero.
_UNDERFLOW_THRESHOLD = 1.4916681462400413e-154


def test_the_threshold_is_the_one_the_numerics_suite_pins() -> None:
    """Quoted, so the two suites cannot drift apart silently."""
    assert sqrt(sys.float_info.min) == _UNDERFLOW_THRESHOLD


def test_the_norm_is_the_square_root_of_the_sum_of_squares_in_that_order() -> None:
    """No rescaling. A hypot-style formulation resists overflow better and
    produces different digits, and section 6 forbids stabilising
    transformations."""
    assert same_bits(l2_norm(SparseVector.from_mapping({0: 3.0, 1: 4.0}, 4)), 5.0)


def test_a_large_vector_overflows_to_infinity_rather_than_being_rescaled() -> None:
    """Asserted as intended behaviour, not as a defect. The overflow is visible;
    a rescaled norm would be quietly different from the native backend's."""
    huge = SparseVector.from_mapping({0: 1e200, 1: 1e200}, 4)
    assert l2_norm(huge) == math.inf


@pytest.mark.parametrize("scale", [1.0, 1e-100, 1e-154, _UNDERFLOW_THRESHOLD, 1e-160, 1e-162])
def test_above_the_underflow_floor_a_vector_is_similar_to_itself(scale: float) -> None:
    """Down to about `1e-162` the squares stay representable -- subnormal below
    `1e-160`, but non-zero -- and the self-similarity is still exactly one."""
    vector = SparseVector.from_mapping({0: 3.0 * scale, 1: 4.0 * scale}, 4)

    assert l2_norm(vector) > 0.0
    # Within a rounding of one rather than exactly one: `dot / (norm * norm)`
    # rounds three times, so the self-similarity lands either side of 1.0
    # depending on the scale. What the floor guarantees is that it is a
    # similarity at all, not that it is exact.
    assert abs(ulps_between(1.0, cosine_of(vector, vector))) <= 3.0


@pytest.mark.parametrize("scale", [1e-164, 1e-170, 5e-324])
def test_below_the_underflow_floor_a_non_zero_vector_reports_a_zero_norm(scale: float) -> None:
    """The squares flush to zero, so the norm does too -- and `cosine_of` then
    takes its zero-vector branch and reports a similarity of zero for a vector
    that is not zero.

    This is the G18 regime stated at its starkest. The measured floor is around
    `1e-163`, well below the `sqrt(DBL_MIN)` threshold at which *precision*
    starts to degrade: the two are different boundaries and the addendum's
    concern is the first, not this one.
    """
    vector = SparseVector.from_mapping({0: 3.0 * scale, 1: 4.0 * scale}, 4)

    assert vector.nnz == 2, "the vector really does have stored entries"
    assert l2_norm(vector) == 0.0
    assert cosine_of(vector, vector) == 0.0


# ---------------------------------------------------------------------------
# cosine_of: precomputed norms are trusted, not verified
# ---------------------------------------------------------------------------
def test_a_supplied_norm_is_used_as_given_rather_than_checked() -> None:
    """The whole point of passing them is to avoid recomputing, so nothing
    validates them. A norm that is wrong but positive gives a wrong similarity
    with no complaint -- which is why `row_norms` and the scorer must be the
    only things that produce them."""
    unit = SparseVector.from_mapping({0: 1.0}, 4)
    assert cosine_of(unit, unit) == 1.0
    assert cosine_of(unit, unit, u_norm=2.0) == 0.5


def test_a_supplied_zero_norm_short_circuits_a_non_zero_vector_to_zero() -> None:
    """The zero-vector convention is triggered by the norm, not by the support,
    so a zero norm supplied for a non-zero vector takes that branch."""
    unit = SparseVector.from_mapping({0: 1.0}, 4)
    assert cosine_of(unit, unit, u_norm=0.0) == 0.0


@pytest.mark.parametrize(("norm", "check"), [(math.nan, math.isnan), (-1.0, lambda x: x == -1.0)])
def test_a_degenerate_supplied_norm_propagates_rather_than_being_clamped(
    norm: float, check: object
) -> None:
    """NaN propagates and a negative norm yields a negative cosine. Both are
    outside `[0, 1]`, and neither is clamped -- consistent with G24's refusal to
    clamp the ordinary excess."""
    unit = SparseVector.from_mapping({0: 1.0}, 4)
    assert check(cosine_of(unit, unit, u_norm=norm))  # type: ignore[operator]


def test_the_expression_is_the_pinned_one_and_not_an_algebraic_equal() -> None:
    """`dot / (nu * nv)`. `(dot / nu) / nv` and `dot * (1 / (nu * nv))` are
    algebraically equal and numerically different; the native backend matches
    this form, so the grouping is part of the specification."""
    left = SparseVector.from_mapping({0: 1.0, 1: 2.0, 2: 3.0}, 4)
    right = SparseVector.from_mapping({0: 3.0, 1: 2.0, 2: 1.0}, 4)
    nu, nv = l2_norm(left), l2_norm(right)
    numerator = dot(left, right)

    assert same_bits(cosine_of(left, right), numerator / (nu * nv))


# ---------------------------------------------------------------------------
# G24: cosine exceeds one, and is deliberately not clamped
# ---------------------------------------------------------------------------
def test_self_similarity_can_exceed_one_by_a_single_ulp() -> None:
    """`cos(v, v)` is `dot / (norm * norm)`, and the two roundings do not
    cancel. G24 records the effect and rules out clamping: a clamp would change
    published digits and hide the very quantity this project measures.

    Bounded rather than merely observed -- an excess of more than a few ulp
    would mean something other than rounding.
    """
    # A measured witness, not a guessed one: many ordinary vectors -- including
    # {0.1, 0.2, 0.3} -- come out exactly 1.0, so the case has to be chosen
    # rather than assumed.
    witness = SparseVector.from_mapping({0: 2.663, 1: 5.162, 2: 4.109}, 4)
    similarity = cosine_of(witness, witness)

    assert similarity > 1.0, "the witness still exceeds one on this platform"
    assert abs(ulps_between(1.0, similarity)) <= 3.0


@pytest.mark.property
@given(
    st.lists(st.floats(min_value=0.1, max_value=10.0, allow_nan=False), min_size=2, max_size=8),
)
def test_the_excess_over_one_never_grows_beyond_a_few_ulp(values: list[float]) -> None:
    """Searched rather than constructed. The claim is not that the excess never
    happens -- it happens on about a quarter of ordinary vectors -- but that it
    is always rounding and never a defect in the formula."""
    vector = SparseVector.from_mapping(dict(enumerate(values)), len(values))
    similarity = cosine_of(vector, vector)

    assert similarity <= 1.0 or abs(ulps_between(1.0, similarity)) <= 3.0


# ---------------------------------------------------------------------------
# CsrMatrix
# ---------------------------------------------------------------------------
def test_a_matrix_of_no_rows_still_has_its_opening_offset() -> None:
    """`indptr` is one longer than the row count, so an empty matrix has the
    single entry zero rather than none. A bare empty tuple would fail every
    canonical check downstream."""
    empty = CsrMatrix.from_rows([], n_cols=4)

    assert empty.indptr == (0,)
    assert empty.nnz == 0
    assert empty.is_canonical()
    assert empty.row_norms() == ()


@pytest.mark.parametrize("position", [0, 1, 2])
def test_a_row_of_the_wrong_dimension_is_refused_wherever_it_sits(position: int) -> None:
    """The check runs per row rather than on the first, so a mismatch in the
    middle of a corpus is caught rather than assembled into a matrix whose
    columns mean different things in different rows."""
    rows = [SparseVector.zero(4)] * 3
    rows[position] = SparseVector.zero(5)

    with pytest.raises(ValueError, match="row dimension 5 does not match n_cols=4"):
        CsrMatrix.from_rows(rows, n_cols=4)


def test_a_negative_row_index_yields_an_empty_row_rather_than_raising() -> None:
    """`indptr[-1]` is the final offset and `indptr[0]` is zero, so the slice
    runs backwards and comes out empty. Silent, and the fourth instance of a
    negative index being accepted where the positive side is checked.
    """
    matrix = CsrMatrix.from_rows(
        [SparseVector.from_mapping({0: 1.0}, 4), SparseVector.from_mapping({1: 2.0}, 4)], n_cols=4
    )
    assert matrix.row(-1).nnz == 0
    assert matrix.row(0).nnz == 1


def test_a_row_index_past_the_end_is_an_index_error() -> None:
    matrix = CsrMatrix.from_rows([SparseVector.from_mapping({0: 1.0}, 4)], n_cols=4)
    with pytest.raises(IndexError, match="tuple index out of range"):
        matrix.row(1)


# ---------------------------------------------------------------------------
# term_frequencies: what "length" counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "features"), [("nothing at all", []), ("only out-of-vocabulary tokens", ["zzz"])]
)
def test_a_document_with_no_in_vocabulary_tokens_has_no_terms_and_no_length(
    label: str, features: list[str]
) -> None:
    """Both routes to the zero vector. The length is zero as well as the
    support, which is what stops `w / idf * length` dividing by a length the
    document does not have."""
    vocab = _two_token_vocabulary()
    vector, length = term_frequencies(features, vocab)

    assert vector.nnz == 0, label
    assert length == 0
    assert vector.dim == len(vocab)


def test_the_length_counts_in_vocabulary_tokens_only() -> None:
    """`tf = count / length` with `length` the in-vocabulary count, so an
    out-of-vocabulary token must not dilute the frequencies of the tokens that
    did survive -- otherwise a document's weights would depend on how much of it
    the vocabulary happened to drop."""
    vocab = _two_token_vocabulary()
    vector, length = term_frequencies(["a", "zzz", "b"], vocab)

    assert length == 2, "zzz is not counted"
    assert sum(vector.values) == pytest.approx(1.0)


def test_the_frequencies_of_an_in_vocabulary_document_sum_to_one() -> None:
    """`||tf||_1 == 1` exactly. It is the premise of the `1/sqrt(nnz)` norm
    lower bound, so it holds as an identity rather than approximately."""
    vocab = _two_token_vocabulary()
    vector, _ = term_frequencies(["a", "a", "b"], vocab)

    assert vector.values == (2 / 3, 1 / 3)
    assert sum(vector.values) == pytest.approx(1.0)


def test_a_repeated_token_raises_its_frequency_rather_than_its_count() -> None:
    """The vector is frequencies, not counts, so the length divides through."""
    vocab = _two_token_vocabulary()
    once, _ = term_frequencies(["a", "b"], vocab)
    twice, length = term_frequencies(["a", "a", "b"], vocab)

    assert length == 3
    assert twice.values[0] > once.values[0]


# ---------------------------------------------------------------------------
# transform_query: G12, nothing is refitted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("features", [[], ["zzz"], ["zzz", "qqq"]])
def test_a_query_of_unknown_tokens_embeds_to_the_zero_vector(features: list[str]) -> None:
    """Degenerate but legitimate: the ranking then falls entirely to the
    tie-break, which is the regime section 4.5 studies. Not an error."""
    model = _two_document_model()
    query = TfidfVectoriser.transform_query(features, model)

    assert query.nnz == 0
    assert query.dim == len(model.vocabulary)


def test_a_query_uses_the_fitted_models_own_idf_and_never_extends_it() -> None:
    """G12. Refitting on the query would give a token an idf derived from one
    document, and every score in the corpus would move."""
    model = _two_document_model()
    before = tuple(model.idf)

    query = TfidfVectoriser.transform_query(["a", "zzz"], model)

    assert tuple(model.idf) == before, "the model is unchanged"
    assert query.dim == len(model.vocabulary), "and the query lives in its space"
    assert all(i < len(model.vocabulary) for i in query.indices)
