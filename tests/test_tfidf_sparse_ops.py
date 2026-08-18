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

import pytest

from tfidf_stability.similarity.cosine import cosine
from tfidf_stability.utils.numerics import Reduction, same_bits, ulps_between
from tfidf_stability.vectorisation.idf import LogImpl
from tfidf_stability.vectorisation.sparse import CsrMatrix, SparseVector, dot, l2_norm
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

sklearn = pytest.importorskip("sklearn", reason="external oracle")
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import linear_kernel  # noqa: E402

_IDENTITY_ANALYSER = list  # bypass sklearn's own tokeniser; we supply features


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
    from tfidf_stability.vectorisation.idf import smoothed_idf_one
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
