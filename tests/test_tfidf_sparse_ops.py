"""TF-IDF construction, sparse primitives, and the scikit-learn cross-check.

The scikit-learn tests are the project's *external oracle*: an independent
implementation of the same weighting scheme, written by other people. Agreeing
with it rules out a whole class of shared-misreading errors that comparing our
own two backends against each other never could.

The algebra that makes the comparison possible is worth stating, because the
paper does not. scikit-learn uses raw counts where section 2.2 uses
``count / L``, and L2-normalises rows where section 2.2 does not normalise at
all. Both differences are *positive per-vector scalings*, and cosine similarity
is invariant under those. So:

* ``idf`` must agree exactly (``smooth_idf=True`` is literally section 2.1);
* the **vectors must differ**, by the scalar ``1 / L_i``;
* the **similarities must agree**.

The middle point is asserted as a guard: if our norms ever came out equal to
sklearn's, it would mean we had accidentally implemented sklearn rather than the
paper.
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

    for i in range(mini_model.n_documents):
        row = mini_model.matrix.row(i)
        if row.nnz == 0:
            continue
        total = Fraction(0)
        for tid, w in zip(row.indices, row.values, strict=True):
            # w = tf * idf, so tf = w / idf recovers the ratio.
            total += Fraction(w / mini_model.idf[tid]).limit_denominator(10**9)
        assert abs(total - 1) < Fraction(1, 10**9)


def test_zero_in_vocabulary_document_is_the_zero_vector(mini_model) -> None:  # type: ignore[no-untyped-def]
    """Section 2.2: "documents whose in-vocabulary token count is zero are
    mapped to the zero vector"."""
    i = mini_model.doc_ids.index("d5")
    assert mini_model.matrix.row(i).nnz == 0
    assert mini_model.norms[i] == 0.0
    assert mini_model.lengths[i] == 0


def test_identical_documents_produce_bit_identical_rows(mini_model) -> None:  # type: ignore[no-untyped-def]
    """d3 and d4 have identical text, so they must tie *exactly*, not nearly."""
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

    Exact because ``L`` is an integer sum and the division is correctly rounded.
    A test that had to allow a tolerance here would be hiding a real defect.
    """
    doc = ["a", "b", "b", "c"]
    m1 = TfidfVectoriser().fit([doc, ["a"], ["b"]])
    m2 = TfidfVectoriser().fit([doc + doc, ["a"], ["b"]])
    assert m1.vocabulary.tokens == m2.vocabulary.tokens
    assert m1.vocabulary.df == m2.vocabulary.df  # df counts documents, not occurrences
    assert m1.matrix.row(0).values == m2.matrix.row(0).values


def test_renaming_tokens_order_preservingly_leaves_scores_unchanged() -> None:
    """An order-preserving bijection cannot change identifiers' relative order,
    so every summation order -- and hence every bit -- is preserved."""
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
# scikit-learn -- the external oracle
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
    """Proof that our formula *is* sklearn's.

    Under ``LogImpl.PLATFORM`` the agreement must be perfect. Any residual
    difference under the default is therefore attributable to the deliberate
    correctly-rounded logarithm (spec_addenda G13), not to a different formula.
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
    for i in range(len(random_corpus)):
        dense = raw[i].toarray()[0]
        row = ours.matrix.row(i)
        for tid, w in zip(row.indices, row.values, strict=True):
            assert abs(ulps_between(dense[tid] / ours.lengths[i], w)) <= 2.0


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
    """Guard against having accidentally implemented sklearn instead of the paper.

    Section 2.2 applies no vector normalisation, and section 6 is explicit that
    none is introduced. If every norm were 1 we would have silently adopted
    sklearn's convention and the section 4.2/4.3 bounds would be measuring the
    wrong geometry.
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
    """Weights are products, not sums, so only the norms can depend on the policy."""
    a = TfidfVectoriser(reduction=Reduction.NAIVE).fit(list(mini_features))
    b = TfidfVectoriser(reduction=Reduction.EXACT).fit(list(mini_features))
    assert a.matrix.values == b.matrix.values
