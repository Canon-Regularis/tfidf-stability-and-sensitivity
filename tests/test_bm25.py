"""BM25: Lucene's IDF and length code, exact weights, fitting and scoring.

Expected weights come from an independent route in exact ``Fraction`` arithmetic,
rounded once, so each test pins the formula and its single rounding together.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from tfidf_stability.similarity.scoring import InvertedIndex, daat_scores, taat_scores
from tfidf_stability.utils.numerics import Reduction, reduce_sum, same_bits
from tfidf_stability.vectorisation.bm25 import (
    Bm25Model,
    Bm25Vectoriser,
    LengthNorm,
    bm25_idf,
    bm25_idf_one,
    bm25_weight,
    byte4_to_int,
    encoded_length,
    int_to_byte4,
)
from tfidf_stability.vectorisation.idf import LogImpl
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

_DOCS = [
    ["apple", "banana", "apple"],
    ["banana", "cherry"],
    [],
    ["apple", "cherry", "cherry", "durian"],
]


def _oracle(idf: float, f: int, dl: int, *, k1: str, b: str, total: int, docs: int) -> float:
    """The weight in exact rational arithmetic, rounded once at the end."""
    k, beta = Fraction(k1), Fraction(b)
    avgdl = Fraction(total, docs)
    sat = Fraction(f) / (f + k * (1 - beta + beta * Fraction(dl) / avgdl))
    return float(Fraction(idf) * sat)


# ---------------------------------------------------------------------------
# Lucene's one-byte length code
# ---------------------------------------------------------------------------
def test_lengths_up_to_39_survive_the_byte_code() -> None:
    assert all(byte4_to_int(int_to_byte4(n)) == n for n in range(40))


@pytest.mark.parametrize(
    ("length", "decoded"), [(40, 40), (41, 40), (42, 42), (43, 42), (56, 56), (59, 56)]
)
def test_longer_lengths_collapse_onto_the_code_below(length: int, decoded: int) -> None:
    assert encoded_length(length, LengthNorm.LUCENE_BYTE) == decoded
    assert encoded_length(length, LengthNorm.EXACT) == length


def test_the_code_spans_one_byte_and_decodes_in_order() -> None:
    assert int_to_byte4(2**31 - 1) == 255
    assert byte4_to_int(255) == 2013265944
    assert all(int_to_byte4(byte4_to_int(c)) == c for c in range(256))
    assert all(byte4_to_int(c) < byte4_to_int(c + 1) for c in range(255))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: int_to_byte4(-1), r"length must lie in \[0, 2\*\*31 - 1\], got -1"),
        (lambda: int_to_byte4(2**31), "got 2147483648"),
        (lambda: byte4_to_int(256), r"code must lie in \[0, 255\], got 256"),
        (lambda: byte4_to_int(-1), "got -1"),
    ],
)
def test_out_of_range_lengths_and_codes_are_refused(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------
def test_the_idf_matches_lucene_by_hand() -> None:
    """D = 2: df = 1 gives ln 2 and df = 2 gives ln(6/5), each correctly rounded."""
    assert bm25_idf_one(1, 2) == float.fromhex("0x1.62e42fefa39efp-1")
    assert bm25_idf_one(2, 2) == pytest.approx(math.log(6 / 5), rel=1e-15)
    assert bm25_idf_one(0, 0) == pytest.approx(math.log(2), rel=1e-15)


def test_the_idf_is_positive_even_for_a_term_in_every_document() -> None:
    assert all(bm25_idf_one(df, 50) > 0.0 for df in range(51))


def test_the_platform_logarithm_is_selectable() -> None:
    assert bm25_idf_one(3, 9, LogImpl.PLATFORM) == math.log(20 / 7)


def test_the_idf_vector_evaluates_each_frequency_alike() -> None:
    values = bm25_idf([3, 1, 3, 0], 9)

    assert values == tuple(bm25_idf_one(d, 9) for d in (3, 1, 3, 0))


@pytest.mark.parametrize(("df", "doc_count"), [(-1, 5), (6, 5)])
def test_a_frequency_outside_the_corpus_is_refused(df: int, doc_count: int) -> None:
    with pytest.raises(ValueError, match=f"need 0 <= df <= doc_count, got df={df}"):
        bm25_idf_one(df, doc_count)


# ---------------------------------------------------------------------------
# One weight
# ---------------------------------------------------------------------------
def test_every_weight_is_the_exact_ratio_rounded_once() -> None:
    rng = random.Random(20261119)
    for _ in range(2000):
        idf = rng.uniform(0.01, 9.0)
        f, dl = rng.randint(1, 40), rng.randint(1, 900)
        total, docs = rng.randint(dl, 10**6), rng.randint(1, 5000)
        k1, b = rng.choice(["0", "0.9", "1.2", "2"]), rng.choice(["0", "0.4", "0.75", "1"])

        got = bm25_weight(
            idf, f, dl, k1=Fraction(k1), b=Fraction(b), total_length=total, doc_count=docs
        )

        assert same_bits(got, _oracle(idf, f, dl, k1=k1, b=b, total=total, docs=docs)), (
            idf,
            f,
            dl,
            k1,
            b,
        )


def test_the_smallest_legal_operands_are_weighted() -> None:
    """A zero length and a one-feature corpus are valid operands."""
    weight = bm25_weight(
        1.0, 1, 0, k1=Fraction(9, 10), b=Fraction(2, 5), total_length=1, doc_count=1
    )

    assert same_bits(weight, _oracle(1.0, 1, 0, k1="0.9", b="0.4", total=1, docs=1))


def test_k1_of_zero_returns_the_idf_itself() -> None:
    idf = bm25_idf_one(2, 7)
    kwargs = {"k1": Fraction(0), "b": Fraction(3, 4), "total_length": 30, "doc_count": 7}

    assert all(bm25_weight(idf, f, dl, **kwargs) == idf for f, dl in [(1, 1), (5, 40)])


def test_b_of_zero_ignores_the_document_length() -> None:
    kwargs = {"k1": Fraction(9, 10), "b": Fraction(0), "total_length": 30, "doc_count": 7}

    assert bm25_weight(1.5, 2, 3, **kwargs) == bm25_weight(1.5, 2, 300, **kwargs)


def test_the_weight_rises_with_the_count_and_falls_with_the_length() -> None:
    kwargs = {"k1": Fraction(9, 10), "b": Fraction(2, 5), "total_length": 300, "doc_count": 10}

    assert bm25_weight(1.0, 2, 30, **kwargs) > bm25_weight(1.0, 1, 30, **kwargs)
    assert bm25_weight(1.0, 2, 60, **kwargs) < bm25_weight(1.0, 2, 30, **kwargs)


def test_a_byte_coded_length_below_the_count_is_still_weighted() -> None:
    """One term 41 times: Lucene codes the length as 40, below the count."""
    weight = bm25_weight(
        1.0, 41, 40, k1=Fraction(9, 10), b=Fraction(2, 5), total_length=41, doc_count=1
    )

    assert same_bits(weight, _oracle(1.0, 41, 40, k1="0.9", b="0.4", total=41, docs=1))


@pytest.mark.parametrize(
    ("count", "length", "total", "docs"), [(0, 3, 9, 3), (1, -1, 9, 3), (1, 3, 0, 3), (1, 3, 9, 0)]
)
def test_an_impossible_weight_is_refused(count: int, length: int, total: int, docs: int) -> None:
    with pytest.raises(ValueError, match="need count >= 1, length >= 0 and positive totals"):
        bm25_weight(
            1.0, count, length, k1=Fraction(1), b=Fraction(0), total_length=total, doc_count=docs
        )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def test_fitting_weights_every_entry_from_its_count_and_length() -> None:
    model = Bm25Vectoriser().fit(_DOCS, ["a", "b", "c", "d"])

    assert model.lengths == (3, 2, 0, 4)
    assert (model.doc_count, model.total_length) == (3, 9), "the empty document is not counted"
    assert model.avgdl == Fraction(3)
    assert model.norms == (1.0, 1.0, 1.0, 1.0)
    for i in range(model.n_documents):
        lo, hi = model.matrix.indptr[i], model.matrix.indptr[i + 1]
        for j in range(lo, hi):
            t = model.matrix.indices[j]
            assert model.idf[t] == bm25_idf_one(model.vocabulary.df[t], 3)
            expected = _oracle(
                model.idf[t], model.counts[j], model.lengths[i], k1="0.9", b="0.4", total=9, docs=3
            )
            assert same_bits(model.matrix.values[j], expected)


def test_a_one_feature_document_counts_towards_the_corpus() -> None:
    model = Bm25Vectoriser().fit([["solo"], [], ["solo", "pair", "pair"]])

    assert model.lengths == (1, 0, 3)
    assert model.doc_count == 2
    assert model.avgdl == Fraction(2)
    solo = model.vocabulary.id_of("solo")
    assert model.idf[solo] == bm25_idf_one(2, 2), "D is 2: the one-feature document counts"


def test_counts_lengths_and_vocabulary_agree_with_the_tfidf_fit() -> None:
    bm25 = Bm25Vectoriser().fit(_DOCS)
    tfidf = TfidfVectoriser().fit(_DOCS)

    assert bm25.vocabulary == tfidf.vocabulary
    assert bm25.lengths == tfidf.lengths
    assert bm25.matrix.indptr == tfidf.matrix.indptr
    assert bm25.matrix.indices == tfidf.matrix.indices
    assert bm25.counts == (2, 1, 1, 1, 1, 2, 1)
    assert bm25.document(2).nnz == 0
    assert bm25.n_features == len(bm25.vocabulary)


def test_the_byte_code_ties_documents_whose_lengths_differ() -> None:
    """Lengths 40 and 41 share one code, so equal counts give equal weights."""
    docs = [["x"] + ["pad"] * 39, ["x"] + ["fill"] * 40, ["y"]]

    lossy = Bm25Vectoriser(length_norm=LengthNorm.LUCENE_BYTE).fit(docs)
    exact = Bm25Vectoriser().fit(docs)
    x = lossy.vocabulary.id_of("x")

    def weight_of(model: Bm25Model, doc: int) -> float:
        row = model.document(doc)
        return row.values[row.indices.index(x)]

    assert weight_of(lossy, 0) == weight_of(lossy, 1)
    assert weight_of(exact, 0) > weight_of(exact, 1)


def test_a_query_term_weighs_by_its_count_and_unknown_terms_drop() -> None:
    model = Bm25Vectoriser().fit(_DOCS)
    query = Bm25Vectoriser.transform_query(["cherry", "apple", "cherry", "zebra"], model)

    ids = {
        model.vocabulary.token_of(t): v for t, v in zip(query.indices, query.values, strict=True)
    }
    assert ids == {"apple": 1.0, "cherry": 2.0}


@pytest.mark.parametrize("policy", list(Reduction))
def test_term_and_document_at_a_time_give_the_per_document_sum(policy: Reduction) -> None:
    """Unit norms leave each dot product as it is, for every summation policy."""
    model = Bm25Vectoriser(reduction=policy).fit(_DOCS)
    query = Bm25Vectoriser.transform_query(["apple", "cherry", "cherry"], model)
    index = InvertedIndex.from_csr(model.matrix)

    taat = taat_scores(query, index, model.norms, policy, query_norm=Bm25Model.QUERY_NORM)
    daat = daat_scores(query, model.matrix, model.norms, policy, query_norm=Bm25Model.QUERY_NORM)
    by_hand = []
    for i in range(model.n_documents):
        row = dict(zip(model.document(i).indices, model.document(i).values, strict=True))
        products = [
            q * row[t] for t, q in zip(query.indices, query.values, strict=True) if t in row
        ]
        by_hand.append(reduce_sum(products, policy) if products else 0.0)

    assert all(same_bits(a, b) for a, b in zip(taat, by_hand, strict=True))
    assert all(same_bits(a, b) for a, b in zip(daat, by_hand, strict=True))
    assert taat[2] == 0.0, "the empty document scores zero"


def test_intermediates_name_every_quantity_behind_a_weight() -> None:
    model = Bm25Vectoriser(length_norm="lucene_byte").fit(_DOCS, ["a", "b", "c", "d"])
    record = model.intermediates(0)

    assert record["doc_id"] == "a"
    assert (record["in_vocabulary_length"], record["encoded_length"]) == (3, 3)
    assert [(t["token"], t["count"]) for t in record["terms"]] == [("apple", 2), ("banana", 1)]
    assert record["terms"][0]["weight"] == model.document(0).values[0]
    last = model.intermediates(3)
    assert [(t["token"], t["count"]) for t in last["terms"]] == [
        ("apple", 1),
        ("cherry", 2),
        ("durian", 1),
    ], "a later row reads its counts from its own offset"


# ---------------------------------------------------------------------------
# Parameters and identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"k1": 0.9}, TypeError, "k1 must be a decimal string such as '0.9', got 0.9"),
        ({"b": "abc"}, ValueError, "b is not a decimal number: 'abc'"),
        ({"k1": "nan"}, ValueError, "k1 must be finite, got 'nan'"),
        ({"k1": "-1"}, ValueError, "k1 must be non-negative, got '-1'"),
        ({"b": "1.5"}, ValueError, r"b must lie in \[0, 1\], got '1.5'"),
        ({"b": "-0.1"}, ValueError, r"b must lie in \[0, 1\], got '-0.1'"),
    ],
)
def test_bad_parameters_are_refused(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        Bm25Vectoriser(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(("k1", "b"), [("0", "0.4"), ("0.9", "0"), ("0.9", "1")])
def test_the_closed_ends_of_each_range_are_accepted(k1: str, b: str) -> None:
    vectoriser = Bm25Vectoriser(k1=k1, b=b)

    assert vectoriser.fit(_DOCS).k1 == Fraction(k1)


def test_enum_spellings_are_coerced() -> None:
    vectoriser = Bm25Vectoriser(length_norm="lucene_byte", log_impl="platform", reduction="exact")

    assert vectoriser.length_norm is LengthNorm.LUCENE_BYTE
    assert vectoriser.log_impl is LogImpl.PLATFORM
    assert vectoriser.reduction is Reduction.EXACT
    assert str(LengthNorm.EXACT) == "exact"


def test_mismatched_identifiers_are_refused() -> None:
    with pytest.raises(ValueError, match="doc_ids has length 1 but there are 4 documents"):
        Bm25Vectoriser().fit(_DOCS, ["a"])


def test_the_digest_follows_the_parameters_and_not_the_policy() -> None:
    base = Bm25Vectoriser().fit(_DOCS).digest()

    assert Bm25Vectoriser(k1="0.90", b="0.40").fit(_DOCS).digest() == base
    assert Bm25Vectoriser(reduction=Reduction.EXACT).fit(_DOCS).digest() == base
    assert Bm25Vectoriser(k1="1.2").fit(_DOCS).digest() != base
    assert Bm25Vectoriser(b="0.75").fit(_DOCS).digest() != base
    assert (
        Bm25Vectoriser(length_norm=LengthNorm.LUCENE_BYTE).fit([["x"] * 41, ["y"]]).digest()
        != Bm25Vectoriser().fit([["x"] * 41, ["y"]]).digest()
    )
    assert base != TfidfVectoriser().fit(_DOCS).digest()
