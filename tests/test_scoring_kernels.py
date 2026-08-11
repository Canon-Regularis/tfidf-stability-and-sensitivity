"""The two Python scoring kernels, and their agreement with the cosine reference.

The centrepiece is ``TAAT == DAAT == cosine_against_corpus`` on **raw bit
patterns**. Those three share no loop nesting and no data structure: TAAT walks
postings lists out of an inverted index into a dense accumulator, DAAT merges
each row against the query independently, and ``cosine_against_corpus`` merges
against a materialised sequence of document vectors. Identical binary64 output
from all three leaves very little room for an indexing or accumulation bug to
hide, and it is a far stronger statement than any of them matching a recorded
expectation.

Nothing here uses a tolerance. ``pytest.approx`` compares numbers that are
*close*; the property under test is that they are *the same*, and a tolerance
would pass while quietly permitting exactly the divergence this project exists
to detect.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.similarity.scoring import (
    InvertedIndex,
    ScoringAlgorithm,
    ScoringScratch,
    daat_scores,
    score,
    taat_scores,
)
from tfidf_stability.utils.numerics import Reduction, same_bits
from tfidf_stability.vectorisation.sparse import CsrMatrix, SparseVector
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

POLICIES = list(Reduction)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def corpus() -> list[list[str]]:
    """A corpus large enough for the comparison to be more than anecdotal.

    Deliberately includes empty documents, which embed to the zero vector and so
    exercise the zero-norm branch of both kernels on every single query.
    """
    rng = random.Random(20260811)
    alpha = [f"t{i}" for i in range(400)]
    return [[rng.choice(alpha) for _ in range(rng.randint(0, 40))] for _ in range(300)]


@pytest.fixture(scope="module")
def model(corpus: list[list[str]]) -> TfidfModel:
    return TfidfVectoriser().fit(corpus)


@pytest.fixture(scope="module")
def index(model: TfidfModel) -> InvertedIndex:
    return InvertedIndex.from_csr(model.matrix)


def queries(
    model: TfidfModel, corpus: Sequence[Sequence[str]], n: int, seed: int
) -> list[SparseVector]:
    """``n`` query vectors embedded through the model's own vocabulary and IDF."""
    rng = random.Random(seed)
    alpha = sorted({t for d in corpus for t in d})
    out = []
    for _ in range(n):
        features = [rng.choice(alpha) for _ in range(rng.randint(0, 25))]
        out.append(TfidfVectoriser.transform_query(features, model))
    return out


def sv(mapping: dict[int, float], dim: int) -> SparseVector:
    return SparseVector.from_mapping(mapping, dim)


# ---------------------------------------------------------------------------
# The inverted index
# ---------------------------------------------------------------------------
def test_transposition_is_canonical_and_complete(model: TfidfModel, index: InvertedIndex) -> None:
    """Postings must be ascending in document id, which the scoring loops assume.

    That comes for free from the counting sort visiting rows in ascending
    document order -- but the accumulation order, and hence every digit of every
    score, depends on it, so it is asserted rather than trusted.
    """
    assert index.is_canonical()
    assert index.nnz == model.matrix.nnz
    assert all(index.df(t) == model.vocabulary.df[t] for t in range(model.n_features))

    # Every (doc, term, weight) triple in the CSR appears once in the CSC, with
    # the identical bit pattern -- a transpose may not perturb a value.
    seen = 0
    for d in range(model.n_documents):
        row = model.matrix.row(d)
        for t, w in zip(row.indices, row.values, strict=True):
            postings = {
                index.rowidx[p]: index.values[p]
                for p in range(index.postings_begin(t), index.postings_end(t))
            }
            assert d in postings
            assert same_bits(postings[d], w)
            seen += 1
    assert seen == model.matrix.nnz


def test_single_term_postings_are_handled(mini_model: TfidfModel) -> None:
    """A term occurring in exactly one document: the shortest non-empty list.

    A postings list of length one is where an off-by-one in the column bounds
    would first show. The random fixture corpus has no such term -- its smallest
    df is 5 -- so the hand-written mini corpus is used instead, where 10 of the
    26 vocabulary entries occur in a single document.
    """
    idx = InvertedIndex.from_csr(mini_model.matrix)
    docs = list(mini_model.matrix.rows())
    singletons = [t for t in range(mini_model.n_features) if idx.df(t) == 1]
    assert len(singletons) == 10

    for t in singletons:
        d = idx.rowidx[idx.postings_begin(t)]
        query = sv({t: 1.0}, mini_model.n_features)
        got = taat_scores(query, idx, mini_model.norms)
        expected = cosine_against_corpus(query, docs, mini_model.norms)
        assert all(same_bits(a, b) for a, b in zip(got, expected, strict=True))
        # Exactly one document can score non-zero against a df = 1 query.
        assert [i for i, s in enumerate(got) if s != 0.0] == [d]


# ---------------------------------------------------------------------------
# The headline equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_taat_daat_and_cosine_agree_bit_for_bit(
    model: TfidfModel, index: InvertedIndex, corpus: list[list[str]], policy: Reduction
) -> None:
    """Three independent traversals, one bit pattern, every document."""
    docs = list(model.matrix.rows())
    norms = model.matrix.row_norms(policy)
    scratch = ScoringScratch()

    compared = 0
    for q in queries(model, corpus, 40, seed=555):
        reference = cosine_against_corpus(q, docs, norms, policy)
        taat = taat_scores(q, index, norms, policy, scratch=scratch)
        daat = daat_scores(q, model.matrix, norms, policy)
        for i, (r, t, d) in enumerate(zip(reference, taat, daat, strict=True)):
            assert same_bits(r, t), f"doc {i}: reference {r!r} != taat {t!r}"
            assert same_bits(r, d), f"doc {i}: reference {r!r} != daat {d!r}"
            compared += 1
    assert compared == 40 * model.n_documents


def test_the_comparison_is_not_vacuous(
    model: TfidfModel, index: InvertedIndex, corpus: list[list[str]]
) -> None:
    """Agreement on an all-zero score vector would prove nothing.

    So it is checked that the queries actually reach a substantial fraction of
    the corpus with non-zero scores. Ten queries reach 901 documents here; the
    threshold is set well below that so a change of seed does not turn a
    measurement into a failure.
    """
    reached = 0
    for q in queries(model, corpus, 10, seed=99):
        reached += sum(1 for s in taat_scores(q, index, model.norms) if s != 0.0)
    assert reached > 500


def test_scratch_reuse_is_numerically_invisible(
    model: TfidfModel, index: InvertedIndex, corpus: list[list[str]]
) -> None:
    """The dense accumulator survives between calls; a stale slot would leak.

    The touched-list reset is ``O(|touched|)``, so it clears only what the last
    query wrote. If that bookkeeping were wrong the contamination would be
    silent -- a plausible score, just the wrong one.
    """
    qs = queries(model, corpus, 6, seed=4242)
    fresh = [taat_scores(q, index, model.norms, scratch=ScoringScratch()) for q in qs]

    shared = ScoringScratch()
    for _ in range(3):  # several passes, so state has every chance to accumulate
        for q, expected in zip(qs, fresh, strict=True):
            got = taat_scores(q, index, model.norms, scratch=shared)
            assert all(same_bits(a, b) for a, b in zip(expected, got, strict=True))


def test_score_dispatches_to_both_kernels(
    model: TfidfModel, index: InvertedIndex, corpus: list[list[str]]
) -> None:
    q = queries(model, corpus, 1, seed=7)[0]
    taat = score(q, model.matrix, index, model.norms, algorithm=ScoringAlgorithm.TAAT)
    daat = score(q, model.matrix, index, model.norms, algorithm=ScoringAlgorithm.DAAT)
    assert all(same_bits(a, b) for a, b in zip(taat, daat, strict=True))
    assert str(ScoringAlgorithm.TAAT) == "taat"


def test_precomputed_query_norm_changes_nothing(
    model: TfidfModel, index: InvertedIndex, corpus: list[list[str]]
) -> None:
    from tfidf_stability.vectorisation.sparse import l2_norm

    q = queries(model, corpus, 1, seed=13)[0]
    a = taat_scores(q, index, model.norms)
    b = taat_scores(q, index, model.norms, query_norm=l2_norm(q))
    assert all(same_bits(x, y) for x, y in zip(a, b, strict=True))


def test_long_intersections_exercise_the_pairwise_tree() -> None:
    """Pairwise is the policy most sensitive to how the summation tree is built.

    Its base case is 128 elements, so an intersection of a few hundred terms is
    what distinguishes a streaming binary-counter merge from any other pairwise
    scheme. The generic corpus above never produces one; this does.
    """
    dim = 400
    row = sv({t: 1.0 + t * 1e-17 for t in range(dim)}, dim)
    other = sv(dict.fromkeys(range(0, dim, 3), 3.0), dim)
    matrix = CsrMatrix.from_rows([row, other], dim)
    index = InvertedIndex.from_csr(matrix)
    query = sv({t: 0.5 + t * 1e-16 for t in range(dim)}, dim)

    for policy in POLICIES:
        norms = matrix.row_norms(policy)
        reference = cosine_against_corpus(query, list(matrix.rows()), norms, policy)
        taat = taat_scores(query, index, norms, policy)
        daat = daat_scores(query, matrix, norms, policy)
        assert all(same_bits(a, b) for a, b in zip(reference, taat, strict=True)), policy
        assert all(same_bits(a, b) for a, b in zip(reference, daat, strict=True)), policy


# ---------------------------------------------------------------------------
# Degenerate inputs (spec_addenda G3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_empty_query_scores_zero_everywhere(
    model: TfidfModel, index: InvertedIndex, policy: Reduction
) -> None:
    """Degenerate but legitimate: no NaN may escape into a sort key."""
    empty = SparseVector.zero(model.n_features)
    for got in (
        taat_scores(empty, index, model.norms, policy),
        daat_scores(empty, model.matrix, model.norms, policy),
        cosine_against_corpus(empty, list(model.matrix.rows()), model.norms, policy),
    ):
        assert len(got) == model.n_documents
        # Bit patterns, not `== 0.0`: that comparison accepts a negative zero,
        # which would then serialise as "-0.0" and break the run digest.
        assert all(same_bits(s, 0.0) for s in got)


def test_query_with_no_matching_term_scores_zero() -> None:
    """A non-zero query whose support misses every document's support.

    Distinct from the empty query: the norm is positive, so the early return does
    not fire and the kernels must reach the end with an empty touched list.
    """
    # Every term of a fitted vocabulary occurs in some document, so the
    # disjointness is built explicitly in a two-document corpus instead.
    dim = 8
    matrix = CsrMatrix.from_rows([sv({0: 1.0, 1: 2.0}, dim), sv({1: 3.0}, dim)], dim)
    idx = InvertedIndex.from_csr(matrix)
    norms = matrix.row_norms()
    query = sv({5: 4.0, 7: 1.0}, dim)

    taat = taat_scores(query, idx, norms)
    daat = daat_scores(query, matrix, norms)
    reference = cosine_against_corpus(query, list(matrix.rows()), norms)
    assert taat == [0.0, 0.0]
    assert all(same_bits(a, b) for a, b in zip(reference, taat, strict=True))
    assert all(same_bits(a, b) for a, b in zip(reference, daat, strict=True))

    # The touched list must have stayed empty: every score comes from the initial
    # fill, not from an accumulator that happened to sum back to zero.
    scratch = ScoringScratch()
    taat_scores(query, idx, norms, scratch=scratch)
    assert scratch.touched == []


@pytest.mark.parametrize("policy", POLICIES)
def test_zero_norm_document_scores_zero_rather_than_dividing(policy: Reduction) -> None:
    """``cos := 0`` for a zero-norm document (section 2.3), not ``0/0``.

    A zero-norm document is common in short-text corpora and stays *rankable* at
    score 0, so it must produce a real zero rather than a NaN -- a NaN in a sort
    key is undefined behaviour, and these documents form the large exact-tie
    block the section 4.5 tie-break analysis is about.
    """
    dim = 6
    docs = [sv({0: 1.0, 2: 2.0}, dim), SparseVector.zero(dim), sv({0: 3.0}, dim)]
    matrix = CsrMatrix.from_rows(docs, dim)
    idx = InvertedIndex.from_csr(matrix)
    norms = matrix.row_norms(policy)
    assert norms[1] == 0.0

    query = sv({0: 1.0, 2: 1.0}, dim)
    taat = taat_scores(query, idx, norms, policy)
    daat = daat_scores(query, matrix, norms, policy)
    reference = cosine_against_corpus(query, docs, norms, policy)
    assert same_bits(taat[1], 0.0)
    assert all(same_bits(a, b) for a, b in zip(reference, taat, strict=True))
    assert all(same_bits(a, b) for a, b in zip(reference, daat, strict=True))

    # A *stored* zero norm on a document that does have postings takes the same
    # branch -- the guard is on the norm, not on the structure.
    forced = list(norms)
    forced[0] = 0.0
    assert same_bits(taat_scores(query, idx, forced, policy)[0], 0.0)
    assert same_bits(daat_scores(query, matrix, forced, policy)[0], 0.0)


def test_empty_corpus_scores_nothing() -> None:
    matrix = CsrMatrix.from_rows([], 4)
    idx = InvertedIndex.from_csr(matrix)
    query = sv({0: 1.0}, 4)
    assert taat_scores(query, idx, []) == []
    assert daat_scores(query, matrix, []) == []


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_dimension_mismatch_is_rejected(model: TfidfModel, index: InvertedIndex) -> None:
    wrong = sv({0: 1.0}, model.n_features + 1)
    with pytest.raises(ValueError, match="dimension mismatch"):
        taat_scores(wrong, index, model.norms)
    with pytest.raises(ValueError, match="dimension mismatch"):
        daat_scores(wrong, model.matrix, model.norms)


def test_norm_count_mismatch_is_rejected(model: TfidfModel, index: InvertedIndex) -> None:
    query = sv({0: 1.0}, model.n_features)
    with pytest.raises(ValueError, match="norms were supplied"):
        taat_scores(query, index, model.norms[:-1])
    with pytest.raises(ValueError, match="norms were supplied"):
        daat_scores(query, model.matrix, model.norms[:-1])


# ---------------------------------------------------------------------------
# The hand-written corpus, whose degenerate cases are all real text
# ---------------------------------------------------------------------------
def test_mini_corpus_agrees_across_all_three_paths(mini_model: TfidfModel) -> None:
    """Real text, including an exact-duplicate pair and an all-stopword document."""
    idx = InvertedIndex.from_csr(mini_model.matrix)
    assert idx.is_canonical()
    assert mini_model.zero_norm_documents, "the mini corpus should embed a zero vector"

    docs = list(mini_model.matrix.rows())
    for policy in POLICIES:
        norms = mini_model.matrix.row_norms(policy)
        for d in range(mini_model.n_documents):
            q = docs[d]  # each document as its own query: the self-similarity case
            reference = cosine_against_corpus(q, docs, norms, policy)
            taat = taat_scores(q, idx, norms, policy)
            daat = daat_scores(q, mini_model.matrix, norms, policy)
            assert all(same_bits(a, b) for a, b in zip(reference, taat, strict=True))
            assert all(same_bits(a, b) for a, b in zip(reference, daat, strict=True))
