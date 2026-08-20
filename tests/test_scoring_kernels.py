"""The two Python scoring kernels, and their agreement with the cosine reference.

``TAAT == DAAT == cosine_against_corpus`` on raw bit patterns. The three share
no loop nesting and no data structure: TAAT walks postings lists out of an
inverted index into a dense accumulator, DAAT merges each row against the query
independently, and ``cosine_against_corpus`` merges against a materialised
sequence of document vectors. Identical binary64 from all three leaves an
indexing or accumulation bug nowhere to hide, and says more than any of them
matching a recorded expectation.

No tolerances. ``pytest.approx`` compares numbers that are close; the property
under test is bit identity, and a tolerance would pass while permitting the
divergence this project exists to detect.
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

    Includes empty documents, which embed to the zero vector and so exercise the
    zero-norm branch of both kernels on every query.
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
    """Postings must ascend in document id, which the scoring loops assume.

    It follows from the counting sort visiting rows in ascending document order.
    Accumulation order, and so every digit of every score, depends on it, hence
    the assertion rather than trust.
    """
    assert index.is_canonical()
    assert index.nnz == model.matrix.nnz
    assert all(index.df(t) == model.vocabulary.df[t] for t in range(model.n_features))

    # Every (doc, term, weight) triple in the CSR appears once in the CSC with
    # the same bit pattern: a transpose may not perturb a value.
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
    """A term occurring in one document: the shortest non-empty postings list.

    Length one is where an off-by-one in the column bounds shows first. The
    random fixture corpus has no such term (its smallest df is 5), so the
    hand-written mini corpus is used, where 10 of its 26 vocabulary entries
    occur in a single document.
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

    Ten queries reach 901 documents with non-zero scores here; the threshold
    sits well below that so a change of seed does not turn a measurement into a
    failure.
    """
    reached = 0
    for q in queries(model, corpus, 10, seed=99):
        reached += sum(1 for s in taat_scores(q, index, model.norms) if s != 0.0)
    assert reached > 500


def test_scratch_reuse_is_numerically_invisible(
    model: TfidfModel, index: InvertedIndex, corpus: list[list[str]]
) -> None:
    """The dense accumulator survives between calls; a stale slot would leak.

    The touched-list reset is ``O(|touched|)`` and clears only what the last
    query wrote. Wrong bookkeeping there yields a plausible score that is the
    wrong one.
    """
    qs = queries(model, corpus, 6, seed=4242)
    # The comparison below runs one iteration per query, so an empty grid would
    # make the whole test vacuous.
    assert len(qs) == 6, "the query grid is the test's only source of coverage"
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
    """Pairwise is the policy most sensitive to the shape of the summation tree.

    Its base case is 128 elements, so only an intersection of a few hundred
    terms separates a streaming binary-counter merge from any other pairwise
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
        # Bit patterns rather than `== 0.0`, which accepts a negative zero that
        # would serialise as "-0.0" and break the run digest.
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

    # The touched list must have stayed empty, so every score comes from the
    # initial fill rather than an accumulator that summed back to zero.
    scratch = ScoringScratch()
    taat_scores(query, idx, norms, scratch=scratch)
    assert scratch.touched == []


@pytest.mark.parametrize("policy", POLICIES)
def test_zero_norm_document_scores_zero_rather_than_dividing(policy: Reduction) -> None:
    """``cos := 0`` for a zero-norm document (section 2.3) in place of ``0/0``.

    Zero-norm documents are common in short-text corpora and stay rankable at
    score 0, so the result must be a real zero: a NaN in a sort key is undefined
    behaviour, and these documents form the large exact-tie block the section
    4.5 tie-break analysis is about.
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

    # A stored zero norm on a document that does have postings takes the same
    # branch: the guard reads the norm rather than the structure.
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


# ---------------------------------------------------------------------------
# What `is_canonical` is for
# ---------------------------------------------------------------------------
# The scoring loops read postings lists without re-establishing their invariants,
# on the strength of this check. It is only worth that if it actually rejects,
# so each rejection arm gets a specimen. The mirror of `save_load._check_csr`,
# one transpose along.
def _index(colptr, rowidx, values, n_rows=3, n_cols=2) -> InvertedIndex:
    return InvertedIndex(
        colptr=tuple(colptr),
        rowidx=tuple(rowidx),
        values=tuple(values),
        n_rows=n_rows,
        n_cols=n_cols,
    )


def test_a_canonical_index_passes_every_arm() -> None:
    """The reference the rejections below are rejections from."""
    assert _index([0, 2, 3], [0, 2, 1], [1.0, 2.0, 3.0]).is_canonical()


@pytest.mark.parametrize(
    ("case", "colptr", "rowidx", "values"),
    [
        ("colptr is the wrong length", [0, 2], [0, 2], [1.0, 2.0]),
        ("colptr does not start at zero", [1, 2, 3], [0, 2, 1], [1.0, 2.0, 3.0]),
        ("colptr does not end at nnz", [0, 2, 9], [0, 2, 1], [1.0, 2.0, 3.0]),
        ("a posting has no weight", [0, 2, 3], [0, 2, 1], [1.0, 2.0]),
    ],
)
def test_a_malformed_index_is_rejected_before_its_postings_are_read(
    case: str, colptr: list[int], rowidx: list[int], values: list[float]
) -> None:
    """These four are settled from the array lengths alone, so they are checked
    once rather than per column: a colptr that disagrees with its own arrays
    makes every subsequent postings slice meaningless."""
    assert not _index(colptr, rowidx, values).is_canonical(), case


def test_a_postings_list_that_repeats_a_document_is_rejected() -> None:
    """Strict increase makes a postings list a set. A repeat double-counts one
    document's contribution to that term in every score TAAT computes."""
    assert not _index([0, 2, 3], [1, 1, 2], [1.0, 2.0, 3.0]).is_canonical()


def test_a_postings_list_in_descending_order_is_rejected() -> None:
    """Ascending order is what makes the DAAT merge and the accumulator agree;
    the same documents in the other order is not the same index."""
    assert not _index([0, 2, 3], [2, 0, 1], [1.0, 2.0, 3.0]).is_canonical()


def test_a_posting_naming_a_document_outside_the_corpus_is_rejected() -> None:
    """It would index past the accumulator, and a negative one would silently
    wrap round to a real document at the other end."""
    assert not _index([0, 2, 3], [0, 7, 1], [1.0, 2.0, 3.0]).is_canonical()
    assert not _index([0, 2, 3], [-1, 2, 1], [1.0, 2.0, 3.0]).is_canonical()


def test_a_term_with_no_postings_leaves_the_index_canonical() -> None:
    """An empty column is `lo == hi`, not an error. The check rejects a colptr
    that *decreases*; treating equality as a decrease would condemn any index
    holding a term that survived the vocabulary but appears in no document."""
    empty_middle = _index([0, 1, 1, 2], [0, 2], [1.0, 2.0], n_rows=3, n_cols=3)
    assert empty_middle.is_canonical()
    assert empty_middle.df(1) == 0, "the middle term really has no postings"


def test_a_posting_one_past_the_last_document_is_out_of_range() -> None:
    """Documents are 0-based, so `n_rows` itself is the first invalid index and
    the one an off-by-one produces. It would read past the score accumulator."""
    assert not _index([0, 1, 2], [0, 3], [1.0, 2.0], n_rows=3, n_cols=2).is_canonical()
    assert _index([0, 1, 2], [0, 2], [1.0, 2.0], n_rows=3, n_cols=2).is_canonical()


def test_a_query_of_explicit_zeros_scores_zero_everywhere(
    model: TfidfModel, index: InvertedIndex
) -> None:
    """A vector can carry stored entries whose values are all 0.0 -- a query
    built from tokens that are all in the vocabulary but weightless. Its norm is
    zero while its nnz is not, so the degenerate check has to catch either
    condition. Requiring both divides by zero (spec_addenda G3).
    """
    zero_valued = SparseVector(indices=(0, 2), values=(0.0, 0.0), dim=len(model.vocabulary))
    assert zero_valued.nnz == 2, "the premise: stored entries, no magnitude"

    norms = model.matrix.row_norms()
    assert taat_scores(zero_valued, index, norms) == [0.0] * model.n_documents
    assert daat_scores(zero_valued, model.matrix, norms) == [0.0] * model.n_documents


def test_the_scratch_records_each_touched_document_once(
    model: TfidfModel, index: InvertedIndex, corpus: list[list[str]]
) -> None:
    """`touched` is the reset list: everything the accumulator wrote, cleared
    before the next query. A document appended once per matching term instead of
    once per query still produces the right scores, and grows without bound as
    the scratch is reused."""
    query = TfidfVectoriser.transform_query(corpus[0], model)

    # Every policy: the naive path accumulates into a dense array and the
    # compensated ones collect per-document addend lists, and they keep their
    # own copies of this bookkeeping.
    for policy in POLICIES:
        scratch = ScoringScratch()
        taat_scores(query, index, model.matrix.row_norms(policy), policy, scratch=scratch)
        assert scratch.touched, f"the query matched something under {policy}"
        assert len(scratch.touched) == len(set(scratch.touched)), (
            f"a document was recorded more than once under {policy}"
        )
