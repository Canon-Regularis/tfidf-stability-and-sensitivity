// Sparse structures and query scoring.
//
// TAAT (postings into a dense accumulator) and DAAT (per-row merge) share no
// data structure or loop nesting, so bitwise agreement cross-checks both.
// Rationale: docs/engineering_notes.md#test-scoring-taat-daat-cross-check
#include <tfidf/core/reduction.hpp>
#include <tfidf/similarity/scoring.hpp>
#include <tfidf/vectorisation/sparse.hpp>

#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace tfidf;

namespace {

bool same_bits(Real a, Real b) {
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

/// A small corpus held in owning vectors, with views onto it.
struct Corpus {
    std::vector<Offset> indptr;
    std::vector<TermId> indices;
    std::vector<Real> values;
    DocId n_rows = 0;
    TermId n_cols = 0;

    [[nodiscard]] CsrView view() const {
        return CsrView{indptr, indices, values, n_rows, n_cols};
    }
};


/// Return the first row with at least two terms, or row 0 if there is none.
///
/// A random corpus contains empty rows, and an empty query takes a shortcut
/// that bypasses the scoring loops under test. Two terms let one document
/// be reached by more than one posting of the query.
SparseView first_non_empty_row(const CsrView& csr) {
    for (DocId d = 0; d < csr.n_rows; ++d) {
        if (csr.row(d).nnz() >= 2) {
            return csr.row(d);
        }
    }
    return csr.row(0);
}

/// Build a random sparse corpus with strictly ascending indices per row.
Corpus random_corpus(DocId n_docs, TermId n_terms, std::size_t max_nnz, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> len(0, static_cast<int>(max_nnz));
    std::uniform_int_distribution<int> term(0, n_terms - 1);
    std::uniform_real_distribution<Real> val(0.01, 5.0);

    Corpus c;
    c.n_rows = n_docs;
    c.n_cols = n_terms;
    c.indptr.push_back(0);
    for (DocId d = 0; d < n_docs; ++d) {
        std::vector<TermId> ts;
        const int k = len(rng);
        for (int i = 0; i < k; ++i) {
            ts.push_back(term(rng));
        }
        std::sort(ts.begin(), ts.end());
        ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
        for (const TermId t : ts) {
            c.indices.push_back(t);
            c.values.push_back(val(rng));
        }
        c.indptr.push_back(static_cast<Offset>(c.values.size()));
    }
    return c;
}

}  // namespace

// -----------------------------------------------------------------------------
// Sparse primitives
// -----------------------------------------------------------------------------
TEST_CASE("sparse: dot ignores non-overlapping support") {
    const std::vector<TermId> ai{0, 1, 5};
    const std::vector<Real> av{1.0, 2.0, 3.0};
    const std::vector<TermId> bi{1, 2, 5};
    const std::vector<Real> bv{4.0, 9.0, 1.0};
    const SparseView a{ai, av, 8};
    const SparseView b{bi, bv, 8};
    CHECK(dot(a, b, Reduction::Naive) == 2.0 * 4.0 + 3.0 * 1.0);
}

TEST_CASE("sparse: norms and the zero vector") {
    const std::vector<TermId> i{0, 1};
    const std::vector<Real> v{3.0, 4.0};
    CHECK(l2_norm(SparseView{i, v, 4}, Reduction::Naive) == 5.0);
    CHECK(l2_norm(SparseView{{}, {}, 4}, Reduction::Naive) == 0.0);
}

TEST_CASE("sparse: canonical-form detection") {
    const std::vector<TermId> good{0, 2, 5};
    const std::vector<TermId> unsorted{0, 5, 2};
    const std::vector<TermId> dup{0, 2, 2};
    const std::vector<TermId> oob{0, 2, 99};
    const std::vector<Real> v{1.0, 1.0, 1.0};
    CHECK(SparseView{good, v, 8}.is_canonical());
    CHECK_FALSE(SparseView{unsorted, v, 8}.is_canonical());
    CHECK_FALSE(SparseView{dup, v, 8}.is_canonical());
    CHECK_FALSE(SparseView{oob, v, 8}.is_canonical());
}

TEST_CASE("sparse: a decreasing indptr is not canonical") {
    // Like `_check_csr` in persistence/save_load.py, `is_canonical()` rejects a decreasing indptr.
    // `front == 0` and `back == nnz` both hold, so only the monotonicity arm rejects
    // this input; without it, `row(1)` wraps `2 - 3` to `std::dynamic_extent`.
    // Rationale: docs/engineering_notes.md#test-scoring-csr-indptr-monotonicity
    const std::vector<Offset> backwards{0, 3, 2, 4};
    const std::vector<TermId> indices{0, 1, 2, 3};
    const std::vector<Real> values{1.0, 2.0, 3.0, 4.0};

    const CsrView csr{backwards, indices, values, 3, 4};
    CHECK(csr.indptr.front() == 0);              // the endpoint arms
    CHECK(csr.indptr.back() == csr.nnz());       // both pass on this input
    CHECK_FALSE(csr.is_canonical());
}

TEST_CASE("sparse: a non-decreasing indptr with an empty row is canonical") {
    // The guard's other half. An empty row makes two adjacent offsets equal,
    // which is ordinary -- a document with no in-vocabulary terms -- so a
    // monotonicity check written as strictly increasing would reject the zero
    // norm documents this project is largely about.
    const std::vector<Offset> with_empty_row{0, 2, 2, 4};
    const std::vector<TermId> indices{0, 1, 2, 3};
    const std::vector<Real> values{1.0, 2.0, 3.0, 4.0};

    const CsrView csr{with_empty_row, indices, values, 3, 4};
    CHECK(csr.is_canonical());
    CHECK(csr.row(1).empty());
}

TEST_CASE("sparse: transpose is a faithful inverted index") {
    const Corpus c = random_corpus(40, 25, 8, 99);
    const CsrView csr = c.view();
    REQUIRE(csr.is_canonical());

    const Csc csc = transpose(csr);
    CHECK(csc.rowidx.size() == static_cast<std::size_t>(csr.nnz()));

    // Every (doc, term, value) in the CSR must appear once in the CSC.
    std::size_t matched = 0;
    for (DocId d = 0; d < csr.n_rows; ++d) {
        const SparseView row = csr.row(d);
        for (std::size_t k = 0; k < row.nnz(); ++k) {
            const TermId t = row.indices[k];
            bool found = false;
            for (std::size_t p = csc.postings_begin(t); p < csc.postings_end(t); ++p) {
                if (csc.rowidx[p] == d) {
                    CHECK(same_bits(csc.values[p], row.values[k]));
                    found = true;
                    break;
                }
            }
            CHECK(found);
            ++matched;
        }
    }
    CHECK(matched == static_cast<std::size_t>(csr.nnz()));
}

TEST_CASE("sparse: postings lists come out ascending in document id") {
    // Free, since the counting sort visits rows in ascending document order.
    // The scoring loops rely on it, so it is asserted rather than assumed.
    const Corpus c = random_corpus(60, 20, 6, 4242);
    const Csc csc = transpose(c.view());
    for (TermId t = 0; t < csc.n_cols; ++t) {
        for (std::size_t p = csc.postings_begin(t) + 1; p < csc.postings_end(t); ++p) {
            CHECK(csc.rowidx[p - 1] < csc.rowidx[p]);
        }
    }
}

// -----------------------------------------------------------------------------
// Scoring
// -----------------------------------------------------------------------------
TEST_CASE("scoring: TAAT and DAAT agree bit for bit") {
    // Two structurally unrelated traversals producing identical bits.
    for (const std::uint64_t seed : {1u, 2u, 3u, 17u, 20260811u}) {
        const Corpus c = random_corpus(120, 45, 12, seed);
        const CsrView csr = c.view();
        REQUIRE(csr.is_canonical());
        const Csc csc = transpose(csr);
        const std::vector<Real> norms = row_norms(csr, Reduction::Naive);

        std::mt19937_64 rng(seed * 31 + 7);
        std::uniform_int_distribution<int> qterm(0, csr.n_cols - 1);
        std::uniform_real_distribution<Real> qval(0.1, 3.0);

        for (int q = 0; q < 20; ++q) {
            std::vector<TermId> qi;
            for (int i = 0; i < 10; ++i) {
                qi.push_back(qterm(rng));
            }
            std::sort(qi.begin(), qi.end());
            qi.erase(std::unique(qi.begin(), qi.end()), qi.end());
            std::vector<Real> qv;
            qv.reserve(qi.size());
            for (std::size_t i = 0; i < qi.size(); ++i) {
                qv.push_back(qval(rng));
            }
            const SparseView query{qi, qv, csr.n_cols};
            const Real qn = l2_norm(query, Reduction::Naive);

            std::vector<Real> a(static_cast<std::size_t>(csr.n_rows));
            std::vector<Real> b(static_cast<std::size_t>(csr.n_rows));
            ScoringScratch scratch;
            score_taat(query, csc, norms, qn, a, scratch, Reduction::Naive);
            score_daat(query, csr, norms, qn, b, Reduction::Naive);

            for (std::size_t i = 0; i < a.size(); ++i) {
                CHECK(same_bits(a[i], b[i]));
            }
        }
    }
}

TEST_CASE("scoring: reusing scratch across queries changes nothing") {
    // The touched-list reset must clear every slot; a stale accumulator would
    // contaminate the next query.
    const Corpus c = random_corpus(80, 30, 10, 555);
    const CsrView csr = c.view();
    const Csc csc = transpose(csr);
    const std::vector<Real> norms = row_norms(csr, Reduction::Naive);

    const std::vector<TermId> q1i{1, 4, 9};
    const std::vector<Real> q1v{1.0, 2.0, 0.5};
    const std::vector<TermId> q2i{0, 4, 20};
    const std::vector<Real> q2v{3.0, 1.0, 2.0};
    const SparseView q1{q1i, q1v, csr.n_cols};
    const SparseView q2{q2i, q2v, csr.n_cols};

    std::vector<Real> fresh(static_cast<std::size_t>(csr.n_rows));
    std::vector<Real> reused(static_cast<std::size_t>(csr.n_rows));
    ScoringScratch s1;
    ScoringScratch s2;

    score_taat(q2, csc, norms, l2_norm(q2, Reduction::Naive), fresh, s1, Reduction::Naive);
    // Same scratch, second query after a first.
    std::vector<Real> tmp(static_cast<std::size_t>(csr.n_rows));
    score_taat(q1, csc, norms, l2_norm(q1, Reduction::Naive), tmp, s2, Reduction::Naive);
    score_taat(q2, csc, norms, l2_norm(q2, Reduction::Naive), reused, s2, Reduction::Naive);

    for (std::size_t i = 0; i < fresh.size(); ++i) {
        CHECK(same_bits(fresh[i], reused[i]));
    }
}

TEST_CASE("scoring: the zero-vector convention of section 2.3") {
    const Corpus c = random_corpus(20, 10, 5, 3);
    const CsrView csr = c.view();
    const Csc csc = transpose(csr);
    std::vector<Real> norms = row_norms(csr, Reduction::Naive);
    std::vector<Real> out(static_cast<std::size_t>(csr.n_rows));
    ScoringScratch scratch;

    // A zero query scores +0.0 against everything, never NaN or -0.0.
    // `out` is filled with NaN first so a kernel that writes nothing fails, and
    // the sign is checked because `s == 0.0` also holds for -0.0.
    // Rationale: docs/engineering_notes.md#test-scoring-zero-query-poisoned-output
    std::fill(out.begin(), out.end(), std::numeric_limits<Real>::quiet_NaN());
    const SparseView zero{{}, {}, csr.n_cols};
    score_taat(zero, csc, norms, 0.0, out, scratch, Reduction::Naive);
    for (const Real s : out) {
        CHECK_FALSE(std::isnan(s));  // fails now if the kernel wrote nothing
        CHECK(s == 0.0);
        CHECK(std::signbit(s) == false);  // +0.0, never -0.0
    }

    // A zero-norm document scores 0 rather than dividing by zero.
    norms[0] = 0.0;
    const std::vector<TermId> qi{0, 1};
    const std::vector<Real> qv{1.0, 1.0};
    const SparseView q{qi, qv, csr.n_cols};
    score_taat(q, csc, norms, l2_norm(q, Reduction::Naive), out, scratch, Reduction::Naive);
    CHECK(out[0] == 0.0);
    CHECK_FALSE(std::isnan(out[0]));
}

TEST_CASE("scoring: self-similarity is one to within a few ulp") {
    const Corpus c = random_corpus(30, 15, 8, 77);
    const CsrView csr = c.view();
    const Csc csc = transpose(csr);
    const std::vector<Real> norms = row_norms(csr, Reduction::Naive);
    std::vector<Real> out(static_cast<std::size_t>(csr.n_rows));
    ScoringScratch scratch;

    for (DocId d = 0; d < csr.n_rows; ++d) {
        if (norms[static_cast<std::size_t>(d)] == 0.0) {
            continue;
        }
        const SparseView self = csr.row(d);
        score_taat(self, csc, norms, norms[static_cast<std::size_t>(d)], out, scratch,
                   Reduction::Naive);
        CHECK(out[static_cast<std::size_t>(d)] == doctest::Approx(1.0).epsilon(1e-12));
    }
}

TEST_CASE("scoring: every score of non-negative data lies in [0, 1]") {
    const Corpus c = random_corpus(150, 40, 14, 8080);
    const CsrView csr = c.view();
    const Csc csc = transpose(csr);
    const std::vector<Real> norms = row_norms(csr, Reduction::Naive);
    std::vector<Real> out(static_cast<std::size_t>(csr.n_rows));
    ScoringScratch scratch;

    const std::vector<TermId> qi{2, 5, 11, 30};
    const std::vector<Real> qv{1.0, 0.4, 2.2, 0.7};
    const SparseView q{qi, qv, csr.n_cols};
    score_taat(q, csc, norms, l2_norm(q, Reduction::Naive), out, scratch, Reduction::Naive);
    for (const Real s : out) {
        CHECK(s >= 0.0);
        CHECK(s <= 1.0 + 1e-12);
    }
}

TEST_CASE("scoring: TAAT agrees with DAAT under every reduction policy") {
    const Corpus c = random_corpus(70, 28, 10, 2024);
    const CsrView csr = c.view();
    const Csc csc = transpose(csr);

    const std::vector<TermId> qi{1, 3, 7, 12, 19};
    const std::vector<Real> qv{1.5, 0.2, 3.0, 0.8, 1.1};
    const SparseView q{qi, qv, csr.n_cols};

    for (const auto p : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                         Reduction::Exact}) {
        const std::vector<Real> norms = row_norms(csr, p);
        const Real qn = l2_norm(q, p);
        std::vector<Real> a(static_cast<std::size_t>(csr.n_rows));
        std::vector<Real> b(static_cast<std::size_t>(csr.n_rows));
        ScoringScratch scratch;
        score_taat(q, csc, norms, qn, a, scratch, p);
        score_daat(q, csr, norms, qn, b, p);
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(same_bits(a[i], b[i]));
        }
    }
}

TEST_CASE("sparse: the vector guards are pinned at their boundaries, not past them") {
    // Each bound is tested at the value that separates it from a relaxed bound:
    // an index equal to dim, and a bad first pair. The canonical-form case above
    // (index 99, bad pair at position 2) is still rejected under `> dim` or `i > 1`.
    // Rationale: docs/engineering_notes.md#test-scoring-sparse-guard-boundaries
    const std::vector<Real> v{1.0, 1.0};

    SUBCASE("an index equal to dim is out of range") {
        const std::vector<TermId> at_dim{0, 8};
        const std::vector<TermId> below_dim{0, 7};
        CHECK_FALSE(SparseView{at_dim, v, 8}.is_canonical());
        CHECK(SparseView{below_dim, v, 8}.is_canonical());
    }

    SUBCASE("the first adjacent pair is checked like every other") {
        const std::vector<TermId> descending{5, 2};
        const std::vector<TermId> equal{5, 5};
        CHECK_FALSE(SparseView{descending, v, 8}.is_canonical());
        CHECK_FALSE(SparseView{equal, v, 8}.is_canonical());
    }

    SUBCASE("indices and values must be the same length") {
        // `dot` and `l2_norm` index `values` with a position taken from
        // `indices`, so a short `values` is a read past the end rather than a
        // smaller vector.
        const std::vector<TermId> two{0, 1};
        const std::vector<Real> one{1.0};
        CHECK_FALSE(SparseView{two, one, 8}.is_canonical());
    }

    SUBCASE("a negative index is out of range at the other end") {
        const std::vector<TermId> negative{-1, 2};
        CHECK_FALSE(SparseView{negative, v, 8}.is_canonical());
    }
}

TEST_CASE("sparse: each CSR envelope guard rejects on its own") {
    // Each subcase breaks one envelope condition; an input that breaks front and
    // back at once cannot distinguish their `||` from `&&`. A short `values` also
    // moves `nnz()`, so the length subcase trips the back arm too; the next case isolates it.
    // Rationale: docs/engineering_notes.md#test-scoring-csr-envelope-guards
    const std::vector<TermId> indices{0, 1, 2, 3};
    const std::vector<Real> values{1.0, 2.0, 3.0, 4.0};

    const std::vector<Offset> well_formed{0, 2, 4};

    SUBCASE("a well formed matrix is canonical, so the subcases below differ by one thing") {
        const CsrView csr{well_formed, indices, values, 2, 4};
        CHECK(csr.is_canonical());
    }

    SUBCASE("indptr must have one more entry than there are rows") {
        const CsrView csr{well_formed, indices, values, 3, 4};
        CHECK_FALSE(csr.is_canonical());
    }

    SUBCASE("indptr must start at zero, whatever its last entry is") {
        // `back() == nnz()` still holds, so only the front arm can refuse it.
        const std::vector<Offset> late_start{1, 2, 4};
        REQUIRE(late_start.back() == static_cast<Offset>(values.size()));
        const CsrView csr{late_start, indices, values, 2, 4};
        CHECK_FALSE(csr.is_canonical());
    }

    SUBCASE("indptr must end at nnz, whatever its first entry is") {
        // And the mirror: `front() == 0` holds, so only the back arm can.
        const std::vector<Offset> short_end{0, 2, 3};
        REQUIRE(short_end.front() == 0);
        const CsrView csr{short_end, indices, values, 2, 4};
        CHECK_FALSE(csr.is_canonical());
    }

    SUBCASE("indices and values must be the same length") {
        const std::vector<Real> three{1.0, 2.0, 3.0};
        const CsrView csr{well_formed, indices, three, 2, 4};
        CHECK_FALSE(csr.is_canonical());
    }
}

TEST_CASE("sparse: the CSR guards past the envelope reject on their own too") {
    // Continues the subcases above, for the three arms an envelope violation
    // reaches first. Each input satisfies every earlier arm, so only the one
    // named can refuse it -- otherwise the guard could be deleted and the test
    // would still pass on an earlier arm's verdict.
    const std::vector<TermId> four{0, 1, 2, 3};
    const std::vector<Real> values{1.0, 2.0, 3.0, 4.0};

    SUBCASE("indices longer than values, with the envelope still intact") {
        // `nnz()` is measured on `values`, so an indptr ending at 3 keeps the
        // front and back arms happy and leaves the length arm to catch it.
        const std::vector<Offset> ends_at_three{0, 2, 3};
        const std::vector<Real> three{1.0, 2.0, 3.0};
        const CsrView csr{ends_at_three, four, three, 2, 4};
        REQUIRE(csr.indptr.front() == 0);
        REQUIRE(csr.indptr.back() == csr.nnz());
        CHECK_FALSE(csr.is_canonical());
    }

    SUBCASE("indptr decreasing at the very first pair") {
        // The existing decreasing-indptr case puts its decrease at row 1, so a
        // monotonicity loop starting at row 1 still catches it. This one
        // decreases immediately, which only a loop starting at row 0 sees.
        const std::vector<Offset> drops_first{0, -1, 4};
        const CsrView csr{drops_first, four, values, 2, 4};
        REQUIRE(csr.indptr.front() == 0);
        REQUIRE(csr.indptr.back() == csr.nnz());
        CHECK_FALSE(csr.is_canonical());
    }

    SUBCASE("a non-ascending row zero") {
        // And the same boundary for the per-row loop: row 0 is the row a loop
        // starting at 1 never reads, so the descending pair is put there.
        const std::vector<Offset> indptr{0, 2, 4};
        const std::vector<TermId> row_zero_descends{5, 2, 6, 7};
        const CsrView csr{indptr, row_zero_descends, values, 2, 8};
        REQUIRE(csr.row(1).is_canonical());
        CHECK_FALSE(csr.is_canonical());
    }
}

TEST_CASE("sparse: df is a column length and row_norms starts at row zero") {
    // Two off-by-one sites that the corpus-scale tests cannot see: they compare
    // whole structures, so a wrong first row or a df that adds where it should
    // subtract is only visible when a single value is named.
    const std::vector<Offset> indptr{0, 2, 4};
    const std::vector<TermId> indices{0, 2, 1, 2};
    const std::vector<Real> values{3.0, 4.0, 6.0, 8.0};
    const CsrView csr{indptr, indices, values, 2, 3};
    REQUIRE(csr.is_canonical());

    // Built directly rather than by `transpose`, so this states what `df` reads
    // off a colptr and nothing about how the colptr was produced.
    const Csc index{{0, 1, 2, 4}, {0, 1, 0, 1}, {3.0, 6.0, 4.0, 8.0}, 2, 3};
    // Term 2 is in both documents, terms 0 and 1 in one each. Read as a sum
    // rather than a difference, term 1's df would be 3 in a corpus of 2.
    CHECK(index.df(0) == 1);
    CHECK(index.df(1) == 1);
    CHECK(index.df(2) == 2);

    const std::vector<Real> norms = row_norms(csr, Reduction::Naive);
    REQUIRE(norms.size() == 2);
    CHECK(norms[0] == 5.0);   // the row a loop starting at 1 would leave at 0.0
    CHECK(norms[1] == 10.0);
}

TEST_CASE("sparse: a default constructed view is empty rather than one wide") {
    // Pins the member initialisers, which aggregate initialisation elsewhere
    // always overwrites. A default `dim` of 1 gives an empty vector a coordinate
    // with no value; a default `n_rows` of 1 makes an empty matrix claim a row
    // whose `indptr` entries do not exist.
    const SparseView vector{};
    CHECK(vector.dim == 0);
    CHECK(vector.nnz() == 0);
    CHECK(vector.empty());

    const CsrView matrix{};
    CHECK(matrix.n_rows == 0);
    CHECK(matrix.n_cols == 0);
    CHECK(matrix.nnz() == 0);

    const Csc index{};
    CHECK(index.n_rows == 0);
    CHECK(index.n_cols == 0);
}

TEST_CASE("score: a query with terms but no weight scores zero rather than NaN") {
    // Indices with all-zero values make a query that is not empty but has zero
    // norm, so only the `query_norm == 0.0` arm of the guard stops 0.0 / 0.0 = NaN.
    // Section 2.3 defines the similarity as zero. Every other zero query here is empty.
    const Corpus c = random_corpus(30, 20, 6, 4242);
    const CsrView csr = c.view();
    const Csc index = transpose(csr);
    const std::vector<Real> norms = row_norms(csr, Reduction::Naive);

    const std::vector<TermId> terms{1, 4, 9};
    const std::vector<Real> no_weight{0.0, 0.0, 0.0};
    const SparseView query{terms, no_weight, csr.n_cols};
    const Real query_norm = l2_norm(query, Reduction::Naive);
    REQUIRE(query_norm == 0.0);
    REQUIRE_FALSE(query.empty());

    ScoringScratch scratch;
    scratch.reset(csr.n_rows);
    std::vector<Real> out(static_cast<std::size_t>(csr.n_rows), -1.0);

    for (const auto algorithm : {ScoringAlgorithm::Taat, ScoringAlgorithm::Daat}) {
        std::fill(out.begin(), out.end(), -1.0);
        scratch.reset(csr.n_rows);
        score(query, csr, index, norms, query_norm, out, scratch, Reduction::Naive, algorithm);
        for (const Real s : out) {
            CHECK_FALSE(std::isnan(s));
            CHECK(s == 0.0);
        }
    }
}

TEST_CASE("score: nothing is written past the end of the output span") {
    // `out` is exactly n_rows long, so a loop bound one too large corrupts the
    // memory after it while the scores stay correct. A sentinel after the span
    // makes the overrun visible without a sanitiser.
    const Corpus c = random_corpus(25, 15, 5, 606);
    const CsrView csr = c.view();
    const Csc index = transpose(csr);
    const std::vector<Real> norms = row_norms(csr, Reduction::Naive);
    const SparseView query = csr.row(2);
    const Real query_norm = l2_norm(query, Reduction::Naive);
    REQUIRE(query_norm > 0.0);

    const Real sentinel = -12345.5;
    ScoringScratch scratch;
    for (const auto algorithm : {ScoringAlgorithm::Taat, ScoringAlgorithm::Daat}) {
        std::vector<Real> buffer(static_cast<std::size_t>(csr.n_rows) + 1, sentinel);
        const std::span<Real> out{buffer.data(), static_cast<std::size_t>(csr.n_rows)};
        scratch.reset(csr.n_rows);
        score(query, csr, index, norms, query_norm, out, scratch, Reduction::Naive, algorithm);
        CHECK(buffer.back() == sentinel);
    }
}

TEST_CASE("score: the algorithm asked for is the algorithm that runs") {
    // TAAT and DAAT return identical bits, so `out` cannot show which one ran.
    // TAAT fills the scratch's touched list and DAAT never uses it, so the
    // scratch identifies the algorithm.
    const Corpus c = random_corpus(30, 12, 6, 31337);
    const CsrView csr = c.view();
    const Csc index = transpose(csr);
    const std::vector<Real> norms = row_norms(csr, Reduction::Naive);
    // A random corpus has empty rows -- documents with no in-vocabulary terms
    // are the case this project is largely about -- and an empty query takes
    // the zero-norm shortcut in both algorithms, which is the one path where
    // neither touches the scratch.
    const SparseView query = first_non_empty_row(csr);
    const Real query_norm = l2_norm(query, Reduction::Naive);
    REQUIRE(query_norm > 0.0);

    std::vector<Real> taat_out(static_cast<std::size_t>(csr.n_rows));
    std::vector<Real> daat_out(static_cast<std::size_t>(csr.n_rows));
    ScoringScratch scratch;

    scratch.reset(csr.n_rows);
    score(query, csr, index, norms, query_norm, taat_out, scratch, Reduction::Naive,
          ScoringAlgorithm::Taat);
    CHECK_FALSE(scratch.touched.empty());

    scratch.reset(csr.n_rows);
    score(query, csr, index, norms, query_norm, daat_out, scratch, Reduction::Naive,
          ScoringAlgorithm::Daat);
    CHECK(scratch.touched.empty());

    // And the reason the dispatch is otherwise invisible, restated: the two
    // agree bit for bit.
    for (std::size_t i = 0; i < taat_out.size(); ++i) {
        CHECK(same_bits(taat_out[i], daat_out[i]));
    }
}

TEST_CASE("score: a document enters the touched list once however many terms hit it") {
    // Each document enters the touched list once, however many query terms hit
    // it. The compensated policies track this with a separate `seen` array.
    // Duplicates leave the scores unchanged but lengthen the next `clear_touched`.
    const Corpus c = random_corpus(40, 10, 8, 8080);
    const CsrView csr = c.view();
    const Csc index = transpose(csr);

    for (const auto policy : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                              Reduction::Exact}) {
        const std::vector<Real> norms = row_norms(csr, policy);
        const SparseView query = first_non_empty_row(csr);
        const Real query_norm = l2_norm(query, policy);
        REQUIRE(query.nnz() >= 2);

        ScoringScratch scratch;
        scratch.reset(csr.n_rows);
        std::vector<Real> out(static_cast<std::size_t>(csr.n_rows));
        score(query, csr, index, norms, query_norm, out, scratch, policy,
              ScoringAlgorithm::Taat);

        std::vector<DocId> sorted = scratch.touched;
        std::sort(sorted.begin(), sorted.end());
        CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());

        // The corpus must actually exercise the case. Every posting of every
        // query term is one visit to a document, so if the visits outnumber the
        // documents touched, some document was reached more than once and a
        // list with no duplicates says something.
        std::size_t visits = 0;
        for (std::size_t k = 0; k < query.nnz(); ++k) {
            visits += index.df(query.indices[k]);
        }
        CHECK(visits > sorted.size());
    }
}
