// Sparse structures and query scoring.
//
// The centrepiece is `taat == daat` bit-for-bit. The two share no data
// structure and no loop nesting: TAAT walks postings lists out of an inverted
// index into a dense accumulator, DAAT merges each document's row against the
// query independently. Identical binary64 from both says more than either one
// matching a recorded expectation.
#include <tfidf/core/reduction.hpp>
#include <tfidf/similarity/scoring.hpp>
#include <tfidf/vectorisation/sparse.hpp>

#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
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


/// The first row with any terms in it. A random corpus contains empty rows, and
/// an empty query takes a shortcut that skips the code most of these tests are
/// about.
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
    // The normative Python rejects this: `_check_csr` in
    // persistence/save_load.py raises "indptr decreases at row {i}". This
    // mirror did not, and the two disagreed on exactly these arrays.
    //
    // `front == 0` and `back == nnz` both hold, so the arms `is_canonical()`
    // already had could not see it. `row(1)` then computes `hi - lo` as
    // `2 - 3` on `std::size_t`, which wraps to `std::dynamic_extent`, and
    // `subspan` reads that as "to the end" -- so row 1 silently spanned the
    // rest of the arrays and overlapped row 2 instead of being refused.
    const std::vector<Offset> backwards{0, 3, 2, 4};
    const std::vector<TermId> indices{0, 1, 2, 3};
    const std::vector<Real> values{1.0, 2.0, 3.0, 4.0};

    const CsrView csr{backwards, indices, values, 3, 4};
    CHECK(csr.indptr.front() == 0);              // the arms that did exist
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

    // A zero query scores 0 against everything and must not produce NaN.
    //
    // Poisoned first, and compared on bits. `std::vector<Real> out(n)` value-
    // initialises to 0.0, so the previous form also passed when score_taat wrote
    // nothing at all: it could not tell "wrote zeros" from "never ran". And
    // `s == 0.0` is true of -0.0, which this repository treats as a distinct
    // value everywhere else (ranking/margins.py reasons that -0.0 cannot occur,
    // and every score comparison elsewhere is bitwise).
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
    // The case above uses index 99 against dim 8, which is out of range by so
    // much that relaxing the bound to `> dim` still rejects it; and it puts its
    // first offending pair at position 2, so restricting the ascent check to
    // `i > 1` still rejects that too. Both relaxations are real defects -- one
    // admits an index one past the end, the other stops reading the first pair
    // -- so each bound is stated here at the value that separates it.
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
    // The envelope is four separate conditions and a malformed matrix usually
    // breaks one of them. Written as one test with everything wrong at once,
    // three of the four could be deleted and it would still pass -- and the
    // front/back pair is a single `||`, so an input that violates both cannot
    // tell it from `&&`. One violation per subcase is what distinguishes them.
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
    // The member initialisers. Every test builds these by aggregate
    // initialisation, which overwrites all of them, so nothing else states what
    // a default costs: a `dim` of 1 makes an empty vector claim a coordinate it
    // has no value for, and an `n_rows` of 1 makes an empty matrix claim a row
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
    // The guard is `query_norm == 0.0 || query.empty()`, and every existing
    // case that reaches it is empty, so the norm arm is never the one that
    // fires. A query with indices and all-zero values is not empty and has a
    // zero norm: without the norm arm it reaches the divide, where the
    // accumulated 0.0 over a query norm of 0.0 is NaN, and section 2.3's
    // convention says the similarity is zero.
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
    // `out` is sized by the caller to exactly n_rows, so a loop bound one too
    // large writes into whatever follows it. The corruption is silent: the
    // scores themselves stay correct, and only a build with a sanitiser would
    // otherwise notice. A sentinel immediately after the span makes it visible
    // in an ordinary build.
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
    // TAAT and DAAT must return identical bits, which is what makes the pair a
    // check on each other -- and also what makes the dispatch invisible to
    // every test that only reads `out`. The two differ in what they touch:
    // TAAT walks the inverted index through the scratch, DAAT merges each row
    // and never uses it. So the scratch is the evidence of which one ran.
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
    // The comment at the accumulator says a double push "would only rewrite the
    // same quotient" and is "asserted in the test suite". It was not. The
    // compensated path tracks this with a separate `seen` array, and a document
    // carrying several query terms is pushed once per posting without it.
    //
    // The scores stay right either way, so only the list itself shows it, and
    // `clear_touched` walks that list on every subsequent query.
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
