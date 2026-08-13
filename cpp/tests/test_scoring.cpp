// Sparse structures and query scoring.
//
// The centrepiece is `taat == daat` bit-for-bit. Those two algorithms share no
// data structure and no loop nesting: TAAT walks postings lists out of an
// inverted index into a dense accumulator, DAAT merges each document's row
// against the query independently. Getting identical binary64 output from both
// is a far stronger statement than either one matching a recorded expectation.
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
    // Free, because the counting sort visits rows in ascending document order.
    // Relied upon by the scoring loops, so asserted rather than assumed.
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
    // The strongest correctness signal in the native suite: two structurally
    // unrelated traversals producing identical bits.
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
    // The touched-list reset must fully clear state; a stale accumulator would
    // silently contaminate the next query.
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
    // initialises to 0.0, so the previous form passed unchanged if score_taat
    // wrote *nothing at all* -- it could not tell "correctly wrote zeros" from
    // "never ran". And `s == 0.0` is true of -0.0, which this repository treats
    // as a distinct value everywhere else (ranking/margins.py reasons that -0.0
    // cannot occur, and every score comparison elsewhere is bitwise).
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
