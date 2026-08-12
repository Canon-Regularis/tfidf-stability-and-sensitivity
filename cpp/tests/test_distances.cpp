// Ordering distances: inversion counting, Kendall tau, FKS, and the top-k set
// measures.
//
// Two properties are worth more than the rest of this file put together, and
// both are guards against a plausible "improvement" rather than against a typo:
//
//   * the FKS triangle-inequality violation is asserted, with G2's witness and
//     its exact values. K^(p) is a near-metric; a future patch that "repairs"
//     it would silently move every published disagreement number.
//   * fks_max(1) == 1.0. Two disjoint singletons contribute one case-3 pair, so
//     the maximum is 1 -- an early `if k < 2: return 0` guard normalised two
//     entirely disjoint lists to distance zero.
#include <tfidf/ranking/distances.hpp>

#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tfidf;
using namespace tfidf::ranking;

namespace {

bool same_bits(Real a, Real b) {
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

std::int64_t brute_force_inversions(const std::vector<std::int32_t>& v) {
    std::int64_t total = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        for (std::size_t j = i + 1; j < v.size(); ++j) {
            total += v[i] > v[j] ? 1 : 0;
        }
    }
    return total;
}

/// `kendall_fks` is symmetric in its arguments; every call site here says so.
Real fks_raw(const std::vector<DocId>& a, const std::vector<DocId>& b, Real p = kFksPenalty) {
    const Real forward = kendall_fks(a, b, p, false);
    CHECK(same_bits(kendall_fks(b, a, p, false), forward));
    return forward;
}

}  // namespace

// -----------------------------------------------------------------------------
// Inversion counting
// -----------------------------------------------------------------------------
TEST_CASE("inversions: merge sort agrees with the quadratic definition") {
    std::mt19937_64 rng(20260811);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = static_cast<std::size_t>(rng() % 13);
        std::vector<std::int32_t> v(n);
        for (auto& x : v) {
            // A small alphabet, so equal elements -- the case the `<=` in the
            // merge decides -- occur constantly rather than never.
            x = static_cast<std::int32_t>(rng() % 5);
        }
        CHECK(inversion_count(v) == brute_force_inversions(v));
    }
}

TEST_CASE("inversions: extremes") {
    CHECK(inversion_count(std::vector<std::int32_t>{}) == 0);
    CHECK(inversion_count(std::vector<std::int32_t>{1, 2, 3, 4}) == 0);
    CHECK(inversion_count(std::vector<std::int32_t>{4, 3, 2, 1}) == 6);  // C(4, 2)
    CHECK(inversion_count(std::vector<std::int32_t>{1, 1, 1}) == 0);     // equals are not inversions
}

TEST_CASE("inversions: a reversed run of 1000 is C(1000, 2)") {
    // Exercises the recursion at a depth the k <= 50 top-k cases never reach;
    // tie groups are not bounded by k, which is why this path is O(n log n).
    std::vector<std::int32_t> v(1000);
    std::iota(v.rbegin(), v.rend(), 0);
    CHECK(inversion_count(v) == 1000LL * 999 / 2);
}

// -----------------------------------------------------------------------------
// Same-set Kendall tau
// -----------------------------------------------------------------------------
TEST_CASE("kendall tau: identical is 0, reversed is 1") {
    const std::vector<DocId> a{1, 2, 3, 4};
    std::vector<DocId> reversed(a.rbegin(), a.rend());
    CHECK(kendall_tau_distance(a, a) == 0.0);
    CHECK(kendall_tau_distance(a, reversed) == 1.0);
}

TEST_CASE("kendall tau: symmetric and normalised over all pairs of permutations") {
    std::vector<DocId> a{0, 1, 2, 3};
    do {
        std::vector<DocId> b{0, 1, 2, 3};
        do {
            const Real d = kendall_tau_distance(a, b);
            CHECK(d >= 0.0);
            CHECK(d <= 1.0);
            CHECK(same_bits(d, kendall_tau_distance(b, a)));
        } while (std::next_permutation(b.begin(), b.end()));
    } while (std::next_permutation(a.begin(), a.end()));
}

TEST_CASE("kendall tau: fewer than two elements is 0, not undefined") {
    CHECK(kendall_tau_distance(std::vector<DocId>{}, std::vector<DocId>{}) == 0.0);
    CHECK(kendall_tau_distance(std::vector<DocId>{7}, std::vector<DocId>{7}) == 0.0);
}

TEST_CASE("kendall tau: differing sets are refused, not approximated") {
    // The refusal is the point: it is the signal that kendall_fks is the
    // function actually wanted.
    const auto attempt = [](const std::vector<DocId>& a, const std::vector<DocId>& b) {
        static_cast<void>(kendall_tau_distance(a, b));  // discarding a [[nodiscard]] Real
    };
    CHECK_THROWS_AS(attempt({1, 2}, {1, 3}), std::invalid_argument);
    CHECK_THROWS_AS(attempt({1, 2}, {1}), std::invalid_argument);
}

TEST_CASE("kendall tau: unlike FKS, this one really is a metric") {
    std::vector<std::vector<DocId>> perms;
    std::vector<DocId> p{0, 1, 2, 3};
    do {
        perms.push_back(p);
    } while (std::next_permutation(p.begin(), p.end()));

    Real worst = 0.0;
    for (const auto& a : perms) {
        for (const auto& b : perms) {
            for (const auto& c : perms) {
                worst = std::max(worst, kendall_tau_distance(a, c) -
                                            (kendall_tau_distance(a, b) +
                                             kendall_tau_distance(b, c)));
            }
        }
    }
    // Not exactly zero: the distances are sixths, which binary64 cannot hold, so
    // the slack is rounding rather than structure.
    CHECK(worst <= 1e-12);
}

// -----------------------------------------------------------------------------
// FKS generalised Kendall
// -----------------------------------------------------------------------------
TEST_CASE("fks: identical and empty lists are 0") {
    CHECK(kendall_fks(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{1, 2, 3}) == 0.0);
    CHECK(kendall_fks(std::vector<DocId>{}, std::vector<DocId>{}) == 0.0);
}

TEST_CASE("fks: with equal sets only case 1 can fire, so it is plain Kendall") {
    std::vector<DocId> a{0, 1, 2, 3};
    do {
        std::vector<DocId> b{0, 1, 2, 3};
        do {
            std::vector<std::int32_t> mapped;
            for (const DocId x : a) {
                mapped.push_back(static_cast<std::int32_t>(
                    std::find(b.begin(), b.end(), x) - b.begin()));
            }
            CHECK(fks_raw(a, b) == static_cast<Real>(inversion_count(mapped)));
        } while (std::next_permutation(b.begin(), b.end()));
    } while (std::next_permutation(a.begin(), a.end()));
}

TEST_CASE("fks: disjoint lists attain the maximum exactly, bitwise") {
    for (const std::int32_t k : {1, 2, 3, 5, 50}) {
        std::vector<DocId> a(static_cast<std::size_t>(k));
        std::vector<DocId> b(static_cast<std::size_t>(k));
        std::iota(a.begin(), a.end(), 0);
        std::iota(b.begin(), b.end(), 1000);
        // Bit equality, not approximate: at p = 1/2 every addend is dyadic, so
        // the accumulated sum of k^2 ones and k(k-1) halves is exact.
        CHECK(same_bits(fks_raw(a, b), fks_max(k)));
        CHECK(kendall_fks(a, b) == 1.0);
    }
}

TEST_CASE("fks: fks_max matches the closed form k(3k-1)/2 at p = 1/2") {
    for (std::int32_t k = 1; k < 60; ++k) {
        CHECK(same_bits(fks_max(k), static_cast<Real>(k * (3 * k - 1)) / 2.0));
    }
    CHECK(fks_max(0) == 0.0);
    CHECK(fks_max(-3) == 0.0);
}

TEST_CASE("fks: two disjoint singletons are maximally distant, not identical") {
    // The k = 1 trap. One case-3 pair exists, so the ceiling is 1.
    CHECK(fks_max(1) == 1.0);
    CHECK(kendall_fks(std::vector<DocId>{1}, std::vector<DocId>{2}) == 1.0);
    CHECK(kendall_fks(std::vector<DocId>{1}, std::vector<DocId>{1}) == 0.0);
}

TEST_CASE("fks: is a near-metric, not a metric -- G2's witness") {
    // A and C are disjoint, so their distance is maximal, while B shares one
    // element with each. The triangle inequality fails by a wide margin. This
    // is intended behaviour and must not be "fixed".
    const std::vector<DocId> a{3, 1, 0};
    const std::vector<DocId> b{5, 3, 4};
    const std::vector<DocId> c{5, 4, 2};
    const Real d_ab = fks_raw(a, b);
    const Real d_bc = fks_raw(b, c);
    const Real d_ac = fks_raw(a, c);

    CHECK(d_ab == 6.0);
    CHECK(d_bc == 2.0);
    CHECK(d_ac == 12.0);
    CHECK(d_ac == fks_max(3));
    CHECK(d_ac > d_ab + d_bc);
}

TEST_CASE("fks: no penalty value restores the triangle inequality") {
    const std::vector<DocId> a{3, 1, 0};
    const std::vector<DocId> b{5, 3, 4};
    const std::vector<DocId> c{5, 4, 2};
    for (const Real p : {0.0, 0.25, 0.5, 1.0}) {
        CHECK(fks_raw(a, c, p) > fks_raw(a, b, p) + fks_raw(b, c, p));
    }
}

TEST_CASE("fks: p = 1/2 is exactly the midpoint of the two biased readings") {
    const std::vector<DocId> a{1, 2};  // disjoint, so only cases 3 and 4 arise
    const std::vector<DocId> b{3, 4};
    const Real optimistic = fks_raw(a, b, 0.0);
    const Real neutral = fks_raw(a, b, kFksPenalty);
    const Real pessimistic = fks_raw(a, b, 1.0);
    CHECK(optimistic < neutral);
    CHECK(neutral < pessimistic);
    CHECK(same_bits(neutral, (optimistic + pessimistic) / 2.0));
}

TEST_CASE("fks: case 2 penalises the absent element ranked first") {
    // b = [1] says 1 outranks the absent 2. a = [1, 2] agrees; a = [2, 1] does not.
    CHECK(kendall_fks(std::vector<DocId>{1, 2}, std::vector<DocId>{1}, kFksPenalty, false) == 0.0);
    CHECK(kendall_fks(std::vector<DocId>{2, 1}, std::vector<DocId>{1}, kFksPenalty, false) == 1.0);
}

TEST_CASE("fks: case 3 always counts, case 4 costs exactly p") {
    CHECK(kendall_fks(std::vector<DocId>{1}, std::vector<DocId>{2}, kFksPenalty, false) == 1.0);
    // {1, 2} versus {1, 2, 3, 4}: the pair (3, 4) is witnessed by b alone.
    const std::vector<DocId> a{1, 2};
    const std::vector<DocId> b{1, 2, 3, 4};
    CHECK(kendall_fks(a, b, 0.0, false) == 0.0);
    CHECK(kendall_fks(a, b, kFksPenalty, false) == 0.5);
    CHECK(kendall_fks(a, b, 1.0, false) == 1.0);
}

TEST_CASE("fks: stays within [0, 1] on random pairs of lists") {
    std::mt19937_64 rng(99);
    for (int trial = 0; trial < 300; ++trial) {
        std::vector<DocId> pool(10);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), rng);
        const std::size_t na = static_cast<std::size_t>(rng() % 7);
        const std::size_t nb = static_cast<std::size_t>(rng() % 7);
        const std::vector<DocId> a(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(na));
        std::shuffle(pool.begin(), pool.end(), rng);
        const std::vector<DocId> b(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(nb));
        const Real d = kendall_fks(a, b);
        CHECK(d >= 0.0);
        CHECK(d <= 1.0);
    }
}

// -----------------------------------------------------------------------------
// Set measures
// -----------------------------------------------------------------------------
TEST_CASE("disagreement is about the set, not the order") {
    CHECK_FALSE(top_k_disagreement(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{3, 2, 1}));
    CHECK(top_k_disagreement(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{1, 2, 4}));
}

TEST_CASE("jaccard distance") {
    CHECK(jaccard_distance(std::vector<DocId>{1, 2}, std::vector<DocId>{1, 2}) == 0.0);
    CHECK(jaccard_distance(std::vector<DocId>{1, 2}, std::vector<DocId>{3, 4}) == 1.0);
    CHECK(jaccard_distance(std::vector<DocId>{}, std::vector<DocId>{}) == 0.0);
    CHECK(same_bits(jaccard_distance(std::vector<DocId>{1, 2}, std::vector<DocId>{2, 3}),
                    1.0 - 1.0 / 3.0));
}

// -----------------------------------------------------------------------------
// The combined comparison
// -----------------------------------------------------------------------------
TEST_CASE("compare_top_k: identical lists") {
    const std::vector<DocId> a{1, 2, 3};
    const TopKComparison c = compare_top_k(a, a, 3);
    CHECK_FALSE(c.sets_differ);
    CHECK(c.fks == 0.0);
    CHECK(c.kendall_intersection == 0.0);
    CHECK(c.intersection_size == 3);
    CHECK(c.jaccard == 0.0);
    CHECK(c.swapped == 0);
    CHECK(c.k == 3);
}

TEST_CASE("compare_top_k: the intersection Kendall is undefined below two shared") {
    // Why K_int is never reported alone: it reads as "no reordering" here while
    // the set indicator and the FKS distance both see the change.
    const TopKComparison c =
        compare_top_k(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{1, 4, 5}, 3);
    CHECK(std::isnan(c.kendall_intersection));
    CHECK(c.intersection_size == 1);
    CHECK(c.sets_differ);
    CHECK(c.fks > 0.0);
    CHECK(c.swapped == 2);
}

TEST_CASE("compare_top_k: the intersection Kendall reports its support") {
    const TopKComparison c =
        compare_top_k(std::vector<DocId>{1, 2, 3, 4}, std::vector<DocId>{4, 3, 9, 8}, 4);
    CHECK(c.intersection_size == 2);
    CHECK(c.kendall_intersection == 1.0);  // 3 and 4 appear in opposite order
}

TEST_CASE("compare_top_k: truncates to k") {
    const std::vector<DocId> a{1, 2, 3, 4, 5};
    const std::vector<DocId> b{1, 2, 9, 8, 7};
    CHECK_FALSE(compare_top_k(a, b, 2).sets_differ);
    CHECK(compare_top_k(a, b, 3).sets_differ);
    CHECK(compare_top_k(a, b, 2).k == 2);
}

TEST_CASE("compare_top_k: invariants on random pairs") {
    std::mt19937_64 rng(5150);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<DocId> pool(10);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), rng);
        const auto na = static_cast<std::ptrdiff_t>(1 + rng() % 6);
        const std::vector<DocId> a(pool.begin(), pool.begin() + na);
        std::shuffle(pool.begin(), pool.end(), rng);
        const auto nb = static_cast<std::ptrdiff_t>(1 + rng() % 6);
        const std::vector<DocId> b(pool.begin(), pool.begin() + nb);
        const auto k = static_cast<std::int32_t>(1 + rng() % 6);

        const TopKComparison c = compare_top_k(a, b, k);
        CHECK(c.fks >= 0.0);
        CHECK(c.fks <= 1.0);
        CHECK(c.jaccard >= 0.0);
        CHECK(c.jaccard <= 1.0);
        CHECK(c.intersection_size <= static_cast<std::int32_t>(std::min(
                                         std::min(a.size(), static_cast<std::size_t>(k)),
                                         std::min(b.size(), static_cast<std::size_t>(k)))));
        if (!c.sets_differ) {
            CHECK(c.jaccard == 0.0);
            CHECK(c.swapped == 0);
        }
    }
}

TEST_CASE("compare_top_k: k beyond the lists, and k = 0") {
    const std::vector<DocId> a{1, 2};
    const std::vector<DocId> b{2, 1};
    CHECK(compare_top_k(a, b, 99).kendall_intersection == 1.0);
    const TopKComparison empty = compare_top_k(a, b, 0);
    CHECK_FALSE(empty.sets_differ);
    CHECK(empty.fks == 0.0);  // no pairs at all, and the ceiling is 0 -> 0, not NaN
    CHECK(empty.jaccard == 0.0);
    CHECK(std::isnan(empty.kendall_intersection));
}
