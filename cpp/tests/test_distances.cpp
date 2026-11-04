// Tests for the ordering distances: inversions, Kendall tau, FKS, set measures.
//
// K^(p) is a near-metric (G2's witness), and fks_max(1) == 1.0 because two
// disjoint singletons form one case-3 pair. Both are asserted as specified.
// Rationale: docs/engineering_notes.md#fks-near-metric-and-singleton-maximum
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

/// Returns the unnormalised FKS distance and checks that it is symmetric.
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
            // A small alphabet, so equal elements (the case the `<=` in the
            // merge decides) occur constantly rather than never.
            x = static_cast<std::int32_t>(rng() % 5);
        }
        CHECK(inversion_count(v) == brute_force_inversions(v));
    }
}

TEST_CASE("inversions: extremes") {
    CHECK(inversion_count(std::vector<std::int32_t>{}) == 0);
    CHECK(inversion_count(std::vector<std::int32_t>{1, 2, 3, 4}) == 0);
    CHECK(inversion_count(std::vector<std::int32_t>{4, 3, 2, 1}) == 6);  // C(4, 2)
    CHECK(inversion_count(std::vector<std::int32_t>{1, 1, 1}) == 0);     // equals: no inversion
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
    // The refusal is the signal that kendall_fks is the function wanted.
    const auto attempt = [](const std::vector<DocId>& a, const std::vector<DocId>& b) {
        static_cast<void>(kendall_tau_distance(a, b));  // discarding a [[nodiscard]] Real
    };
    CHECK_THROWS_AS(attempt({1, 2}, {1, 3}), std::invalid_argument);
    CHECK_THROWS_AS(attempt({1, 2}, {1}), std::invalid_argument);

    // The guard is `a.size() != b.size() || sa != sb` over sorted copies.
    // Without it {1, 2} against {1, 2, 2} maps to [0, 2], since `positions`
    // keeps the last index of a repeat, and zero inversions over the one pair
    // of a.size() = 2 give 0.0 instead of a refusal.
    CHECK_THROWS_AS(attempt({1, 2}, {1, 2, 2}), std::invalid_argument);
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
    // Nonzero because the distances are sixths, which binary64 cannot hold, so
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
        // Bit equality rather than approximate: at p = 1/2 every addend is
        // dyadic, so the sum of k^2 ones and k(k-1) halves is exact.
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
    // At k = 1 one case-3 pair exists, so the ceiling is 1.
    CHECK(fks_max(1) == 1.0);
    CHECK(kendall_fks(std::vector<DocId>{1}, std::vector<DocId>{2}) == 1.0);
    CHECK(kendall_fks(std::vector<DocId>{1}, std::vector<DocId>{1}) == 0.0);
}

TEST_CASE("fks: is a near-metric, not a metric -- G2's witness") {
    // A and C are disjoint, so their distance is maximal, while B shares one
    // element with each. The triangle inequality then fails by a wide margin,
    // which G2 specifies.
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

TEST_CASE("compare_top_k: prefixes of different lengths report what differs") {
    // Membership changes come in pairs only when the prefixes have the same
    // length. `swapped` reports the larger one-way difference, so an odd
    // symmetric difference cannot round to 0 while `sets_differ` reports true.
    // G19 makes unequal prefixes a protocol case.
    const TopKComparison one =
        compare_top_k(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{1, 2}, 3);
    CHECK(one.sets_differ);
    CHECK(one.swapped == 1);  // document 3 left the top-k
    CHECK(one.intersection_size == 2);

    // Disjoint and unequal: three left, two entered, and the larger is reported
    // rather than the mean of the two rounded down.
    const TopKComparison disjoint =
        compare_top_k(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{4, 5}, 3);
    CHECK(disjoint.swapped == 3);
    CHECK(disjoint.intersection_size == 0);
}

TEST_CASE("compare_top_k: the intersection Kendall is undefined below two shared") {
    // Why K_int is never reported alone: here it reads as "no reordering" while
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
    CHECK(empty.fks == 0.0);  // no pairs, and a ceiling of 0 yields 0 rather than NaN
    CHECK(empty.jaccard == 0.0);
    CHECK(std::isnan(empty.kendall_intersection));
}

TEST_CASE("fks: the union is walked in first-appearance order, not sorted") {
    // At a dyadic penalty every partial sum is exact, so the union order is
    // invisible. At p = 1/3 the addition order decides the last bit; the lists
    // are disjoint, so the addends depend only on the union order.
    // Rationale: docs/engineering_notes.md#fks-union-first-appearance-order
    const std::vector<DocId> a{5, 3, 1};
    const std::vector<DocId> b{4, 2, 0};

    CHECK(same_bits(kendall_fks(a, b, 1.0 / 3.0, false), 11.000000000000002));

    // The same input at a dyadic penalty, to show why the case above is needed:
    // here the sum is exact and no ordering can be distinguished.
    CHECK(kendall_fks(a, b, 0.5, false) == 12.0);
}

TEST_CASE("compare_top_k: a negative k is an empty prefix here, unlike in Python") {
    // `prefix` takes `std::max(k, 0)`, so a negative k compares nothing and
    // `k` carries the negative value through. Both Python entry points refuse
    // a negative k, so the divergence is unreachable from Python.
    // Rationale: docs/engineering_notes.md#compare-top-k-negative-k
    const std::vector<DocId> a{1, 2, 3};
    const std::vector<DocId> b{3, 2, 1};
    const std::vector<DocId> empty{};

    const auto c = compare_top_k(a, b, -1);
    CHECK(c.k == -1);  // carried through unclamped, so a caller can see what it asked for
    CHECK_FALSE(c.sets_differ);
    CHECK(c.intersection_size == 0);
    CHECK(c.swapped == 0);
    CHECK(std::isnan(c.kendall_intersection));
    // Stated as "identical to an empty prefix" rather than as constants: that
    // is the contract, and it holds whatever value the empty case defines.
    CHECK(same_bits(c.fks, kendall_fks(empty, empty)));
    CHECK(same_bits(c.jaccard, jaccard_distance(empty, empty)));

    // The prefix the clamp empties is informative at k = 2, so the divergence
    // is a difference in answers rather than in presentation.
    const auto two = compare_top_k(a, b, 2);
    CHECK(two.sets_differ);
    CHECK(two.intersection_size == 1);
}

TEST_CASE("fks: the normaliser is zero for every k below one, not just at zero") {
    // The normaliser is zero wherever no pair exists. Relaxing `k < 1` to `k < 0`
    // differs only at k == 0, where 0*0 + p*0*(0-1) = 0.0 + (-0.0) = +0.0, the
    // guard's own value. configs/equivalent_mutants_cpp.txt records that mutant.
    // Rationale: docs/engineering_notes.md#fks-max-guard-equivalence
    CHECK(fks_max(0) == 0.0);
    CHECK(fks_max(-1) == 0.0);
    CHECK(fks_max(-2) == 0.0);

    // Without any guard, at kFksPenalty = 0.5, k = -1 gives
    // 1 + 0.5*(-1)*(-2) = 2.0 and k = -2 gives 4 + 0.5*(-2)*(-3) = 7.0.
    CHECK(fks_max(1) == 1.0);   // and k = 1 is NOT degenerate: one case-3 pair
    CHECK(fks_max(2) == 5.0);   // 4 + 0.5*2*1
}

TEST_CASE("kendall tau: the same set with different multiplicities is refused") {
    // {1, 1, 2} and {1, 2, 2} share a set and a size, and `positions` keeps
    // only the last index of a repeat, so a set test would report 0.0. Sorted
    // copies compare multisets; the normative Python compares Counters, so
    // both sides refuse this input.
    const auto attempt = [](const std::vector<DocId>& a, const std::vector<DocId>& b) {
        static_cast<void>(kendall_tau_distance(a, b));
    };
    CHECK_THROWS_AS(attempt({1, 1, 2}, {1, 2, 2}), std::invalid_argument);

    // The same multiset is accepted, which is the limit of the check; no ranking
    // repeats a document, because every caller passes a permutation.
    CHECK(kendall_tau_distance(std::vector<DocId>{1, 1, 2},
                               std::vector<DocId>{1, 1, 2}) == 0.0);
}

TEST_CASE("fks: a penalty outside [0, 1] is refused by both entry points") {
    // G2 fixes the case-4 penalty in [0, 1]. Unguarded, a NaN penalty, or one negative enough
    // to make `fks_max` non-positive, fails `ceiling > 0.0`, and `kendall_fks` returns the 0.0
    // reserved for k = 0. The lambdas return void because both functions are [[nodiscard]].
    // Rationale: docs/engineering_notes.md#fks-penalty-domain-guard
    const auto max_attempt = [](Real p) { (void)fks_max(3, p); };
    const auto fks_attempt = [](Real p) {
        (void)kendall_fks(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{4, 5, 6}, p, true);
    };

    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    for (const Real bad : {nan, inf, -inf, -1.0, -0.5, 1.5, 2.0}) {
        CHECK_THROWS_AS(max_attempt(bad), std::invalid_argument);
        CHECK_THROWS_AS(fks_attempt(bad), std::invalid_argument);
    }

    // The endpoints stay admissible: G2 discusses p = 0 and p = 1 as the two
    // biased readings, so a guard that excluded them would refuse the spec.
    for (const Real good : {0.0, 0.25, kFksPenalty, 1.0}) {
        CHECK(fks_max(1, good) == 1.0);
        CHECK(kendall_fks(std::vector<DocId>{1, 2, 3}, std::vector<DocId>{4, 5, 6}, good, true) ==
              1.0);
    }
}

TEST_CASE("fks: k = 0 short-circuits before the penalty term runs") {
    // The guard is not observable: at k = 0 the fall-through computes
    // 0 + p * 0.0 * -1.0, which is bit-identically +0.0 for every admissible
    // penalty. A non-finite penalty would make that NaN, and `checked_penalty`
    // refuses one before k is tested.
    for (const Real p : {0.0, 0.25, kFksPenalty, 1.0}) {
        CHECK(fks_max(0, p) == 0.0);
        CHECK(std::signbit(fks_max(0, p)) == false);  // positive zero
    }
}
