// The ranking layer: sort keys, selection strategies, margins and tie groups.
//
// The central case is that all five selection strategies emit the identical
// permutation. Like `TAAT == DAAT` in the scoring layer, it pits structurally
// unrelated algorithms against each other and demands byte equality, which
// leaves little room for a comparator or indexing bug.
#include <tfidf/ranking/attributes.hpp>
#include <tfidf/ranking/margins.hpp>
#include <tfidf/ranking/ranker.hpp>
#include <tfidf/ranking/sort_keys.hpp>
#include <tfidf/ranking/tie_groups.hpp>

#include <doctest.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace tfidf;
using namespace tfidf::ranking;

namespace {

bool same_bits(Real a, Real b) {
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

/// A table with one attribute plus identifier ranks, as Python would supply.
RankTable make_table(const std::vector<std::int32_t>& attr) {
    RankTable t;
    t.n_docs = static_cast<std::int32_t>(attr.size());
    t.n_attrs = 1;
    t.ranks = attr;
    t.id_ranks.resize(attr.size());
    std::iota(t.id_ranks.begin(), t.id_ranks.end(), 0);
    return t;
}

std::vector<DocId> rank_with(const std::vector<Real>& scores,
                             const RankTable& table,
                             const std::vector<std::int32_t>& priority,
                             Selection how) {
    std::vector<SortKey> keys(scores.size());
    build_keys(scores, table, priority, keys);
    std::vector<DocId> out(scores.size());
    rank_full(keys, out, how);
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// The key
// -----------------------------------------------------------------------------
TEST_CASE("sort key: 32 bytes, two per cache line") {
    CHECK(sizeof(SortKey) == 32);
}

TEST_CASE("sort key: score negation is exact") {
    // A sign-bit flip never rounds, which is why Python's tuple `<` and this
    // `operator<` are the same relation.
    for (const Real s : {0.1, 0.3, 1e-300, 1e300, 0.0}) {
        CHECK(same_bits(-(-s), s));
    }
}

TEST_CASE("sort key: injective whenever identifier ranks are a bijection") {
    const RankTable t = make_table({7, 7, 7, 7});  // every attribute identical
    const std::vector<Real> scores(4, 0.5);        // every score identical
    std::vector<SortKey> keys(4);
    build_keys(scores, t, {}, keys);
    CHECK(keys_are_injective(keys));
    CHECK(t.id_ranks_are_a_bijection());
}

TEST_CASE("sort key: a duplicated identifier rank destroys injectivity") {
    RankTable t = make_table({1, 1});
    t.id_ranks = {0, 0};  // not a bijection
    CHECK_FALSE(t.id_ranks_are_a_bijection());
    const std::vector<Real> tied{0.5, 0.5};
    std::vector<SortKey> keys(2);
    build_keys(tied, t, {}, keys);
    CHECK_FALSE(keys_are_injective(keys));
}

TEST_CASE("sort key: finiteness guard") {
    CHECK(all_finite(std::vector<Real>{0.0, 1.0, -2.5}));
    CHECK_FALSE(all_finite(std::vector<Real>{0.0, std::nan("")}));
    CHECK_FALSE(all_finite(std::vector<Real>{0.0, INFINITY}));
}

// -----------------------------------------------------------------------------
// Selection
// -----------------------------------------------------------------------------
TEST_CASE("ranker: all five selection strategies agree") {
    std::mt19937_64 rng(20260811);
    // A small discrete alphabet, so exact ties are the rule. Uniform-random
    // doubles tie with probability ~0 and exercise none of the tie-break.
    const std::vector<Real> alphabet{0.0, 0.25, 0.5, 0.75};
    std::uniform_int_distribution<int> pick(0, 3);
    std::uniform_int_distribution<int> attr(0, 2);

    for (int trial = 0; trial < 50; ++trial) {
        const std::size_t n = 1 + static_cast<std::size_t>(rng() % 60);
        std::vector<Real> scores(n);
        std::vector<std::int32_t> ranks(n);
        for (std::size_t i = 0; i < n; ++i) {
            scores[i] = alphabet[static_cast<std::size_t>(pick(rng))];
            ranks[i] = attr(rng);
        }
        const RankTable t = make_table(ranks);
        const std::vector<std::int32_t> priority{0};

        const auto reference = rank_with(scores, t, priority, Selection::FullSort);
        for (const auto how : {Selection::StableSort, Selection::BoundedHeap}) {
            CHECK(rank_with(scores, t, priority, how) == reference);
        }
    }
}

TEST_CASE("ranker: the order is independent of the input order") {
    // The stronger corollary of totality: a non-total comparator can pass the
    // five-strategy check by luck, but cannot survive a permuted input.
    std::mt19937_64 rng(99);
    const std::size_t n = 40;
    std::vector<Real> scores(n);
    std::vector<std::int32_t> attrs(n);
    for (std::size_t i = 0; i < n; ++i) {
        scores[i] = static_cast<Real>(rng() % 3) * 0.25;
        attrs[i] = static_cast<std::int32_t>(rng() % 3);
    }

    const RankTable t = make_table(attrs);
    const std::vector<std::int32_t> priority{0};
    const auto forward = rank_with(scores, t, priority, Selection::FullSort);

    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<Real> permuted_scores(n);
    std::vector<std::int32_t> permuted_attrs(n);
    RankTable pt;
    pt.n_docs = static_cast<std::int32_t>(n);
    pt.n_attrs = 1;
    pt.id_ranks.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        permuted_scores[i] = scores[perm[i]];
        permuted_attrs[i] = attrs[perm[i]];
        pt.id_ranks[i] = static_cast<std::int32_t>(perm[i]);  // identity travels with the doc
    }
    pt.ranks = permuted_attrs;

    const auto shuffled = rank_with(permuted_scores, pt, {0}, Selection::FullSort);
    REQUIRE(shuffled.size() == forward.size());
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(static_cast<std::size_t>(forward[i]) == perm[static_cast<std::size_t>(shuffled[i])]);
    }
}

TEST_CASE("ranker: partial selection matches the full ranking's prefix") {
    std::mt19937_64 rng(7);
    const std::size_t n = 50;
    std::vector<Real> scores(n);
    std::vector<std::int32_t> attrs(n);
    for (std::size_t i = 0; i < n; ++i) {
        scores[i] = static_cast<Real>(rng() % 4) * 0.25;
        attrs[i] = static_cast<std::int32_t>(rng() % 3);
    }
    const RankTable t = make_table(attrs);
    const std::vector<std::int32_t> priority{0};
    const auto full = rank_with(scores, t, priority, Selection::FullSort);

    for (const std::size_t m : {std::size_t{1}, std::size_t{5}, std::size_t{20}, n}) {
        for (const auto how : {Selection::PartialSort, Selection::NthElement}) {
            std::vector<SortKey> keys(n);
            build_keys(scores, t, priority, keys);
            std::vector<DocId> out(m);
            select_top(keys, m, out, how);
            for (std::size_t i = 0; i < m; ++i) {
                CHECK(out[i] == full[i]);
            }
            // The only postcondition guaranteed across standard libraries.
            CHECK(partition_is_valid(keys, m));
        }
    }
}

TEST_CASE("ranker: score dominates every attribute") {
    const RankTable t = make_table({9, 0});  // doc 1 has the better attribute
    const auto order = rank_with({1.0, 0.0}, t, {0}, Selection::FullSort);
    CHECK(order[0] == 0);  // but doc 0 has the better score
}

TEST_CASE("ranker: an empty priority falls through to the identifier") {
    const RankTable t = make_table({9, 0, 5});
    const std::vector<Real> scores{0.5, 0.5, 0.5};
    const auto order = rank_with(scores, t, {}, Selection::FullSort);
    CHECK(order == std::vector<DocId>{0, 1, 2});
}

TEST_CASE("ranker: sorted_scores_desc is non-increasing and a permutation") {
    const std::vector<Real> scores{0.3, 0.9, 0.1, 0.9, 0.0};
    const auto s = sorted_scores_desc(scores);
    CHECK(std::is_sorted(s.begin(), s.end(), std::greater<>()));
    auto a = scores;
    auto b = s;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    CHECK(a == b);
}

// -----------------------------------------------------------------------------
// Margins
// -----------------------------------------------------------------------------
TEST_CASE("margins: boundary and minimum adjacent") {
    const std::vector<Real> s{1.0, 0.75, 0.5, 0.5, 0.25};
    CHECK(boundary_margin(s, 1).value == 0.25);
    CHECK(boundary_margin(s, 3).value == 0.0);
    CHECK(boundary_margin(s, 3).is_exact_tie());
    CHECK(min_adjacent_margin_top(s, 4).value == 0.0);
    CHECK(min_adjacent_margin_top(s, 2).value == 0.25);
}

TEST_CASE("margins: undefined cases are NaN plus a flag, never coerced") {
    const std::vector<Real> s{1.0, 0.5, 0.25};
    const Margin at_n = boundary_margin(s, 3);
    CHECK(std::isnan(at_n.value));
    CHECK_FALSE(at_n.defined);
    CHECK_FALSE(at_n.is_exact_tie());
    CHECK_FALSE(std::isinf(at_n.value));

    const Margin vacuous = min_adjacent_margin_top(s, 1);
    CHECK(std::isnan(vacuous.value));
    CHECK_FALSE(vacuous.defined);
}

TEST_CASE("margins: the flip radius is exactly half, bitwise") {
    const std::vector<std::vector<Real>> cases{{1.0, 0.75}, {0.3, 0.1}, {1.0, 1.0}};
    for (const auto& s : cases) {
        const Margin m = boundary_margin(s, 1);
        CHECK(same_bits(m.flip_radius() * 2.0, m.value));
    }
}

TEST_CASE("margins: adjacent gaps") {
    CHECK(adjacent_gaps(std::vector<Real>{1.0, 0.75, 0.5}) == std::vector<Real>{0.25, 0.25});
    CHECK(adjacent_gaps(std::vector<Real>{1.0}).empty());
    CHECK(adjacent_gaps(std::vector<Real>{}).empty());
}

// -----------------------------------------------------------------------------
// Tie groups
// -----------------------------------------------------------------------------
TEST_CASE("tie groups: the adversarial ladder is not transitive") {
    // Every value and difference is exactly representable, so the case carries
    // no floating-point content; it is about structure.
    constexpr Real kTau = 0x1p-20;
    std::vector<Real> s(6);
    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] = static_cast<Real>(5 - static_cast<int>(i)) * kTau;
    }

    const auto [lo1, hi1] = tie_ball_interval(s, 1, kTau);
    const auto [lo0, hi0] = tie_ball_interval(s, 0, kTau);
    CHECK(lo1 == 0);
    CHECK(hi1 == 3);  // {0, 1, 2}
    CHECK(lo0 == 0);
    CHECK(hi0 == 2);  // {0, 1}; 2 is absent
}

TEST_CASE("tie groups: a chain swallows the ladder, cliques see only pairs") {
    constexpr Real kTau = 0x1p-20;
    std::vector<Real> s(6);
    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] = static_cast<Real>(5 - static_cast<int>(i)) * kTau;
    }
    CHECK(tie_chains(s, kTau).size() == 1);
    CHECK(tie_cliques(s, kTau).size() == 5);
    CHECK(chain_inflation_ratio(s, kTau) == 3.0);  // 6 / 2

    // One ulp below tau the ladder shatters into singletons.
    const Real just_under = kTau - std::nextafter(kTau, 0.0) == 0.0
                                ? kTau
                                : std::nextafter(kTau, 0.0);
    CHECK(tie_chains(s, just_under).size() == 6);
    CHECK(chain_inflation_ratio(s, just_under) == 1.0);
}

TEST_CASE("tie groups: tau = 0 recovers exact equality classes") {
    const std::vector<Real> s{1.0, 0.5, 0.5, 0.5, 0.25};
    const auto chains = tie_chains(s, 0.0);
    const auto cliques = tie_cliques(s, 0.0);
    CHECK(chains == cliques);
    CHECK(chains.size() == 3);
    CHECK(chain_inflation_ratio(s, 0.0) == 1.0);
}

TEST_CASE("tie groups: the ball search agrees with a linear scan") {
    // Certifies the monotone-difference search against G9's literal predicate.
    std::mt19937_64 rng(4242);
    std::uniform_real_distribution<Real> val(0.0, 1.0);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + static_cast<std::size_t>(rng() % 40);
        std::vector<Real> s(n);
        for (auto& x : s) {
            x = val(rng);
        }
        std::sort(s.begin(), s.end(), std::greater<>());
        const Real tau = val(rng) * 0.3;
        const auto j = static_cast<std::int32_t>(rng() % n);

        std::set<std::int32_t> expected;
        for (std::int32_t i = 0; i < static_cast<std::int32_t>(n); ++i) {
            if (std::abs(s[static_cast<std::size_t>(i)] - s[static_cast<std::size_t>(j)]) <= tau) {
                expected.insert(i);
            }
        }
        const auto [lo, hi] = tie_ball_interval(s, j, tau);
        std::set<std::int32_t> got;
        for (std::int32_t i = lo; i < hi; ++i) {
            got.insert(i);
        }
        CHECK(got == expected);
    }
}

TEST_CASE("tie groups: chains partition and cliques have diameter at most tau") {
    std::mt19937_64 rng(31337);
    std::uniform_real_distribution<Real> val(0.0, 1.0);
    for (int trial = 0; trial < 100; ++trial) {
        const std::size_t n = 1 + static_cast<std::size_t>(rng() % 30);
        std::vector<Real> s(n);
        for (auto& x : s) {
            x = val(rng);
        }
        std::sort(s.begin(), s.end(), std::greater<>());
        const Real tau = val(rng) * 0.25;

        std::int32_t covered = 0;
        for (const auto& [lo, hi] : tie_chains(s, tau)) {
            CHECK(lo == covered);
            covered = hi;
        }
        CHECK(covered == static_cast<std::int32_t>(n));

        for (const auto& [lo, hi] : tie_cliques(s, tau)) {
            CHECK(s[static_cast<std::size_t>(lo)] - s[static_cast<std::size_t>(hi) - 1] <= tau);
        }
        CHECK(chain_inflation_ratio(s, tau) >= 1.0);
    }
}

TEST_CASE("tie groups: empty and singleton corpora") {
    CHECK(tie_chains(std::vector<Real>{}, 0.1).empty());
    CHECK(tie_cliques(std::vector<Real>{}, 0.1).empty());
    CHECK(std::isnan(chain_inflation_ratio(std::vector<Real>{}, 0.1)));

    const std::vector<Real> one{0.5};
    CHECK(tie_chains(one, 0.0).size() == 1);
    CHECK(chain_inflation_ratio(one, 0.0) == 1.0);
}

// -----------------------------------------------------------------------------
// Exact rational comparison (used only by the native tests)
// -----------------------------------------------------------------------------
TEST_CASE("attributes: ratio_less separates means that binary64 collides") {
    // 1/3 and (10^17+1)/(3*10^17) are different reals rounding to the same
    // double. The cross-products stay inside int64.
    const std::int64_t a_num = 1;
    const std::int64_t a_den = 3;
    const std::int64_t b_num = 100000000000000001LL;
    const std::int64_t b_den = 300000000000000000LL;

    CHECK(static_cast<Real>(a_num) / static_cast<Real>(a_den) ==
          static_cast<Real>(b_num) / static_cast<Real>(b_den));
    CHECK(ratio_less(a_num, a_den, b_num, b_den));
    CHECK_FALSE(ratio_less(b_num, b_den, a_num, a_den));
}

TEST_CASE("attributes: ratio_less is a strict order") {
    const std::int64_t pairs[][2] = {{1, 3}, {1, 2}, {2, 3}, {3, 4}, {1, 1}};
    for (const auto& a : pairs) {
        CHECK_FALSE(ratio_less(a[0], a[1], a[0], a[1]));  // irreflexive
        for (const auto& b : pairs) {
            if (ratio_less(a[0], a[1], b[0], b[1])) {
                CHECK_FALSE(ratio_less(b[0], b[1], a[0], a[1]));  // asymmetric
            }
        }
    }
}
