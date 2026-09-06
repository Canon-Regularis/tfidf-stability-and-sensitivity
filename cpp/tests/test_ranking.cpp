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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <utility>
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

/// A table with several attributes, `attrs[a][d]` being document `d`'s rank
/// under attribute `a`. `make_table` covers the one-attribute case; priority
/// order cannot be tested with one attribute.
RankTable make_wide_table(const std::vector<std::vector<std::int32_t>>& attrs) {
    RankTable t;
    t.n_docs = static_cast<std::int32_t>(attrs.front().size());
    t.n_attrs = static_cast<std::int32_t>(attrs.size());
    for (const auto& row : attrs) {
        t.ranks.insert(t.ranks.end(), row.begin(), row.end());
    }
    t.id_ranks.resize(attrs.front().size());
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

TEST_CASE("ranker: a score tie is broken by the attribute, not by the identifier") {
    // The case the suite did not have. Every other ranking test either gives
    // the documents distinct scores, so the attribute is never consulted, or
    // passes an empty priority, so there is no attribute to consult. Deleting
    // the whole rank-copying loop from `build_keys` -- leaving every attribute
    // rank at the zero `fill` puts there, so every tie fell through to the
    // identifier -- left all 98 cases in this suite passing.
    //
    // Section 4.5 is about exactly this ordering, so nothing else in the
    // repository has more reason to be pinned.
    const RankTable t = make_table({9, 0});  // doc 1 has the better attribute
    const std::vector<Real> tied{0.5, 0.5};

    CHECK(rank_with(tied, t, {0}, Selection::FullSort) == std::vector<DocId>{1, 0});
    // Contrast with the same scores and no priority, which is the order the
    // identifier alone gives: the two must differ, or the check above would
    // hold whether or not the attribute was read.
    CHECK(rank_with(tied, t, {}, Selection::FullSort) == std::vector<DocId>{0, 1});
}

TEST_CASE("ranker: the priority decides which attribute is consulted first") {
    // G15 makes the reordering of the priority an ablation, so which attribute
    // leads has to be a property of `priority` rather than of the table's row
    // order. The two attributes here disagree, so whichever leads wins.
    const RankTable t = make_wide_table({{1, 0}, {0, 1}});
    const std::vector<Real> tied{0.5, 0.5};

    CHECK(rank_with(tied, t, {0, 1}, Selection::FullSort) == std::vector<DocId>{1, 0});
    CHECK(rank_with(tied, t, {1, 0}, Selection::FullSort) == std::vector<DocId>{0, 1});
}

TEST_CASE("ranker: a later attribute decides only when every earlier one ties") {
    // The lexicographic step. Documents 0 and 1 tie on the leading attribute
    // and separate on the second; document 2 loses on the leading one, so the
    // second never gets to rescue it.
    const RankTable t = make_wide_table({{0, 0, 7}, {5, 1, 0}});
    const std::vector<Real> tied{0.5, 0.5, 0.5};

    CHECK(rank_with(tied, t, {0, 1}, Selection::FullSort) == std::vector<DocId>{1, 0, 2});
    // And with only the leading attribute in the priority, the pair that ties
    // on it falls through to the identifier instead of to attribute 1.
    CHECK(rank_with(tied, t, {0}, Selection::FullSort) == std::vector<DocId>{0, 1, 2});
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

TEST_CASE("margins: an exact tie is a defined margin of zero, not any defined margin") {
    // `is_exact_tie` is `defined && value == 0.0`. The two cases already tested
    // agree with `defined || value == 0.0`: a zero margin is defined, and an
    // undefined one carries NaN, which equals nothing. What separates them is a
    // margin that is defined and NOT zero, which nothing asserted -- so the
    // conjunction could be a disjunction and every top-k boundary would report
    // itself as an exact tie, sending membership to the tie-break.
    const std::vector<Real> s{1.0, 0.75, 0.5, 0.5, 0.25};

    const Margin gap = boundary_margin(s, 1);
    REQUIRE(gap.defined);
    REQUIRE(gap.value == 0.25);
    CHECK_FALSE(gap.is_exact_tie());

    // And the two directions that already held, kept together with it so the
    // three cases read as the truth table they are.
    const Margin tie = boundary_margin(s, 3);
    CHECK(tie.defined);
    CHECK(tie.value == 0.0);
    CHECK(tie.is_exact_tie());

    const Margin undefined = boundary_margin(s, 5);
    CHECK_FALSE(undefined.defined);
    CHECK_FALSE(undefined.is_exact_tie());
}

TEST_CASE("margins: two scores have one gap, which is the boundary of the guard") {
    // `adjacent_gaps` returns early below two scores. The existing cases use
    // three, one and none, and every one of them agrees with a guard placed at
    // three instead of two: the interesting size is exactly two, the smallest
    // input that has a gap at all.
    CHECK(adjacent_gaps(std::vector<Real>{1.0, 0.75}) == std::vector<Real>{0.25});
    CHECK(adjacent_gaps(std::vector<Real>{1.0}).empty());

    // Length is N-1 for every N, which is the property the reserve() above the
    // loop encodes and the loop bound has to agree with.
    for (std::size_t n = 2; n <= 6; ++n) {
        std::vector<Real> scores;
        for (std::size_t i = 0; i < n; ++i) {
            scores.push_back(1.0 - 0.1 * static_cast<Real>(i));
        }
        CHECK(adjacent_gaps(scores).size() == n - 1);
    }
}

TEST_CASE("margins: k = 0 is undefined rather than a read below the first score") {
    // `k <= 0` returns undefined. Relaxed to `k < 0`, k = 0 passes the guard,
    // `k_effective` is 0, the `k_effective >= n` arm is false for any non-empty
    // input, and the subtraction then indexes `sorted_scores[-1]`. The margin
    // comes back marked defined either way, so the flag is what a test can see.
    const std::vector<Real> s{1.0, 0.75, 0.5};

    const Margin zero = boundary_margin(s, 0);
    CHECK_FALSE(zero.defined);
    CHECK(std::isnan(zero.value));
    CHECK(zero.k == 0);

    const Margin negative = boundary_margin(s, -1);
    CHECK_FALSE(negative.defined);
    CHECK(std::isnan(negative.value));

    const Margin zero_top = min_adjacent_margin_top(s, 0);
    CHECK_FALSE(zero_top.defined);
    CHECK(std::isnan(zero_top.value));
}

TEST_CASE("margins: the minimum runs over the gaps inside the top-k and no further") {
    // The loop stops at `j + 1 < k_effective`, so it reads the gaps between
    // ranks 1..k and never the boundary gap at k -> k+1. That is the whole
    // distinction between this and `boundary_margin`.
    //
    // The existing case cannot see it: with {1, .75, .5, .5, .25} at k = 4 the
    // extra gap the relaxed bound would include is 0.25, and the minimum is
    // already 0.0, so including it changes nothing. Here the gap just outside
    // the top-k is the smallest in the list, so reading one too far returns it.
    const std::vector<Real> s{1.0, 0.75, 0.5, 0.4999};

    const Margin top = min_adjacent_margin_top(s, 3);
    CHECK(top.defined);
    CHECK(top.value == 0.25);  // min(1.0-0.75, 0.75-0.5); NOT 0.5-0.4999

    // The gap the loop must not reach is genuinely smaller, so the case
    // discriminates rather than happening to agree.
    const Margin boundary = boundary_margin(s, 3);
    CHECK(boundary.defined);
    CHECK(boundary.value < top.value);
}

TEST_CASE("margins: two scores are enough for a minimum over the top") {
    // The guard is three clauses, and `n < 2` is DEAD in the original: reaching
    // it needs k_effective = min(k, n) >= 2, which forces n >= 2. The Python
    // mirror marks its copy "pragma: no cover - defensive" for that reason.
    // Relaxing it to `n <= 2` resurrects it into a live guard that refuses the
    // smallest corpus with a gap at all, and no case in this file passes a
    // two-element span to this function -- the defined cases use five elements
    // and the undefined case uses three.
    const std::vector<Real> two{1.0, 0.75};

    const Margin m = min_adjacent_margin_top(two, 2);
    CHECK(m.defined);
    CHECK(m.value == 0.25);

    // The normative Python returns the same, so a mutant that refuses this
    // input diverges from the reference on a two-document corpus.
    CHECK(boundary_margin(two, 1).value == 0.25);
}

TEST_CASE("margins: a computed minimum is marked defined") {
    // `m.defined = true` at the end of `min_adjacent_margin_top` had no
    // assertion behind it: every existing case reads `.value`, which is set on
    // the same path, and the only `.defined` assertions are the FALSE ones on
    // the undefined branches. Flipped to false, every certificate built from
    // this margin would silently become undefined.
    const std::vector<Real> s{1.0, 0.75, 0.5};

    const Margin m = min_adjacent_margin_top(s, 2);
    CHECK(m.defined);
    CHECK(m.value == 0.25);
    CHECK_FALSE(m.is_exact_tie());

    const Margin b = boundary_margin(s, 1);
    CHECK(b.defined);
}

TEST_CASE("ranker: select_top writes the first slot, not only the rest") {
    // `select_top` copies the chosen keys with `for (i = 0; i < m; ++i)`. The
    // existing prefix test does compare out[0], but its caller sizes `out` with
    // `std::vector<DocId> out(m)`, which zero-initialises it -- and for that
    // test's seed the top document IS document 0, so a loop starting at 1 left
    // the right answer in place and the case passed. Fifty documents over four
    // distinct scores tie heavily at the top, and ties fall to the identifier,
    // which favours the lowest id.
    //
    // Here document 0 has the worst score, so nothing can leave a correct out[0]
    // behind by accident.
    const std::vector<Real> scores{0.1, 0.9, 0.5};
    const RankTable t = make_table({0, 0, 0});
    const std::vector<std::int32_t> priority{0};

    for (const auto how : {Selection::PartialSort, Selection::NthElement,
                           Selection::FullSort}) {
        std::vector<SortKey> keys(scores.size());
        build_keys(scores, t, priority, keys);
        std::vector<DocId> out(2, -1);  // a sentinel, not the zero doc id
        select_top(keys, 2, out, how);
        CHECK(out[0] == 1);  // the best score, and not document 0
        CHECK(out[1] == 2);
    }
}

TEST_CASE("ranker: partition_is_valid can actually say no") {
    // Its false branch had no test. Every call site asserts the postcondition
    // holds, so a body that returned true unconditionally satisfied all of
    // them -- and the check is the only guarantee this project has about
    // `nth_element`, whose remainder is explicitly implementation-defined.
    const std::vector<Real> scores{0.9, 0.7, 0.5, 0.3, 0.1};
    const RankTable t = make_table({0, 0, 0, 0, 0});
    const std::vector<std::int32_t> priority{0};

    std::vector<SortKey> keys(scores.size());
    build_keys(scores, t, priority, keys);
    std::sort(keys.begin(), keys.end(), key_less);
    REQUIRE(partition_is_valid(keys, 2));

    SUBCASE("a violation at the very first key") {
        // The loop over the prefix starts at i = 0, and m = 1 is what isolates
        // that: with a one-element prefix there is no other index to catch the
        // violation, so a loop starting at 1 runs zero iterations and reports a
        // broken partition as valid. At m = 2 the displaced key lands in the
        // tail and index 1 fails against it too, which hides the off-by-one.
        std::vector<SortKey> broken = keys;
        std::swap(broken[0], broken[1]);
        CHECK_FALSE(partition_is_valid(broken, 1));

        std::vector<SortKey> far = keys;
        std::swap(far[0], far[4]);
        CHECK_FALSE(partition_is_valid(far, 1));
        CHECK_FALSE(partition_is_valid(far, 2));
    }

    SUBCASE("a violation at a later key") {
        std::vector<SortKey> broken = keys;
        std::swap(broken[1], broken[4]);
        CHECK_FALSE(partition_is_valid(broken, 2));
    }

    SUBCASE("m at or past the end is vacuously valid") {
        // The early return: with no remainder there is nothing to compare
        // against, so any arrangement satisfies the postcondition.
        std::vector<SortKey> shuffled = keys;
        std::swap(shuffled[0], shuffled[4]);
        CHECK(partition_is_valid(shuffled, shuffled.size()));
        CHECK(partition_is_valid(shuffled, shuffled.size() + 3));
    }
}

TEST_CASE("attributes: an empty table is vacuously a bijection") {
    // `n_docs` defaults to 0, and the last line of `id_ranks_are_a_bijection`
    // is `id_ranks.size() == n_docs`. Every other case builds the table through
    // a helper that sets `n_docs`, so the default was never read -- and a
    // default of 1 makes an empty table report that its (empty) identifier
    // ranks are not a bijection, which is the one arrangement for which the
    // question is trivially yes.
    const RankTable empty;
    CHECK(empty.n_docs == 0);
    CHECK(empty.n_attrs == 0);
    CHECK(empty.id_ranks_are_a_bijection());

    // And a table whose size and rank count disagree is not, which is the same
    // final comparison read the other way.
    RankTable mismatched;
    mismatched.n_docs = 2;
    mismatched.id_ranks = {0};
    CHECK_FALSE(mismatched.id_ranks_are_a_bijection());
}
