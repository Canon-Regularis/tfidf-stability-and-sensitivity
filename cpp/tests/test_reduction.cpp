// Tests for the summation policies in `tfidf/core/reduction.hpp`.
//
// The policies agree on exactly representable inputs and differ on rounded ones. The
// noise-floor measurement that derives tau (section 7.1) depends on that difference.
#include <tfidf/core/reduction.hpp>

#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace tfidf;

namespace {

/// Compares two values bit for bit, which is what "bit-exact" means here.
///
/// `==` conflates -0.0 with 0.0 and treats two NaNs as unequal.
bool same_bits(Real a, Real b) {
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

Real naive_reference(const std::vector<Real>& v) {
    Real s = 0.0;
    for (const Real x : v) {
        s += x;
    }
    return s;
}

/// Computes the documented `Pairwise` tree independently, as an oracle.
///
/// Runs of `kPairwiseBlock` values are summed left to right, then combined two at a
/// time until one remains. For a power-of-two block count this is the documented tree.
Real pairwise_reference(const std::vector<Real>& v) {
    std::vector<Real> level;
    for (std::size_t i = 0; i < v.size(); i += kPairwiseBlock) {
        Real s = 0.0;
        for (std::size_t j = i; j < std::min(i + kPairwiseBlock, v.size()); ++j) {
            s += v[j];
        }
        level.push_back(s);
    }
    while (level.size() > 1) {
        std::vector<Real> merged;
        for (std::size_t i = 0; i + 1 < level.size(); i += 2) {
            merged.push_back(level[i] + level[i + 1]);
        }
        if (level.size() % 2 != 0) {
            merged.push_back(level.back());
        }
        level = merged;
    }
    return level.empty() ? 0.0 : level.front();
}

/// Folds the same block sums left to right instead of combining them pairwise.
///
/// A result that differs from `pairwise_reference` shows the tree shape is observable.
Real blockwise_fold_reference(const std::vector<Real>& v) {
    Real total = 0.0;
    for (std::size_t i = 0; i < v.size(); i += kPairwiseBlock) {
        Real s = 0.0;
        for (std::size_t j = i; j < std::min(i + kPairwiseBlock, v.size()); ++j) {
            s += v[j];
        }
        total += s;
    }
    return total;
}

}  // namespace

TEST_CASE("reduce: every policy agrees on exactly representable values") {
    // Powers of two sum exactly, so rounding cannot distinguish the policies.
    const std::vector<Real> v{1.0, 2.0, 4.0, 8.0, 0.5, 0.25};
    for (const auto p : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                         Reduction::Exact}) {
        CHECK(reduce::sum(v, p) == 15.75);
    }
}

TEST_CASE("reduce: the empty and singleton sums") {
    const std::vector<Real> empty{};
    const std::vector<Real> one{42.5};
    for (const auto p : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                         Reduction::Exact}) {
        CHECK(reduce::sum(empty, p) == 0.0);
        CHECK(reduce::sum(one, p) == 42.5);
    }
}

TEST_CASE("reduce: Naive is exactly a left-to-right fold") {
    // The normative policy, so it must match the literal reading of the formula
    // and nothing cleverer.
    std::mt19937_64 rng(20260811);
    std::uniform_real_distribution<Real> mag(-1.0, 1.0);
    std::uniform_int_distribution<int> exp10(-12, 3);
    std::vector<Real> v;
    v.reserve(5000);
    for (int i = 0; i < 5000; ++i) {
        v.push_back(mag(rng) * std::pow(10.0, exp10(rng)));
    }
    CHECK(same_bits(reduce::sum(v, Reduction::Naive), naive_reference(v)));
}

TEST_CASE("reduce: compensation recovers what the naive fold discards") {
    // Each addend is individually below half an ulp of the running total, so
    // the naive fold loses all of them; their exact sum is not.
    std::vector<Real> v{1.0};
    v.insert(v.end(), 100, 1e-17);
    REQUIRE(1.0 + 1e-17 == 1.0);

    const Real naive = reduce::sum(v, Reduction::Naive);
    const Real exact = reduce::sum(v, Reduction::Exact);
    const Real neumaier = reduce::sum(v, Reduction::Neumaier);

    CHECK(naive == 1.0);
    CHECK(exact > naive);
    CHECK(neumaier > naive);
    // Exact is correctly rounded, hence the ground truth for the others.
    CHECK(std::abs(exact - (1.0 + 100 * 1e-17)) <= std::numeric_limits<Real>::epsilon());
}

TEST_CASE("reduce: Neumaier selects its branch by magnitude") {
    // `add` subtracts the smaller operand from the larger, where the subtraction is
    // exact. Mixed magnitudes with cancellation expose the choice: with `magnitude`
    // as the identity, Neumaier returns -2.0 on this sum.
    // Rationale: docs/engineering_notes.md#reduction-test-neumaier-branch
    const std::vector<Real> v{0.5, -1e16, -0.5, -2.3, 1e16};

    CHECK(same_bits(reduce::sum(v, Reduction::Neumaier), -2.3));
    // Exact is correctly rounded, so it is the ground truth Neumaier matches.
    CHECK(same_bits(reduce::sum(v, Reduction::Exact), -2.3));
    // The uncompensated policies lose the 0.3, so the case separates the policies.
    CHECK(same_bits(reduce::sum(v, Reduction::Naive), -2.0));
    CHECK(same_bits(reduce::sum(v, Reduction::Pairwise), -2.0));

    // A second case, one ulp wide, for the threshold of `magnitude`. The operands
    // above are far apart in magnitude, which a comparison against 1.0 still orders
    // correctly; these four straddle 1.0, where it does not.
    const std::vector<Real> near_one{-0.1, 0.5, 0.25, -0.5};

    CHECK(same_bits(reduce::sum(near_one, Reduction::Neumaier), 0.15));
    CHECK(same_bits(reduce::sum(near_one, Reduction::Exact), 0.15));
    CHECK(same_bits(reduce::sum(near_one, Reduction::Naive), 0.15000000000000002));
}

TEST_CASE("reduce: Exact recovers a result naive arithmetic destroys") {
    // The classic ill-conditioned sum. Naive loses the first 1.0, below half an ulp of
    // 1e100, and keeps the second, which arrives after the cancellation: it returns 1.0.
    // The exact sum is 2.0.
    // Rationale: docs/engineering_notes.md#reduction-test-ill-conditioned-sum
    const std::vector<Real> v{1e100, 1.0, -1e100, 1.0};
    CHECK(reduce::sum(v, Reduction::Exact) == 2.0);
    CHECK(reduce::sum(v, Reduction::Naive) == 1.0);
}

TEST_CASE("reduce: Exact reproduces the documented fsum half-even case") {
    // CPython's own regression case for the correction step in Exact::value():
    // the 1e-16 nudges the 1 closer to two, so a correctly-rounded collapse
    // must round the last digit up rather than down.
    const std::vector<Real> v{1e-16, 1.0, 1e16};
    CHECK(reduce::sum(v, Reduction::Exact) == 1.0000000000000002e16);
}

TEST_CASE("reduce: Exact is order-independent") {
    // A correctly-rounded sum depends only on the multiset, never on order,
    // which is what lets it serve as ground truth.
    std::mt19937_64 rng(7);
    std::vector<Real> v;
    for (int i = 0; i < 400; ++i) {
        v.push_back(std::ldexp(1.0, i % 80 - 40) * ((i % 3) ? 1.0 : -1.0));
    }
    const Real forward = reduce::sum(v, Reduction::Exact);
    std::shuffle(v.begin(), v.end(), rng);
    CHECK(same_bits(reduce::sum(v, Reduction::Exact), forward));
}

TEST_CASE("reduce: Naive is NOT order-independent") {
    // Counterpart of the previous test. The summation-order sensitivity the paper
    // studies exists only because Naive depends on order.
    std::vector<Real> v{1.0};
    v.insert(v.end(), 100, 1e-17);
    const Real small_first = [&] {
        std::vector<Real> w(v.rbegin(), v.rend());
        return reduce::sum(w, Reduction::Naive);
    }();
    CHECK(reduce::sum(v, Reduction::Naive) != small_first);
}

TEST_CASE("reduce: Pairwise beats Naive on a long uniform sum") {
    // 2^20 identical values: pairwise error grows as O(log n), naive as O(n).
    const std::vector<Real> v(1u << 20, 0.1);
    const Real exact = reduce::sum(v, Reduction::Exact);
    const Real naive_err = std::abs(reduce::sum(v, Reduction::Naive) - exact);
    const Real pair_err = std::abs(reduce::sum(v, Reduction::Pairwise) - exact);
    CHECK(pair_err <= naive_err);
}

TEST_CASE("reduce: overflow follows IEEE-754 and is not silently papered over") {
    // {1e308, 1e308, ...} overflows on the first addition. That is the correct
    // IEEE result, and README section 6 applies no stabilising transformation,
    // so `inf` is what the specification calls for. Asserted so a later rescaled
    // sum cannot change every published digit unnoticed.
    const std::vector<Real> overflowing{1e308, 1e308, -1e308, -1e308};
    CHECK(std::isinf(reduce::sum(overflowing, Reduction::Naive)));

    // Just below the overflow threshold everything stays finite and exact.
    const std::vector<Real> large{1e308, -1e308, 1.0};
    for (const auto p : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                         Reduction::Exact}) {
        const Real s = reduce::sum(large, p);
        CHECK(std::isfinite(s));
        CHECK(s == 1.0);
    }
}

TEST_CASE("reduce: Pairwise combines whole blocks in the documented tree") {
    // `Pairwise` is a streaming accumulator with a partials stack. It matches the
    // documented recursive split only if the block size, the weight of each completed
    // block and the merging of equal weights are all correct.
    // Rationale: docs/engineering_notes.md#reduction-test-pairwise-tree-shape
    std::mt19937_64 rng(20260903);
    std::uniform_real_distribution<Real> mag(-1.0, 1.0);
    std::uniform_int_distribution<int> exp10(-12, 3);
    std::vector<Real> v;
    v.reserve(8 * kPairwiseBlock);
    for (std::size_t i = 0; i < 8 * kPairwiseBlock; ++i) {
        v.push_back(mag(rng) * std::pow(10.0, exp10(rng)));
    }

    CHECK(same_bits(reduce::sum(v, Reduction::Pairwise), pairwise_reference(v)));
    // Eight blocks, so the tree is three rounds deep: a mutation that only
    // reordered the last round would escape a two-block case.
    REQUIRE(v.size() / kPairwiseBlock == 8);
    // The shape is observable on this input: folding the eight block sums left to
    // right gives a different double, so matching the reference constrains the tree
    // and not only the block contents.
    CHECK(pairwise_reference(v) != blockwise_fold_reference(v));
}

TEST_CASE("reduce: the Pairwise block boundary falls after kPairwiseBlock values") {
    // One block is summed left to right with no merge, so a full block equals the
    // naive fold. One value more starts a second block, whose total is added to the
    // first block's total rather than to a running sum that loses it.
    std::vector<Real> exactly_one_block{1.0};
    exactly_one_block.insert(exactly_one_block.end(), kPairwiseBlock - 1, 1e-17);
    REQUIRE(exactly_one_block.size() == kPairwiseBlock);
    CHECK(same_bits(reduce::sum(exactly_one_block, Reduction::Pairwise),
                    naive_reference(exactly_one_block)));
    CHECK(same_bits(reduce::sum(exactly_one_block, Reduction::Pairwise), 1.0));

    // Two full blocks: the second block's 128 addends accumulate among
    // themselves, where they are the same size as each other and survive, and
    // only their total meets the 1.0.
    std::vector<Real> two_blocks = exactly_one_block;
    two_blocks.insert(two_blocks.end(), kPairwiseBlock, 1e-17);
    const Real second_block = [] {
        Real s = 0.0;
        for (std::size_t i = 0; i < kPairwiseBlock; ++i) {
            s += 1e-17;
        }
        return s;
    }();
    CHECK(same_bits(reduce::sum(two_blocks, Reduction::Pairwise), 1.0 + second_block));
    CHECK(reduce::sum(two_blocks, Reduction::Pairwise) > 1.0);
    CHECK(same_bits(reduce::sum(two_blocks, Reduction::Naive), 1.0));
}

TEST_CASE("reduce: the Exact half-even correction fires on sign, in both directions") {
    // Each input is exactly halfway between two doubles once the leading partial is
    // collapsed (1.0 is half an ulp at 1e16), so the sign of the remainder, which the
    // condition reads, decides the direction. Expected values are those of `math.fsum`.
    // Rationale: docs/engineering_notes.md#reduction-test-exact-half-even-sign
    const std::vector<Real> above{1e16, 1.0, 1e-16};
    const std::vector<Real> below{1e16, 1.0, -1e-16};

    CHECK(same_bits(reduce::sum(above, Reduction::Exact), 1.0000000000000002e16));
    CHECK(same_bits(reduce::sum(below, Reduction::Exact), 1e16));

    // The mirror, which exercises the other arm of the condition: the first pair
    // reads `lo > 0 && partials > 0`, this one `lo < 0 && partials < 0`.
    const std::vector<Real> below_negative{-1e16, -1.0, -1e-16};
    const std::vector<Real> above_negative{-1e16, -1.0, 1e-16};

    CHECK(same_bits(reduce::sum(below_negative, Reduction::Exact), -1.0000000000000002e16));
    CHECK(same_bits(reduce::sum(above_negative, Reduction::Exact), -1e16));

    // An exact tie with nothing left over: round-half-even takes it down, and the
    // correction does not run. The descent consumes every partial, so the guard on
    // the remaining count keeps the correction from reading index -1.
    const std::vector<Real> exact_tie{1e16, 1.0};
    CHECK(same_bits(reduce::sum(exact_tie, Reduction::Exact), 1e16));

    // Naive arithmetic loses the 1.0 in all five, so every distinction above comes
    // from the correction.
    for (const auto& v : {above, below, below_negative, above_negative, exact_tie}) {
        CHECK(std::abs(reduce::sum(v, Reduction::Naive)) == 1e16);
    }
}

TEST_CASE("reduce: Pairwise merges equal weights rather than folding the blocks") {
    // These eight block sums give three doubles one ulp apart under the balanced tree,
    // a right-to-left fold and a left-to-right fold. Each block is 127 zeros then the
    // value, so an off-by-one block boundary moves every value into its neighbour.
    // Rationale: docs/engineering_notes.md#reduction-test-pairwise-merge-rule
    const std::vector<Real> block_values{
        -16.15390085620416,      -76.69840247205887, 9.80104494286562e-05,
        -0.4604100456188034,     9.958074373731778e-05, -57.85795571311212,
        0.0009916714408155813,   -0.009157843591376887,
    };
    REQUIRE(block_values.size() == 8);

    std::vector<Real> v;
    v.reserve(block_values.size() * kPairwiseBlock);
    for (const Real value : block_values) {
        v.insert(v.end(), kPairwiseBlock - 1, 0.0);
        v.push_back(value);
    }

    // The balanced tree, stated as a literal so the test does not agree with
    // the implementation merely by recomputing it the same way.
    CHECK(same_bits(reduce::sum(v, Reduction::Pairwise), -151.17863766795136));
    CHECK(same_bits(pairwise_reference(v), -151.17863766795136));

    // The two associations a mutated merge produces, one ulp either side. Named
    // so a failure says which shape the accumulator fell into.
    CHECK(reduce::sum(v, Reduction::Pairwise) != -151.17863766795134);  // right to left
    CHECK(same_bits(blockwise_fold_reference(v), -151.1786376679513));  // left to right
}

TEST_CASE("reduce: Exact builds its expansion and collapses it one ulp exactly") {
    // Six expansions whose collapse is decided in the last bit. Each agrees with
    // `math.fsum` and separates `Exact` from one misreading of the algorithm, not
    // from the naive fold.
    // Rationale: docs/engineering_notes.md#reduction-test-exact-last-bit-cases

    // The expansion itself: writing the surviving remainders from slot 1
    // instead of slot 0 leaves a stale partial behind, and forming the
    // remainder as `y + (hi - x)` instead of `y - (hi - x)` inverts it.
    const std::vector<Real> expansion{-0.5, 3.0, 0.1, 1e16, 1e8, 0.5};
    CHECK(same_bits(reduce::sum(expansion, Reduction::Exact), 1.0000000100000004e16));

    // The descent's own remainder, which is a separate two-sum from the one in
    // `add`: `lo = y - (hi - x)`. Formed as a sum instead of a difference it
    // stops being the discarded part, and both the point the descent stops at
    // and the correction that follows read it.
    const std::vector<Real> descent{-0.3, 3.0, 0.1, -1e8, 1e8};
    CHECK(same_bits(reduce::sum(descent, Reduction::Exact), 2.8));

    // The correction's condition, in four readings. `lo` and the next partial
    // must BOTH be negative, or both positive; either one alone is not enough,
    // and the partial consulted must be the next one down rather than the one
    // just consumed or the one past it.
    const std::vector<Real> both_or_neither{1e-16, -2.0, -0.5, 1.0};
    CHECK(same_bits(reduce::sum(both_or_neither, Reduction::Exact), -1.5));

    const std::vector<Real> reads_the_next_partial{-1e-16, 3.0, -1e-16, 1e16};
    CHECK(same_bits(reduce::sum(reads_the_next_partial, Reduction::Exact),
                    1.0000000000000002e16));

    const std::vector<Real> not_the_one_just_consumed{0.5, 1e16, -1.0, -1.0, 1e16};
    CHECK(same_bits(reduce::sum(not_the_one_just_consumed, Reduction::Exact), 2e16));

    const std::vector<Real> not_the_one_past_it{1.0000000000000002e16, -1e8, 1e16, -0.5, 1e16};
    CHECK(same_bits(reduce::sum(not_the_one_past_it, Reduction::Exact), 2.99999999e16));

    // Order independence holds for each case. `Exact` is the ground truth the other
    // policies are measured against in section 7.1, so a one-ulp error here affects
    // every such measurement.
    for (const auto& original : {expansion, descent, both_or_neither,
                                 reads_the_next_partial, not_the_one_just_consumed,
                                 not_the_one_past_it}) {
        std::vector<Real> reversed(original.rbegin(), original.rend());
        CHECK(same_bits(reduce::sum(original, Reduction::Exact),
                        reduce::sum(reversed, Reduction::Exact)));
    }
}
