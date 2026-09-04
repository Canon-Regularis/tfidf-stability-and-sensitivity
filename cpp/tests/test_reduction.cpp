// Reduction policies.
//
// The policies must differ in the right way and agree in the right way. If they
// never differed, the noise-floor measurement that derives tau (section 7.1)
// would be measuring nothing; if they disagreed on exactly representable
// inputs, one of them would be wrong.
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

/// Bit-level equality, which is what "bit-exact" means here. `==` would
/// conflate -0.0 with 0.0 and call two NaNs unequal.
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

/// The recursive split `Pairwise` names, written out independently: runs of
/// `kPairwiseBlock` summed left to right, then combined two at a time until one
/// remains. For a power-of-two block count this is exactly the documented tree,
/// so it is an oracle for the accumulator rather than a copy of it.
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

/// The same block sums, folded left to right instead of combined pairwise. Only
/// useful to show that on a given input the tree shape is observable at all.
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
    // `add` subtracts the smaller operand from the larger, where the
    // subtraction is exact; the other order loses the bits it exists to
    // recover. Mixed magnitudes with cancellation are what expose the choice.
    //
    // Every other case in this file agrees whatever `magnitude` returns. Three
    // mutations of it survive the suite without this: dropping the negation, so
    // `magnitude` is the identity; comparing against 1.0 rather than 0.0; and
    // the -0.0 case the function's own comment already argues is unobservable.
    // The first two are not: with the identity, this sum returns -2.0.
    const std::vector<Real> v{0.5, -1e16, -0.5, -2.3, 1e16};

    CHECK(same_bits(reduce::sum(v, Reduction::Neumaier), -2.3));
    // Exact is correctly rounded, so it is the ground truth Neumaier matches.
    CHECK(same_bits(reduce::sum(v, Reduction::Exact), -2.3));
    // And the uncompensated policies genuinely lose the 0.3, so the case is
    // discriminating rather than one every policy happens to get right.
    CHECK(same_bits(reduce::sum(v, Reduction::Naive), -2.0));
    CHECK(same_bits(reduce::sum(v, Reduction::Pairwise), -2.0));

    // A second case, one ulp wide, for the threshold itself. The case above
    // uses operands far apart in magnitude, which a comparison against 1.0
    // still orders correctly; these four straddle 1.0, where it does not.
    const std::vector<Real> near_one{-0.1, 0.5, 0.25, -0.5};

    CHECK(same_bits(reduce::sum(near_one, Reduction::Neumaier), 0.15));
    CHECK(same_bits(reduce::sum(near_one, Reduction::Exact), 0.15));
    CHECK(same_bits(reduce::sum(near_one, Reduction::Naive), 0.15000000000000002));
}

TEST_CASE("reduce: Exact recovers a result naive arithmetic destroys") {
    // The classic ill-conditioned sum. Left to right:
    //   0 + 1e100      -> 1e100
    //   1e100 + 1.0    -> 1e100   (the first 1.0 is below half an ulp: lost)
    //   1e100 - 1e100  -> 0
    //   0 + 1.0        -> 1.0     (the second 1.0 survives, having arrived
    //                              after the cancellation)
    // so naive returns 1.0: one of the two units is lost. The exact sum is 2.0.
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
    // Counterpart of the previous test. If this became order independent, the
    // summation-order sensitivity the paper studies would have disappeared.
    std::vector<Real> v{1.0};
    v.insert(v.end(), 100, 1e-17);
    const Real small_first = [&] {
        std::vector<Real> w(v.rbegin(), v.rend());
        return reduce::sum(w, Reduction::Naive);
    }();
    CHECK(reduce::sum(v, Reduction::Naive) != small_first);
}

TEST_CASE("reduce: Pairwise beats Naive on a long uniform sum") {
    // 1e6 identical values: pairwise error grows as O(log n), naive as O(n).
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
    // `Pairwise` is a streaming accumulator with a partials stack, not the
    // recursive split its docstring names. The two agree only if the block
    // size, the weight each completed block is pushed with, and the doubling
    // that merges equal weights are all what they claim to be.
    //
    // The existing long-uniform case checks only that pairwise error does not
    // exceed naive error, which any tree satisfies. Five mutations of the
    // accumulator survive it: shifting the block boundary to 127, pushing a
    // completed block with weight 0, tripling the weight instead of doubling,
    // and two loop bounds. Each of them reshapes the tree, so pinning the
    // shape is what refuses them.
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
    // And the shape is observable on this input: folding the same eight block
    // sums left to right instead of combining them pairwise gives a different
    // double, so matching the reference constrains the tree rather than just
    // the block contents.
    CHECK(pairwise_reference(v) != blockwise_fold_reference(v));
}

TEST_CASE("reduce: the Pairwise block boundary falls after kPairwiseBlock values") {
    // The boundary itself, stated without the tree. One block is summed left to
    // right and nothing is merged, so a full block must equal the naive fold;
    // one value more must not, because that value starts a second block and is
    // added to the first block's total rather than to a running sum that has
    // already lost it.
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
    // The documented case above only shows the correction firing. It cannot
    // separate the condition from the correction, because the inner
    // `y == x - hi` test already refuses most spurious applications: a
    // condition mutated to fire always still gives the right answer on an input
    // that wanted it to fire.
    //
    // These four do separate it. Every one is exactly halfway between two
    // doubles once the leading partial is collapsed -- 1.0 is half an ulp at
    // 1e16 -- so the direction is decided entirely by the sign of what is left
    // over, which is what the condition reads. Values are `math.fsum`'s, the
    // reference this policy exists to match.
    const std::vector<Real> above{1e16, 1.0, 1e-16};
    const std::vector<Real> below{1e16, 1.0, -1e-16};

    CHECK(same_bits(reduce::sum(above, Reduction::Exact), 1.0000000000000002e16));
    CHECK(same_bits(reduce::sum(below, Reduction::Exact), 1e16));

    // The mirror, which exercises the other arm of the condition: the first
    // reads `lo < 0 && partials < 0`, this one `lo > 0 && partials > 0`.
    const std::vector<Real> below_negative{-1e16, -1.0, -1e-16};
    const std::vector<Real> above_negative{-1e16, -1.0, 1e-16};

    CHECK(same_bits(reduce::sum(below_negative, Reduction::Exact), -1.0000000000000002e16));
    CHECK(same_bits(reduce::sum(above_negative, Reduction::Exact), -1e16));

    // An exact tie with nothing left over: round-half-even takes it down, and
    // the correction must not run at all. Here the descent consumes every
    // partial, so the guard on the remaining count is the only thing standing
    // between the correction and an index of -1.
    const std::vector<Real> exact_tie{1e16, 1.0};
    CHECK(same_bits(reduce::sum(exact_tie, Reduction::Exact), 1e16));

    // And the case is discriminating: naive arithmetic loses the 1.0 in all
    // five, so every distinction above is the correction's doing.
    for (const auto& v : {above, below, below_negative, above_negative, exact_tie}) {
        CHECK(std::abs(reduce::sum(v, Reduction::Naive)) == 1e16);
    }
}

TEST_CASE("reduce: Pairwise merges equal weights rather than folding the blocks") {
    // The random tree case above pins the shape only as far as its data can see
    // it: eight block sums of similar magnitude add to the same double in most
    // orders, so a mutated merge rule still matches. These eight do not. They
    // are one ulp apart under the balanced tree, a right-to-left fold and a
    // left-to-right fold, so each association gives a different answer and the
    // rule that picks one is forced.
    //
    // Each block is 127 zeros followed by one value, which makes the block sum
    // exactly that value and puts every value at the END of its block: a block
    // boundary off by one then moves every value into its neighbour, which the
    // leading-value layout would hide.
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
    // Six expansions whose collapse is decided in the last bit, each found by
    // searching for an input that separates one wrong reading of the algorithm
    // from the right one. All six agree with `math.fsum`, which is the
    // reference `Exact` exists to match; none of them agrees with a version
    // that mis-orders the two-sum, starts the expansion at the wrong slot, or
    // gets the correction's condition wrong.
    //
    // The naive fold happens to land on the same double for several of these,
    // so these are not naive-versus-exact cases. What they separate is Exact
    // from a slightly wrong Exact, which is what the policy's contract needs.

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

    // Order independence holds for every one of them, which is the property the
    // correction exists to give and the reason a one-ulp slip here would not
    // stay local: `Exact` is the ground truth the other policies are measured
    // against in section 7.0.
    for (const auto& original : {expansion, descent, both_or_neither,
                                 reads_the_next_partial, not_the_one_just_consumed,
                                 not_the_one_past_it}) {
        std::vector<Real> reversed(original.rbegin(), original.rend());
        CHECK(same_bits(reduce::sum(original, Reduction::Exact),
                        reduce::sum(reversed, Reduction::Exact)));
    }
}
