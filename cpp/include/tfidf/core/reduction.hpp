// Reduction policies: how a sum of binary64 values is accumulated.
//
// Summation order is part of this project's specification, not an
// implementation detail. `(a + b) + c` and `a + (b + c)` are different
// computations in binary64, README section 2 writes its sums without
// bracketing, and so the normative reading is a plain left-to-right fold.
// `Naive` implements exactly that, and every published result uses it.
//
// The other policies are *instruments*. The spread between them is a direct
// measurement of the floating-point noise floor, from which the near-tie
// tolerance tau of section 7.1 is derived rather than asserted. `Exact` gives
// the correctly-rounded sum and so turns that spread into an absolute error.
//
// Design note -- why policy classes and not std::function or virtual calls:
// each kernel is a template on the accumulator type and is instantiated once
// per policy, so the compiler sees a concrete accumulator it can keep in
// registers and the inner loop contains no dispatch at all. Selection happens
// once per call, at the boundary, via a switch over `Reduction`.
#pragma once

#include <tfidf/core/types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tfidf::reduce {

// -----------------------------------------------------------------------------
// Policies
// -----------------------------------------------------------------------------

/// Plain left-to-right accumulation. The normative policy.
///
/// Deliberately the least accurate of the four: silently improving on the
/// paper's arithmetic would mean publishing numbers the stated mathematics does
/// not produce.
struct Naive {
    Real sum = 0.0;

    constexpr void add(Real x) noexcept { sum += x; }
    [[nodiscard]] constexpr Real value() const noexcept { return sum; }
};

/// Kahan-Babuska-Neumaier compensated summation.
///
/// Tracks the rounding error lost at each step in a second accumulator and adds
/// it back once at the end. Unlike plain Kahan this variant is also correct when
/// the running total is smaller in magnitude than the addend, which is the
/// common case when a large term arrives late.
struct Neumaier {
    Real sum = 0.0;
    Real compensation = 0.0;

    constexpr void add(Real x) noexcept {
        const Real t = sum + x;
        // Whichever operand is larger keeps its bits; the other one is the one
        // whose low-order bits were discarded, so that is where we recover them.
        if (std::abs(sum) >= std::abs(x)) {
            compensation += (sum - t) + x;
        } else {
            compensation += (x - t) + sum;
        }
        sum = t;
    }

    [[nodiscard]] constexpr Real value() const noexcept { return sum + compensation; }
};

/// Recursive pairwise summation; error grows as O(log n) rather than O(n).
///
/// Implemented iteratively with a small stack of partial sums, one per level,
/// so it allocates nothing and matches the recursive formulation exactly. The
/// base case is `kPairwiseBlock`, matching numpy so the two agree.
struct Pairwise {
    // 64 levels is enough for 2^64 * kPairwiseBlock elements.
    std::array<Real, 64> partials{};
    std::array<std::size_t, 64> counts{};
    std::size_t levels = 0;
    Real block = 0.0;
    std::size_t in_block = 0;

    constexpr void add(Real x) noexcept {
        block += x;
        if (++in_block == kPairwiseBlock) {
            push(block, 1);
            block = 0.0;
            in_block = 0;
        }
    }

    [[nodiscard]] constexpr Real value() const noexcept {
        // Fold the completed levels from the deepest (largest) upward, which is
        // the order the recursive formulation produces.
        Real total = 0.0;
        for (std::size_t i = levels; i-- > 0;) {
            total += partials[i];
        }
        return total + block;
    }

  private:
    constexpr void push(Real v, std::size_t weight) noexcept {
        // Merge equal-weight partials, mirroring the recursive split.
        while (levels > 0 && counts[levels - 1] == weight) {
            --levels;
            v = partials[levels] + v;
            weight *= 2;
        }
        if (levels < partials.size()) {
            partials[levels] = v;
            counts[levels] = weight;
            ++levels;
        }
    }
};

/// Correctly-rounded summation via Shewchuk's expansion algorithm.
///
/// Maintains a set of non-overlapping partial sums whose exact total equals the
/// exact sum of the inputs, then rounds once. This is the same algorithm behind
/// Python's `math.fsum`, so the two agree bit-for-bit -- which is what lets the
/// noise-floor study use a common ground truth across both languages.
struct Exact {
    std::vector<Real> partials;

    void add(Real x) noexcept {
        std::size_t out = 0;
        for (const Real y_const : partials) {
            Real y = y_const;
            if (std::abs(x) < std::abs(y)) {
                std::swap(x, y);
            }
            // Two-sum: hi is the rounded sum, lo the exact discarded remainder.
            const Real hi = x + y;
            const Real lo = y - (hi - x);
            if (lo != 0.0) {
                partials[out++] = lo;
            }
            x = hi;
        }
        partials.resize(out);
        partials.push_back(x);
    }

    /// Round the expansion to a single correctly-rounded binary64.
    ///
    /// A naive fold over the partials is *not* sufficient, and getting this
    /// wrong is subtle: the expansion is exact, but collapsing it can still be
    /// off by one ulp, which makes the result depend on the order the inputs
    /// arrived in. Since `Exact` is the ground truth against which the other
    /// policies' error is measured -- and must agree with Python's `math.fsum`
    /// bit-for-bit so both languages share that ground truth -- this reproduces
    /// CPython's algorithm exactly, including its half-even correction.
    ///
    /// Sum from the largest partial downward until the running total becomes
    /// inexact, then apply the correction that makes the result independent of
    /// input order (CPython's comment: "so that sum([1e-16, 1, 1e16]) will
    /// round up the last digit to two instead of down to zero").
    [[nodiscard]] Real value() const noexcept {
        std::size_t n = partials.size();
        if (n == 0) {
            return 0.0;
        }
        Real hi = partials[--n];
        Real lo = 0.0;
        while (n > 0) {
            const Real x = hi;
            const Real y = partials[--n];
            hi = x + y;
            const Real yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) {
                break;
            }
        }
        if (n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) ||
                      (lo > 0.0 && partials[n - 1] > 0.0))) {
            const Real y = lo * 2.0;
            const Real x = hi + y;
            if (y == x - hi) {
                hi = x;
            }
        }
        return hi;
    }
};

// -----------------------------------------------------------------------------
// Generic drivers
// -----------------------------------------------------------------------------

/// Sum a contiguous range under a policy given as a template parameter.
template <class Policy>
[[nodiscard]] Real sum_with(const Real* data, std::size_t n) noexcept {
    Policy acc{};
    for (std::size_t i = 0; i < n; ++i) {
        acc.add(data[i]);
    }
    return acc.value();
}

/// Sum a contiguous range under a policy chosen at run time.
///
/// The switch runs once per call and selects a fully monomorphised kernel, so
/// the cost is O(1) per call rather than O(1) per element.
[[nodiscard]] inline Real sum(const Real* data, std::size_t n, Reduction policy) noexcept {
    switch (policy) {
        case Reduction::Naive:
            return sum_with<Naive>(data, n);
        case Reduction::Neumaier:
            return sum_with<Neumaier>(data, n);
        case Reduction::Pairwise:
            return sum_with<Pairwise>(data, n);
        case Reduction::Exact:
            return sum_with<Exact>(data, n);
    }
    return sum_with<Naive>(data, n);  // unreachable; keeps every path defined
}

[[nodiscard]] inline Real sum(const std::vector<Real>& v, Reduction policy) noexcept {
    return sum(v.data(), v.size(), policy);
}

}  // namespace tfidf::reduce
