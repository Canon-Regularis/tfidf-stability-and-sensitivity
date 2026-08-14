// Reduction policies: how a sum of binary64 values is accumulated.
//
// Summation order is part of this project's specification. `(a + b) + c` and
// `a + (b + c)` are different computations in binary64, and README section 2
// writes its sums without bracketing, so the normative reading is a plain
// left-to-right fold. `Naive` is that fold, and every published result uses it.
//
// The other three are instruments. Their spread measures the floating-point
// noise floor, from which the near-tie tolerance tau of section 7.1 is derived
// rather than asserted; `Exact` is correctly rounded, so the spread against it
// is an absolute error.
//
// Policy classes rather than std::function or virtual calls: each kernel is a
// template on the accumulator type, instantiated once per policy, so the
// accumulator stays in registers and the inner loop holds no dispatch.
// Selection happens once per call, at the boundary, via a switch over
// `Reduction`.
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

/// Plain left-to-right accumulation. The normative policy, and the least
/// accurate of the four: improving on the paper's arithmetic would publish
/// numbers its stated mathematics does not produce.
struct Naive {
    Real sum = 0.0;

    constexpr void add(Real x) noexcept { sum += x; }
    [[nodiscard]] constexpr Real value() const noexcept { return sum; }
};

/// ``|x|``, usable in a constant expression on every supported compiler.
///
/// `std::abs` is `constexpr` in libstdc++ (C++23's P0533) and not in MSVC's
/// STL, which rejects it with C3615, "cannot result in a constant expression".
/// CI hit the split: the C++ tests are built by a hand-rolled CMake invocation
/// that picks MinGW GCC, while scikit-build-core builds the Python extension
/// with MSVC, so the same header compiled in one job and broke the other, on
/// Windows only.
///
/// The sole behavioural difference is -0.0, returned here where `std::abs`
/// gives +0.0. The one call site is a magnitude comparison, where the two
/// compare equal. NaN and the infinities are unaffected.
[[nodiscard]] constexpr Real magnitude(Real x) noexcept {
    return x < Real(0) ? -x : x;
}

/// Kahan-Babuska-Neumaier compensated summation.
///
/// Tracks the rounding error lost at each step in a second accumulator and adds
/// it back once at the end. Unlike plain Kahan, correct when the running total
/// is smaller in magnitude than the addend, the common case when a large term
/// arrives late.
struct Neumaier {
    Real sum = 0.0;
    Real compensation = 0.0;

    constexpr void add(Real x) noexcept {
        const Real t = sum + x;
        // The larger operand keeps its bits, so the low-order bits that were
        // discarded belong to the smaller one; recover them from there.
        if (magnitude(sum) >= magnitude(x)) {
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
/// Iterative, with a small stack of partial sums, one per level, so it
/// allocates nothing and matches the recursive formulation.
///
/// The base case is `kPairwiseBlock`, numpy's block size, which does not make
/// the two agree: numpy unrolls its base case into eight independent
/// accumulators, so its order diverges from a straight fold well before the
/// block boundary. Against `np.sum` over 262 sizes the two differ at 208 of
/// them, first at n = 8.
///
/// The contract is with this repository's Python reference: bit-identical
/// across 305 sizes from n = 1 to 10,000, including the 2^k and 2^k+1
/// boundaries where a block-size or recursion-cutoff mismatch would first show.
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
/// Keeps non-overlapping partial sums whose exact total is the exact sum of the
/// inputs, then rounds once. Same algorithm as Python's `math.fsum`, so the two
/// agree bit-for-bit on finite inputs, which gives the noise-floor study one
/// ground truth across both languages.
///
/// Finite inputs only. CPython's `math_fsum` tracks infinities and intermediate
/// overflow separately and raises: `fsum([inf, -inf])` is a ValueError and
/// `fsum([1e308, 1e308])` an OverflowError. This has no such machinery and
/// returns NaN for those two and for `[inf, 1.0]`, where `math.fsum` returns
/// inf. The other three policies agree with the reference on those inputs, so
/// the gap is specific to `Exact`.
///
/// Unreachable from the published pipeline (a weight is a non-negative tf times
/// `idf = ln((1+N)/(1+df)) >= 0`, so no infinity enters a reduction), reachable
/// through the `reduce_sum` binding.
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
        // Conditional, as CPython's `math_fsum` is: it appends the running total
        // only `if (x != 0.0)`. Pushing unconditionally preserves the value of
        // the expansion but can flip its sign: over nothing but negative zeros
        // `x` is -0.0, so `value()` returned -0.0 where `math.fsum` returns
        // +0.0. Measured through the shipped extension, reduce_sum([-0.0],
        // exact) gave bits 8000000000000000 against the reference's
        // 0000000000000000. Since -0.0 == 0.0, no tolerance or equality check
        // sees it.
        if (x != 0.0) {
            partials.push_back(x);
        }
    }

    /// Round the expansion to a single correctly-rounded binary64.
    ///
    /// The expansion is exact, but a plain fold over the partials can still
    /// collapse it one ulp off, which makes the result depend on the order the
    /// inputs arrived in. `Exact` is the ground truth the other policies' error
    /// is measured against and must match `math.fsum` bit-for-bit, so this
    /// reproduces CPython's algorithm including its half-even correction.
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
/// The switch runs once per call and selects a monomorphised kernel, so
/// dispatch costs O(1) per call rather than O(1) per element.
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
