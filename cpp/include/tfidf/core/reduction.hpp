// Reduction policies: how a sum of binary64 values is accumulated.
//
// `Naive` is the normative left-to-right fold of README section 2. The other three measure
// the floating-point noise floor from which the tolerance tau of section 7.1 is derived.
// Rationale: docs/engineering_notes.md#reduction-policies
#pragma once

#include <tfidf/core/types.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace tfidf::reduce {

// -----------------------------------------------------------------------------
// Policies
// -----------------------------------------------------------------------------

/// Plain left-to-right accumulation, the normative policy.
///
/// The least accurate of the four. A more accurate sum would publish numbers that the
/// stated mathematics does not produce.
struct Naive {
    Real sum = 0.0;

    constexpr void add(Real x) noexcept { sum += x; }
    [[nodiscard]] constexpr Real value() const noexcept { return sum; }
};

/// Absolute value usable in a constant expression on every supported compiler.
///
/// `std::abs` is not `constexpr` in MSVC's STL. Returns -0.0 for -0.0, which compares
/// equal to +0.0 at the one call site.
/// Rationale: docs/engineering_notes.md#reduction-constexpr-magnitude
[[nodiscard]] constexpr Real magnitude(Real x) noexcept {
    return x < Real(0) ? -x : x;
}

/// Kahan-Babuska-Neumaier compensated summation.
///
/// Tracks the rounding error of each step in a second accumulator and adds it back at the end.
/// Unlike plain Kahan, it stays correct when an addend exceeds the running total in magnitude.
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

/// Pairwise summation; error grows as O(log n) rather than O(n).
///
/// Streaming, with a stack of one partial sum per level, so it allocates nothing. Bit-identical
/// to the repository's Python reference, not to `np.sum`, despite the shared block size.
/// Rationale: docs/engineering_notes.md#pairwise-reduction-order
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
        // Fold the completed levels from the top of the stack (smallest) down to the largest,
        // then the trailing block, in the order the Python reference uses.
        Real total = 0.0;
        for (std::size_t i = levels; i-- > 0;) {
            total += partials[i];
        }
        return total + block;
    }

  private:
    constexpr void push(Real v, std::size_t weight) noexcept {
        // Merge equal-weight partials, as a binary counter carries. The weight is only
        // compared for equality, so a growth factor of 3 builds the same tree as 2 does.
        // Rationale: docs/engineering_notes.md#pairwise-weight-equivalence
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
/// Same algorithm as Python's `math.fsum`, bit-for-bit on finite inputs, which it requires: an
/// infinite addend or intermediate overflow can give NaN where `math.fsum` raises or returns inf.
/// Rationale: docs/engineering_notes.md#exact-reduction-non-finite
struct Exact {
    std::vector<Real> partials;

    void add(Real x) noexcept {  // noexcept: std::bad_alloc from push_back calls std::terminate
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
        // Append the running total only when nonzero, as CPython's `math_fsum` does, so that a
        // sum of negative zeros gives +0.0.
        // Rationale: docs/engineering_notes.md#exact-reduction-signed-zero
        if (x != 0.0) {
            partials.push_back(x);
        }
    }

    /// Round the expansion to a single correctly-rounded binary64.
    ///
    /// Reproduces CPython's `math_fsum` rounding, including its half-even correction, so
    /// the result is independent of input order and matches `math.fsum` bit-for-bit.
    /// Rationale: docs/engineering_notes.md#exact-reduction-rounding
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
