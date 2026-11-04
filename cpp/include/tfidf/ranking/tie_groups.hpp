// Tie groups: the ball, single-linkage chains and complete-linkage cliques.
//
// Ball: README section 2.3.3 verbatim, O(log N), overlapping. Chains: single linkage, O(N),
// a partition. Cliques: complete linkage, O(N), overlapping. See `spec_addenda.md#g1`.
// Rationale: docs/engineering_notes.md#tie-groups-objects
#pragma once

#include <tfidf/core/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfidf::ranking {

/// Reject a tolerance that is negative or NaN.
///
/// `!(tau >= 0.0)` is used because `tau < 0.0` is false for NaN. With tau = NaN
/// the ball, the chains and the cliques disagree without raising, so all four
/// entry points call this guard.
inline void checked_tau(Real tau) {
    if (!(tau >= 0.0)) {
        throw std::invalid_argument("tau must be non-negative");
    }
}

/// A half-open `[lo, hi)` range of ranks.
///
/// Every group here is contiguous in the sorted order, so an interval describes it fully.
using Interval = std::pair<std::int32_t, std::int32_t>;

/// `{ i : |S[i] - S[j]| <= tau }`, as a rank interval.
///
/// Both binary searches evaluate the `spec_addenda.md#g9` difference, not a rounded bound
/// `fl(S[j] +/- tau)`; on non-increasing input each difference is monotone in i in binary64.
/// Rationale: docs/engineering_notes.md#tie-groups-ball-difference-search
[[nodiscard]] inline Interval tie_ball_interval(std::span<const Real> sorted_scores,
                                                std::int32_t j,
                                                Real tau) {
    const auto n = static_cast<std::int32_t>(sorted_scores.size());
    // `sorted_scores[j]` with j outside `[0, n)` is undefined behaviour. The
    // normative Python raises `IndexError` here and the binding refuses the
    // index; the same guard one level down keeps a C++ caller from reaching
    // the read.
    if (j < 0 || j >= n) {
        throw std::out_of_range("rank index out of range");
    }
    checked_tau(tau);
    const Real centre = sorted_scores[static_cast<std::size_t>(j)];

    // lo: first i in [0, j] with S[i] - centre <= tau.
    std::int32_t lo = 0;
    std::int32_t hi_search = j;
    while (lo < hi_search) {
        const std::int32_t mid = lo + (hi_search - lo) / 2;
        if (sorted_scores[static_cast<std::size_t>(mid)] - centre <= tau) {
            hi_search = mid;
        } else {
            lo = mid + 1;
        }
    }

    // hi: first i in [j, n) with centre - S[i] > tau.
    std::int32_t lo_search = j;
    std::int32_t hi = n;
    while (lo_search < hi) {
        const std::int32_t mid = lo_search + (hi - lo_search) / 2;
        if (centre - sorted_scores[static_cast<std::size_t>(mid)] > tau) {
            hi = mid;
        } else {
            lo_search = mid + 1;
        }
    }
    return {lo, hi};
}

/// The transitive closure: cut wherever an adjacent gap exceeds `tau`.
///
/// This is the closure because on a linearly ordered set any sequence of
/// "within tau" steps can be replaced by the monotone path through the
/// intervening points, along which gaps only shrink.
[[nodiscard]] inline std::vector<Interval> tie_chains(std::span<const Real> sorted_scores,
                                                      Real tau) {
    checked_tau(tau);
    std::vector<Interval> out;
    const auto n = static_cast<std::int32_t>(sorted_scores.size());
    if (n == 0) {
        return out;
    }
    std::int32_t start = 0;
    for (std::int32_t i = 1; i < n; ++i) {
        const Real gap = sorted_scores[static_cast<std::size_t>(i) - 1] -
                         sorted_scores[static_cast<std::size_t>(i)];
        if (gap > tau) {
            out.emplace_back(start, i);
            start = i;
        }
    }
    out.emplace_back(start, n);
    return out;
}

/// Maximal intervals of diameter `<= tau`.
///
/// The near-tie graph is an indifference graph, so its at most N maximal cliques are
/// contiguous; one two-pointer pass keeps `[a, R(a)]` iff `a == 0` or `R(a) > R(a-1)`.
/// Rationale: docs/engineering_notes.md#tie-groups-clique-sweep
[[nodiscard]] inline std::vector<Interval> tie_cliques(std::span<const Real> sorted_scores,
                                                       Real tau) {
    checked_tau(tau);
    std::vector<Interval> out;
    const auto n = static_cast<std::int32_t>(sorted_scores.size());
    if (n == 0) {
        return out;
    }
    std::int32_t right = 0;
    std::int32_t previous_right = -1;
    for (std::int32_t a = 0; a < n; ++a) {
        right = std::max(right, a);
        while (right + 1 < n && sorted_scores[static_cast<std::size_t>(a)] -
                                        sorted_scores[static_cast<std::size_t>(right) + 1] <=
                                    tau) {
            ++right;
        }
        if (right > previous_right) {
            out.emplace_back(a, right + 1);
            previous_right = right;
        }
    }
    return out;
}

/// `rho(tau) = |largest chain| / |largest clique|`.
///
/// `rho >= 1` for non-increasing input, since every clique then lies inside a chain.
/// The order is required and not checked; unsorted scores can give `rho < 1`.
/// Rationale: docs/engineering_notes.md#tie-groups-chain-inflation-bound
[[nodiscard]] inline Real chain_inflation_ratio(std::span<const Real> sorted_scores, Real tau) {
    checked_tau(tau);
    const auto chains = tie_chains(sorted_scores, tau);
    const auto cliques = tie_cliques(sorted_scores, tau);
    if (chains.empty() || cliques.empty()) {
        return std::numeric_limits<Real>::quiet_NaN();
    }
    std::int32_t widest_chain = 0;
    for (const auto& [lo, hi] : chains) {
        widest_chain = std::max(widest_chain, hi - lo);
    }
    std::int32_t widest_clique = 0;
    for (const auto& [lo, hi] : cliques) {
        widest_clique = std::max(widest_clique, hi - lo);
    }
    return static_cast<Real>(widest_chain) / static_cast<Real>(widest_clique);
}

}  // namespace tfidf::ranking
