// Tie groups (README section 2.3.3, `spec_addenda.md#g1`).
//
// Three distinct objects, never conflated:
//
//   tie_ball_interval  verbatim section 2.3.3   O(log N)  overlapping
//   tie_chains         single linkage           O(N)      a partition
//   tie_cliques        complete linkage         O(N)      overlapping
//
// The ball is what the paper defines; the chain is the object with a
// well-defined "group containing document i"; the clique is the set in which
// "mutually indistinguishable" is actually true. The chain-inflation ratio
// rho = |largest chain| / |largest clique| says how far the first two have
// drifted apart.
#pragma once

#include <tfidf/core/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace tfidf::ranking {

/// A half-open `[lo, hi)` range of ranks. Every group here is contiguous in the
/// sorted order, so an interval is a complete description.
using Interval = std::pair<std::int32_t, std::int32_t>;

/// `{ i : |S[i] - S[j]| <= tau }`, as a rank interval.
///
/// The obvious implementation binary-searches for `S[j] +/- tau`. That is
/// **wrong**, and wrong exactly where it matters: those bounds are themselves
/// rounded, so the predicate actually evaluated becomes `S[i] <= fl(S[j] + tau)`,
/// which differs from `spec_addenda.md#g9`'s pinned `|s_i - s_j| <= tau`
/// precisely at the boundary -- the only place tie groups are interesting.
///
/// So the search runs on the difference itself. On a non-increasing array
/// `S[i] - S[j]` is non-increasing in i and `S[j] - S[i]` is non-decreasing in
/// i, so both bounds remain binary-searchable while evaluating exactly the
/// subtraction G9 specifies. The monotonicity holds in binary64, not merely in
/// the reals, because IEEE subtraction is monotone.
[[nodiscard]] inline Interval tie_ball_interval(std::span<const Real> sorted_scores,
                                                std::int32_t j,
                                                Real tau) noexcept {
    const auto n = static_cast<std::int32_t>(sorted_scores.size());
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
/// This *is* the closure because on a linearly ordered set any sequence of
/// "within tau" steps can be replaced by the monotone path through the
/// intervening points, along which gaps only shrink.
[[nodiscard]] inline std::vector<Interval> tie_chains(std::span<const Real> sorted_scores,
                                                      Real tau) {
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
/// The O(N) sweep is *complete*, not merely cheap: the near-tie graph is an
/// indifference graph, so every maximal clique is a contiguous interval of the
/// sorted order, and there are at most N of them. `R(a)`, the largest b with
/// `S[a] - S[b] <= tau`, is non-decreasing, so one two-pointer pass finds all
/// of them; `[a, R(a)]` is maximal exactly when `a == 0` or `R(a) > R(a-1)`.
[[nodiscard]] inline std::vector<Interval> tie_cliques(std::span<const Real> sorted_scores,
                                                       Real tau) {
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

/// `rho(tau) = |largest chain| / |largest clique|`. Always `>= 1`, because a
/// clique's adjacent gaps are all `<= tau` and so it lies inside a chain.
[[nodiscard]] inline Real chain_inflation_ratio(std::span<const Real> sorted_scores, Real tau) {
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
