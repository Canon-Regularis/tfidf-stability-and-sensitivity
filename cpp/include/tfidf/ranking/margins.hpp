// Score-separation margins (README sections 2.3.2 and 4.4).
//
//   m_k        = score(r_k) - score(r_{k+1})
//   m_min^top  = min over 1 <= j < k of (score(r_j) - score(r_{j+1}))
//   eps_k^flip = m_k / 2
//
// Everything here takes a sorted score array and never a ranking. So m_k depends
// only on the score multiset and is identical under every tie-break operator,
// which is what makes research questions A1 (margins) and A2 (tie-breaking)
// independent. Structural, given the dependency direction, rather than asserted.
#pragma once

#include <tfidf/core/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace tfidf::ranking {

/// A margin, carrying the validity flag `spec_addenda.md#g3` requires.
///
/// An undefined margin is NaN plus `defined == false`, never coerced: 0 would
/// read as an exact tie and infinity as perfect stability, and either corrupts
/// the distribution.
struct Margin {
    std::int32_t k = 0;
    std::int32_t k_effective = 0;
    Real value = std::numeric_limits<Real>::quiet_NaN();
    bool defined = false;

    /// eps_k^flip. Exact: dividing by two only shifts the exponent.
    [[nodiscard]] Real flip_radius() const noexcept { return value / 2.0; }

    /// A defined margin of exactly zero: top-k membership then rests entirely
    /// on the tie-break.
    [[nodiscard]] bool is_exact_tie() const noexcept { return defined && value == 0.0; }
};

/// Consecutive differences: `S[j] - S[j+1]`. Length `N - 1`.
[[nodiscard]] inline std::vector<Real> adjacent_gaps(std::span<const Real> sorted_scores) {
    std::vector<Real> gaps;
    if (sorted_scores.size() < 2) {
        return gaps;
    }
    gaps.reserve(sorted_scores.size() - 1);
    for (std::size_t i = 1; i < sorted_scores.size(); ++i) {
        gaps.push_back(sorted_scores[i - 1] - sorted_scores[i]);
    }
    return gaps;
}

/// `m_k`, governing top-k membership. Undefined when `k >= N`.
[[nodiscard]] inline Margin boundary_margin(std::span<const Real> sorted_scores,
                                            std::int32_t k) noexcept {
    const auto n = static_cast<std::int32_t>(sorted_scores.size());
    Margin m;
    m.k = k;
    m.k_effective = std::min(k, n);
    if (k <= 0 || m.k_effective >= n) {
        return m;  // undefined: r_{k+1} does not exist
    }
    const auto i = static_cast<std::size_t>(m.k_effective);
    m.value = sorted_scores[i - 1] - sorted_scores[i];
    m.defined = true;
    return m;
}

/// `m_min^top`, governing the ordering *within* the top-k.
///
/// Undefined at `k = 1`: the minimum is over an empty set. G3 does not cover
/// that case; addendum G16 adopts NaN, where +inf would claim "no constraint".
[[nodiscard]] inline Margin min_adjacent_margin_top(std::span<const Real> sorted_scores,
                                                    std::int32_t k) noexcept {
    const auto n = static_cast<std::int32_t>(sorted_scores.size());
    Margin m;
    m.k = k;
    m.k_effective = std::min(k, n);
    if (k <= 0 || m.k_effective < 2 || n < 2) {
        return m;
    }
    Real best = std::numeric_limits<Real>::infinity();
    for (std::int32_t j = 0; j + 1 < m.k_effective; ++j) {
        const auto i = static_cast<std::size_t>(j);
        best = std::min(best, sorted_scores[i] - sorted_scores[i + 1]);
    }
    m.value = best;
    m.defined = true;
    return m;
}

}  // namespace tfidf::ranking
