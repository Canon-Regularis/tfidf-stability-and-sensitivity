// Ordering distances between rankings: Kendall tau and the FKS `K^(p)`.
//
// README sections 4.5 and 7.3, `spec_addenda.md#g2`. `K^(p)` is a near-metric (G2).
// Mirrors the normative `ranking/distances.py` operation by operation, bit-exact.
// Rationale: docs/engineering_notes.md#ordering-distances-scope-and-bit-exactness
#pragma once

#include <tfidf/core/types.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tfidf::ranking {

/// The FKS penalty for a pair one list ranks and the other does not witness.
///
/// `p = 1/2` is unbiased; `p = 0` biases distances down and `p = 1` biases them up.
/// Rationale: docs/engineering_notes.md#fks-penalty-one-half
inline constexpr Real kFksPenalty = 0.5;

namespace detail {

/// Merge-sort inversion count over `work[lo, hi)`, sorting the range in place.
inline std::int64_t inversion_sort_count(std::vector<std::int32_t>& work,
                                         std::vector<std::int32_t>& buffer,
                                         std::size_t lo,
                                         std::size_t hi) {
    if (hi - lo <= 1) {
        return 0;
    }
    const std::size_t mid = (lo + hi) / 2;
    std::int64_t total = inversion_sort_count(work, buffer, lo, mid) +
                         inversion_sort_count(work, buffer, mid, hi);
    std::size_t i = lo;
    std::size_t j = mid;
    std::size_t out = lo;
    while (i < mid && j < hi) {
        // `<=` rather than `<`: equal elements are no inversion, which makes the
        // count agree with "pairs i < j with seq[i] > seq[j]" on sequences that
        // repeat, and rank vectors do repeat.
        if (work[i] <= work[j]) {
            buffer[out] = work[i];
            ++i;
        } else {
            total += static_cast<std::int64_t>(mid - i);  // work[i..mid) all exceed work[j]
            buffer[out] = work[j];
            ++j;
        }
        ++out;
    }
    while (i < mid) {
        buffer[out++] = work[i++];
    }
    while (j < hi) {
        buffer[out++] = work[j++];
    }
    const auto first = static_cast<std::ptrdiff_t>(lo);
    const auto last = static_cast<std::ptrdiff_t>(hi);
    std::copy(buffer.begin() + first, buffer.begin() + last, work.begin() + first);
    return total;
}

/// `item -> index`, last occurrence winning.
///
/// Last-wins matches the reference's dict comprehension on a repeated item.
/// Rationale: docs/engineering_notes.md#distances-positions-last-wins
[[nodiscard]] inline std::unordered_map<DocId, std::int32_t> positions(
    std::span<const DocId> items) {
    std::unordered_map<DocId, std::int32_t> pos;
    pos.reserve(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        pos[items[i]] = static_cast<std::int32_t>(i);
    }
    return pos;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Same-set Kendall tau
// ---------------------------------------------------------------------------

/// Pairs `i < j` with `sequence[i] > sequence[j]`, by merge sort in O(n log n).
///
/// O(n^2) would do for a top-k list of 50, but tie groups are not bounded by k:
/// on short-text corpora the zero-score block alone can reach a large fraction
/// of the corpus. `kendall_fks` sees no such input and stays quadratic.
[[nodiscard]] inline std::int64_t inversion_count(std::span<const std::int32_t> sequence) {
    std::vector<std::int32_t> work(sequence.begin(), sequence.end());
    std::vector<std::int32_t> buffer(work.size(), 0);
    return detail::inversion_sort_count(work, buffer, 0, work.size());
}

/// Normalised Kendall tau between two orderings of the *same* set, best first.
///
/// Throws `std::invalid_argument` when the two rank different sets. That input
/// is the signal `kendall_fks` is the function wanted; returning a number there
/// would be a wrong answer rather than a missing one.
[[nodiscard]] inline Real kendall_tau_distance(std::span<const DocId> a,
                                               std::span<const DocId> b) {
    // Sorted copies, not sets: a set ignores multiplicity, so {1, 1, 2} against
    // {1, 2, 2} would pass and return 0.0. The reference compares Counters.
    // Rationale: docs/engineering_notes.md#kendall-tau-multiset-check
    std::vector<DocId> sa(a.begin(), a.end());
    std::vector<DocId> sb(b.begin(), b.end());
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    if (a.size() != b.size() || sa != sb) {
        throw std::invalid_argument(
            "kendall_tau_distance requires two orderings of the same set, with the "
            "same multiplicities; for top-k lists that may differ in membership use "
            "kendall_fks (spec_addenda G2)");
    }
    const std::size_t n = a.size();
    if (n < 2) {
        return 0.0;  // no pair exists to disagree about
    }

    const auto pos = detail::positions(b);
    std::vector<std::int32_t> mapped;
    mapped.reserve(n);
    for (const DocId item : a) {
        mapped.push_back(pos.at(item));
    }

    // `n(n-1)/2` is exact in int64 and the product is even, so the conversion is
    // the only rounding, matching the single rounding the reference incurs when
    // it divides an exact Python integer by two.
    const auto count = static_cast<std::int64_t>(n);
    const auto pairs = count * (count - 1) / 2;
    return static_cast<Real>(inversion_count(mapped)) / static_cast<Real>(pairs);
}

// ---------------------------------------------------------------------------
// Fagin-Kumar-Sivakumar generalised Kendall distance
// ---------------------------------------------------------------------------

/// Reject a case-4 penalty outside the `[0, 1]` domain G2 fixes.
///
/// The negated test also rejects NaN, for which every comparison is false.
/// Rationale: docs/engineering_notes.md#fks-penalty-domain-check
inline void checked_penalty(Real penalty) {
    if (!(penalty >= 0.0 && penalty <= 1.0)) {
        throw std::invalid_argument("the case-4 penalty must lie in [0, 1]");
    }
}

/// The maximum of `K^(p)` over two top-k lists, attained when they are disjoint.
///
/// Equals `k^2 + p * k * (k - 1)`; `k = 1` gives 1, not 0. Not `noexcept`: a
/// penalty outside `[0, 1]` throws where the reference raises `ValueError`.
/// Rationale: docs/engineering_notes.md#fks-maximum
[[nodiscard]] inline Real fks_max(std::int32_t k, Real penalty = kFksPenalty) {
    checked_penalty(penalty);
    if (k < 1) {
        return 0.0;  // k = 0 is the only case with no pairs at all
    }
    const auto square = static_cast<std::int64_t>(k) * static_cast<std::int64_t>(k);
    return static_cast<Real>(square) + penalty * static_cast<Real>(k) * static_cast<Real>(k - 1);
}

/// Generalised Kendall distance between two top-k lists, best first (G2b).
///
/// Pair cost: case 1 (in both) 1 if opposed; case 2 (both in one, one in the other)
/// 1 if the holder ranks the missing one first; case 3 (one in each) 1; case 4 `p`.
/// Rationale: docs/engineering_notes.md#fks-generalised-kendall-cases
[[nodiscard]] inline Real kendall_fks(std::span<const DocId> a,
                                      std::span<const DocId> b,
                                      Real penalty = kFksPenalty,
                                      bool normalise = true) {
    checked_penalty(penalty);
    // First-appearance order, a then b. The enumeration order below is part of
    // the contract with the reference.
    std::vector<DocId> uni;
    uni.reserve(a.size() + b.size());
    std::unordered_set<DocId> seen;
    seen.reserve(a.size() + b.size());
    for (const std::span<const DocId> part : {a, b}) {
        for (const DocId item : part) {
            if (seen.insert(item).second) {
                uni.push_back(item);
            }
        }
    }

    const auto pos_a = detail::positions(a);
    const auto pos_b = detail::positions(b);

    // `both` ranks x and y; `other` ranks exactly one of them. An element
    // present in a list outranks one absent from it, so the two lists disagree
    // exactly when `both` puts the element `other` is missing first.
    const auto case_two = [](std::int32_t both_x, std::int32_t both_y, bool x_in_other) {
        return (x_in_other ? both_y < both_x : both_x < both_y) ? 1.0 : 0.0;
    };

    Real total = 0.0;
    for (std::size_t i = 0; i + 1 < uni.size(); ++i) {
        for (std::size_t j = i + 1; j < uni.size(); ++j) {
            const DocId x = uni[i];
            const DocId y = uni[j];
            const auto ax = pos_a.find(x);
            const auto ay = pos_a.find(y);
            const auto bx = pos_b.find(x);
            const auto by = pos_b.find(y);
            const bool x_in_a = ax != pos_a.end();
            const bool y_in_a = ay != pos_a.end();
            const bool x_in_b = bx != pos_b.end();
            const bool y_in_b = by != pos_b.end();

            if (x_in_a && y_in_a && x_in_b && y_in_b) {  // case 1
                if ((ax->second < ay->second) != (bx->second < by->second)) {
                    total += 1.0;
                }
            } else if (x_in_a && y_in_a && (x_in_b || y_in_b)) {  // case 2, `a` holds both
                total += case_two(ax->second, ay->second, x_in_b);
            } else if (x_in_b && y_in_b && (x_in_a || y_in_a)) {  // case 2, `b` holds both
                total += case_two(bx->second, by->second, x_in_a);
            } else if ((x_in_a && y_in_a) || (x_in_b && y_in_b)) {  // case 4
                total += penalty;
            } else {  // case 3: one element from each list alone
                total += 1.0;
            }
        }
    }

    if (!normalise) {
        return total;
    }
    const auto k = static_cast<std::int32_t>(std::max(a.size(), b.size()));
    const Real ceiling = fks_max(k, penalty);
    return ceiling > 0.0 ? total / ceiling : 0.0;
}

// ---------------------------------------------------------------------------
// Set-level measures
// ---------------------------------------------------------------------------

/// Whether the two top-k *sets* differ; section 7.3's headline indicator.
///
/// A pure reordering within an unchanged set is a different phenomenon, and
/// `kendall_fks` is what measures it.
[[nodiscard]] inline bool top_k_disagreement(std::span<const DocId> a, std::span<const DocId> b) {
    const std::unordered_set<DocId> sa(a.begin(), a.end());
    const std::unordered_set<DocId> sb(b.begin(), b.end());
    return sa != sb;
}

/// `1 - |A n B| / |A u B|`; `0.0` when both sets are empty.
[[nodiscard]] inline Real jaccard_distance(std::span<const DocId> a, std::span<const DocId> b) {
    const std::unordered_set<DocId> sa(a.begin(), a.end());
    const std::unordered_set<DocId> sb(b.begin(), b.end());
    std::unordered_set<DocId> uni = sa;
    uni.insert(sb.begin(), sb.end());
    if (uni.empty()) {
        return 0.0;
    }
    std::size_t shared = 0;
    for (const DocId item : sa) {
        if (sb.contains(item)) {
            ++shared;
        }
    }
    return 1.0 - static_cast<Real>(shared) / static_cast<Real>(uni.size());
}

/// Every ordering measure for one pair of top-k lists at one `k`.
///
/// `kendall_intersection` cannot see the membership change section 7.3 measures,
/// so `intersection_size` always travels beside it.
struct TopKComparison {
    std::int32_t k = 0;
    /// 1[topk(a) != topk(b)].
    bool sets_differ = false;
    /// Normalised FKS `K^(1/2)` in [0, 1].
    Real fks = 0.0;
    /// Kendall tau on the intersection, NaN when fewer than two are shared.
    ///
    /// NaN means undefined: 0 would assert "no reordering" rather than withhold a value.
    Real kendall_intersection = std::numeric_limits<Real>::quiet_NaN();
    std::int32_t intersection_size = 0;
    Real jaccard = 0.0;
    /// The larger of the counts of documents that left and entered the top-k.
    ///
    /// On equal-length prefixes the two are equal and give the number of swaps.
    /// On unequal lengths half their sum rounds a one-document difference to 0
    /// while `sets_differ` reports true.
    std::int32_t swapped = 0;
};

/// Compare two top-k lists under every measure of G2(b) and G2(c).
[[nodiscard]] inline TopKComparison compare_top_k(std::span<const DocId> a,
                                                  std::span<const DocId> b,
                                                  std::int32_t k) {
    const auto prefix = [k](std::span<const DocId> s) {
        const auto want = static_cast<std::size_t>(std::max(k, 0));
        return s.subspan(0, std::min(want, s.size()));
    };
    const std::span<const DocId> pa = prefix(a);
    const std::span<const DocId> pb = prefix(b);

    const std::unordered_set<DocId> sa(pa.begin(), pa.end());
    const std::unordered_set<DocId> sb(pb.begin(), pb.end());
    std::unordered_set<DocId> shared;
    for (const DocId item : sa) {
        if (sb.contains(item)) {
            shared.insert(item);
        }
    }

    TopKComparison out;
    out.k = k;
    out.sets_differ = sa != sb;
    out.fks = kendall_fks(pa, pb);
    out.intersection_size = static_cast<std::int32_t>(shared.size());
    out.jaccard = jaccard_distance(pa, pb);
    out.swapped = static_cast<std::int32_t>(std::max(sa.size(), sb.size()) - shared.size());

    if (shared.size() >= 2) {
        std::vector<DocId> ra;
        std::vector<DocId> rb;
        for (const DocId item : pa) {
            if (shared.contains(item)) {
                ra.push_back(item);
            }
        }
        for (const DocId item : pb) {
            if (shared.contains(item)) {
                rb.push_back(item);
            }
        }
        out.kendall_intersection = kendall_tau_distance(ra, rb);
    }
    return out;
}

}  // namespace tfidf::ranking
