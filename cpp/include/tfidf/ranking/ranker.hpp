// The ranking operator: turns scores plus tie-break ranks into a permutation.
//
// All five selection strategies produce the identical permutation. The sort key
// is injective, so sort stability cannot affect the order.
// Rationale: docs/engineering_notes.md#ranker-injective-key-stability
#pragma once

#include <tfidf/core/types.hpp>
#include <tfidf/ranking/sort_keys.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace tfidf::ranking {

/// Which sorting strategy produces the order.
///
/// Python looks each value up by name in the exported `SELECTION`, as it does
/// `Reduction` in `REDUCTION`. Every strategy yields the same permutation.
/// Rationale: docs/engineering_notes.md#ranker-selection-strategies
enum class Selection : std::int32_t {
    FullSort = 0,
    StableSort = 1,
    PartialSort = 2,
    NthElement = 3,
    BoundedHeap = 4,
};

/// Sort `keys` in place and write the resulting document order to `out`.
inline void rank_full(std::span<SortKey> keys,
                      std::span<DocId> out,
                      Selection how = Selection::FullSort) {
    switch (how) {
        case Selection::StableSort:
            std::stable_sort(keys.begin(), keys.end(), key_less);
            break;
        case Selection::BoundedHeap:
            std::make_heap(keys.begin(), keys.end(), [](const SortKey& a, const SortKey& b) {
                return key_less(b, a);  // max-heap on the reversed order
            });
            std::sort_heap(keys.begin(), keys.end(), [](const SortKey& a, const SortKey& b) {
                return key_less(b, a);
            });
            std::reverse(keys.begin(), keys.end());
            break;
        case Selection::FullSort:
        case Selection::PartialSort:
        case Selection::NthElement:
        default:
            std::sort(keys.begin(), keys.end(), key_less);
            break;
    }
    for (std::size_t i = 0; i < keys.size(); ++i) {
        out[i] = keys[i].doc;
    }
}

/// Select the `m` best documents, in order, leaving the rest unspecified.
///
/// `nth_element` leaves the order of `[m, n)` unspecified and the prefix unordered,
/// so the remainder is never read and the prefix is sorted afterwards.
/// Rationale: docs/engineering_notes.md#ranker-nth-element-hazards
inline void select_top(std::span<SortKey> keys,
                       std::size_t m,
                       std::span<DocId> out,
                       Selection how = Selection::NthElement) {
    m = std::min(m, keys.size());
    switch (how) {
        case Selection::PartialSort:
            std::partial_sort(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(m),
                              keys.end(), key_less);
            break;
        case Selection::NthElement:
            if (m < keys.size()) {
                std::nth_element(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(m),
                                 keys.end(), key_less);
            }
            std::sort(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(m), key_less);
            break;
        default:
            // Every other strategy degenerates to ranking everything and taking
            // a prefix; the agreement test relies on that.
            std::sort(keys.begin(), keys.end(), key_less);
            break;
    }
    for (std::size_t i = 0; i < m; ++i) {
        out[i] = keys[i].doc;
    }
}

/// Whether the first `m` keys all compare less than the remainder.
///
/// The only postcondition of a partial selection that is guaranteed across
/// standard-library implementations.
[[nodiscard]] inline bool partition_is_valid(std::span<const SortKey> keys,
                                             std::size_t m) noexcept {
    if (m >= keys.size()) {
        return true;
    }
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = m; j < keys.size(); ++j) {
            if (!key_less(keys[i], keys[j])) {
                return false;
            }
        }
    }
    return true;
}

/// Scores in non-increasing order.
///
/// `std::stable_sort` matches the stable Python `sorted(..., reverse=True)`
/// and keeps `-0.0` and `0.0`, which compare equal, in input order.
/// Rationale: docs/engineering_notes.md#ranker-stable-score-sort
[[nodiscard]] inline std::vector<Real> sorted_scores_desc(std::span<const Real> scores) {
    std::vector<Real> out(scores.begin(), scores.end());
    std::stable_sort(out.begin(), out.end(), std::greater<>());
    return out;
}

}  // namespace tfidf::ranking
