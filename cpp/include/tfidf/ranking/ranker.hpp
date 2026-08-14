// The ranking operator: turn scores plus tie-break ranks into a permutation.
//
// All five selection strategies must produce the identical permutation, the
// ranking analogue of the `TAAT == DAAT` check in the scoring layer, and the
// operational content of the claim that sort stability cannot matter here.
//
// The claim holds because the key is injective (identifier ranks are a
// bijection), so no two documents ever compare equal, so "elements that compare
// equal keep their input order" quantifies over the empty set and every correct
// algorithm satisfies it vacuously.
#pragma once

#include <tfidf/core/types.hpp>
#include <tfidf/ranking/attributes.hpp>
#include <tfidf/ranking/sort_keys.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace tfidf::ranking {

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
/// Two hazards in the standard algorithms, handled explicitly here:
///
///   * `nth_element` says nothing about the contents of the remainder and
///     standard libraries partition it differently, so `[m, n)` must never be
///     read;
///   * `nth_element` leaves the selected prefix unordered, so it is sorted
///     afterwards.
///
/// The implementation-independent postcondition (every key before `m` compares
/// less than every key after it) is what `partition_is_valid` checks in the
/// test build.
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
/// Sorted as raw doubles, independently of any ranking. One array serves every
/// operator, every `k` and every `tau`, which is what makes margins provably
/// independent of the tie-break.
[[nodiscard]] inline std::vector<Real> sorted_scores_desc(std::span<const Real> scores) {
    std::vector<Real> out(scores.begin(), scores.end());
    std::sort(out.begin(), out.end(), std::greater<>());
    return out;
}

}  // namespace tfidf::ranking
