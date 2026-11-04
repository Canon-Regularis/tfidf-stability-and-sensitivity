// Lexicographic ranking sort key, a strict total order on documents.
//
// key = (-score, rank_1, ..., rank_m, id_rank), compared ascending, as in the Python reference.
// Identifier ranks are a bijection, so the key is injective and the sorted order is unique.
// Rationale: docs/engineering_notes.md#sort-key-strict-total-order
#pragma once

#include <tfidf/core/types.hpp>
#include <tfidf/ranking/attributes.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace tfidf::ranking {

/// One document's position in the ordering, exactly 32 bytes.
struct SortKey {
    Score neg_score = 0.0;                                    //  8
    std::array<std::int32_t, kMaxAttributes> ranks{};         // 16
    std::int32_t id_rank = 0;                                 //  4
    DocId doc = 0;                                            //  4
};

static_assert(sizeof(SortKey) == 32, "SortKey should stay at two per cache line");

/// Ascending lexicographic comparison.
///
/// A defaulted `<=>` over a `double` yields `std::partial_ordering`, which `std::sort`
/// cannot use. With finite scores on entry, plain `<` is a strict total order.
/// Rationale: docs/engineering_notes.md#sort-key-explicit-less-than
[[nodiscard]] inline bool key_less(const SortKey& a, const SortKey& b) noexcept {
    if (a.neg_score != b.neg_score) {
        return a.neg_score < b.neg_score;
    }
    for (std::size_t i = 0; i < kMaxAttributes; ++i) {
        if (a.ranks[i] != b.ranks[i]) {
            return a.ranks[i] < b.ranks[i];
        }
    }
    return a.id_rank < b.id_rank;
}

/// Build one key per document under the given attribute priority.
///
/// `priority` holds row indices into `table.ranks`; an empty priority is the
/// score-only operator, whose ties fall straight through to the identifier.
inline void build_keys(std::span<const Real> scores,
                       const RankTable& table,
                       std::span<const std::int32_t> priority,
                       std::span<SortKey> out) noexcept {
    const auto n = static_cast<std::size_t>(table.n_docs);
    for (std::size_t i = 0; i < n; ++i) {
        SortKey& k = out[i];
        k.neg_score = -scores[i];  // exact: a sign-bit flip
        k.ranks.fill(0);
        for (std::size_t a = 0; a < priority.size() && a < kMaxAttributes; ++a) {
            k.ranks[a] = table.rank_of(priority[a], static_cast<DocId>(i));
        }
        k.id_rank = table.id_ranks[i];
        k.doc = static_cast<DocId>(i);
    }
}

/// Whether every score is finite.
///
/// O(N) per query against an O(N log N) sort. A NaN breaks the strict weak ordering,
/// and libstdc++'s final insertion pass then writes out of bounds.
/// Rationale: docs/engineering_notes.md#sort-key-finiteness-guard
[[nodiscard]] inline bool all_finite(std::span<const Real> scores) noexcept {
    for (const Real s : scores) {
        if (!std::isfinite(s)) {
            return false;
        }
    }
    return true;
}

/// Whether every attribute in `priority` fits in a key.
///
/// `build_keys` is `noexcept` and truncates past `kMaxAttributes`, so the key
/// sorts on a prefix and orders ties differently from the unbounded Python
/// tuple. The binding raises; this predicate serves callers that cannot raise.
[[nodiscard]] inline bool priority_fits(std::span<const std::int32_t> priority) noexcept {
    return priority.size() <= kMaxAttributes;
}

/// Whether the keys are pairwise distinct.
///
/// Injectivity is the precondition for the sorted permutation being unique.
/// O(n log n) via a sort of copies; for tests and debug builds, never the
/// published path.
[[nodiscard]] inline bool keys_are_injective(std::span<const SortKey> keys) {
    std::vector<SortKey> copy(keys.begin(), keys.end());
    std::sort(copy.begin(), copy.end(), key_less);
    for (std::size_t i = 1; i < copy.size(); ++i) {
        // Neither less than the other, under a strict order, means equal.
        if (!key_less(copy[i - 1], copy[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace tfidf::ranking
