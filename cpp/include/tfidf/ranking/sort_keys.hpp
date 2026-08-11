// The sort key, and why it is a strict total order.
//
// key = (-score, rank_1, ..., rank_m, id_rank), compared ascending and
// lexicographically -- exactly the tuple the Python reference builds.
//
// Negating the score is *exact*: it flips the sign bit and never rounds. Using
// negation rather than a reversed comparator means Python's tuple `<` and this
// `operator<` are literally the same relation, so there is one comparator to
// reason about instead of two that must be kept in agreement.
//
// Because identifier ranks are a bijection, the key is **injective**, so no two
// documents ever compare equal. Two consequences follow, and both are tested:
//
//   * the sorted sequence is *unique*, so every correct sorting algorithm
//     produces the identical permutation and stability is irrelevant;
//   * the output does not depend on the input order.
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

/// One document's position in the ordering. Exactly 32 bytes.
struct SortKey {
    Score neg_score = 0.0;                                    //  8
    std::array<std::int32_t, kMaxAttributes> ranks{};         // 16
    std::int32_t id_rank = 0;                                 //  4
    DocId doc = 0;                                            //  4
};

static_assert(sizeof(SortKey) == 32, "SortKey should stay at two per cache line");

/// Ascending lexicographic comparison.
///
/// Written with explicit `<` rather than a defaulted `operator<=>`, and that is
/// a correctness requirement rather than a style preference: a defaulted `<=>`
/// on a struct containing a `double` yields `std::partial_ordering`, which
/// `std::sort` cannot consume, and forcing it to `strong_ordering` would
/// reintroduce the NaN question that the boundary guard exists to settle.
///
/// With finiteness guaranteed on entry, plain `<` *is* a strict total order.
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
/// The guard runs once per query, O(N), against an O(N log N) sort. It is the
/// only thing between an arbitrary caller and genuine undefined behaviour: a
/// NaN destroys the strict weak ordering, and libstdc++'s final insertion pass
/// then walks off the front of the array -- a real out-of-bounds write, not
/// merely a wrong answer.
[[nodiscard]] inline bool all_finite(std::span<const Real> scores) noexcept {
    for (const Real s : scores) {
        if (!std::isfinite(s)) {
            return false;
        }
    }
    return true;
}

/// Whether the keys are pairwise distinct.
///
/// Injectivity is the precondition for the sorted permutation being unique, so
/// this is the single most important structural property of the key. O(n log n)
/// via a sort of copies; intended for tests and debug builds, never the
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
