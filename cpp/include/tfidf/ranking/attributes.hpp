// Tie-break attributes as they cross the language boundary.
//
// Python builds the rank encoding once per corpus in exact integer arithmetic and passes
// `int32` ranks as data, as `spec_addenda.md#g13` does for `idf`; comparing them cannot overflow.
// Rationale: docs/engineering_notes.md#tie-break-ranks-cross-the-boundary
#pragma once

#include <tfidf/core/types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tfidf::ranking {

/// The maximum number of tie-break attributes a key can carry.
///
/// README section 2.3.1 names three (popularity, rating, engagement) before the
/// identifier. Four keeps `SortKey` at 32 bytes, two per cache line, with a
/// slot spare.
inline constexpr std::size_t kMaxAttributes = 4;

/// Dense integer ranks for one corpus, stored row-major by attribute.
///
/// `ranks` holds `n_attrs` rows of `n_docs` entries, so each attribute is contiguous.
/// A smaller rank sorts earlier. The Python side folds in direction and missing-value
/// placement.
struct RankTable {
    std::vector<std::int32_t> ranks;     ///< n_attrs * n_docs, row-major
    std::vector<std::int32_t> id_ranks;  ///< n_docs; a bijection onto 0..n-1
    std::int32_t n_docs = 0;
    std::int32_t n_attrs = 0;

    [[nodiscard]] std::int32_t rank_of(std::int32_t attr, DocId doc) const noexcept {
        return ranks[static_cast<std::size_t>(attr) * static_cast<std::size_t>(n_docs) +
                     static_cast<std::size_t>(doc)];
    }

    /// Whether the identifier ranks are a bijection onto `0..n_docs-1`.
    ///
    /// Precondition for the sort key being injective, and hence for the sorted
    /// permutation being unique. Everything this layer claims rests on it, so
    /// it is checked rather than assumed.
    [[nodiscard]] bool id_ranks_are_a_bijection() const {
        std::vector<char> seen(static_cast<std::size_t>(n_docs), 0);
        for (const std::int32_t r : id_ranks) {
            if (r < 0 || r >= n_docs || seen[static_cast<std::size_t>(r)]) {
                return false;
            }
            seen[static_cast<std::size_t>(r)] = 1;
        }
        return static_cast<std::int32_t>(id_ranks.size()) == n_docs;
    }
};

/// Exact rational comparison by cross-multiplication, for C++-only tests.
///
/// Tests `a_num * b_den < b_num * a_den`; both denominators must be positive. Products use
/// 128 bits where `__int128` exists; Python guarantees they fit in 64 bits.
/// Rationale: docs/engineering_notes.md#ratio-less-native-cross-check
[[nodiscard]] inline bool ratio_less(std::int64_t a_num,
                                     std::int64_t a_den,
                                     std::int64_t b_num,
                                     std::int64_t b_den) noexcept {
#if defined(__SIZEOF_INT128__)
// __int128 is a compiler extension, so -Wpedantic objects on ISO conformance
// grounds alone. Silenced here rather than by relaxing the project's warnings.
#  if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wpedantic"
#  endif
    return static_cast<__int128>(a_num) * b_den < static_cast<__int128>(b_num) * a_den;
#  if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#  endif
#else
    return a_num * b_den < b_num * a_den;
#endif
}

}  // namespace tfidf::ranking
