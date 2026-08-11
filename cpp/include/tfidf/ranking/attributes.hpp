// Tie-break attributes, as they cross the language boundary.
//
// The native side never builds a rank encoding and never compares a rational.
// Python does that once per corpus, in unbounded-precision integer arithmetic,
// and hands over the resulting `int32` ranks as data. This is the same move
// `spec_addenda.md#g13` makes for `idf`: compute the delicate thing once, in
// exact arithmetic, on the Python side.
//
// The payoff is that the cross-language question stops being "do two
// rational-comparison implementations agree semantically?" and becomes
// `py_ranks == cpp_ranks`, an integer equality. It also removes any possibility
// of overflow from the hot path, since no multiplication happens there at all.
#pragma once

#include <tfidf/core/types.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace tfidf::ranking {

/// The maximum number of tie-break attributes a key can carry.
///
/// README section 2.3.1 names three (popularity, rating, engagement) before the
/// identifier. Four keeps `SortKey` at exactly 32 bytes -- two per cache line --
/// with a slot spare.
inline constexpr std::size_t kMaxAttributes = 4;

/// Dense integer ranks for one corpus: `n_attrs` rows of `n_docs` entries,
/// stored row-major so a whole attribute is contiguous.
///
/// Smaller means earlier in the final order. Direction and missing-value
/// placement are already folded in by the Python side, so nothing here needs to
/// know about either.
struct RankTable {
    std::vector<std::int32_t> ranks;     ///< n_attrs * n_docs, row-major
    std::vector<std::int32_t> id_ranks;  ///< n_docs; a bijection onto 0..n-1
    std::int32_t n_docs = 0;
    std::int32_t n_attrs = 0;

    [[nodiscard]] std::int32_t rank_of(std::int32_t attr, DocId doc) const noexcept {
        return ranks[static_cast<std::size_t>(attr) * static_cast<std::size_t>(n_docs) +
                     static_cast<std::size_t>(doc)];
    }

    /// Whether the identifier ranks really are a bijection onto `0..n_docs-1`.
    ///
    /// This is the precondition for the sort key being *injective*, and hence
    /// for the sorted permutation being unique. Everything this layer claims
    /// rests on it, so it is checked rather than assumed.
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

/// Exact rational comparison, for the C++-only tests and benchmarks.
///
/// The published path never calls this -- Python has already reduced ratios to
/// ranks. It exists so the native test suite can verify that the two languages
/// agree on the *rule*, not merely on the pre-computed answer.
///
/// Uses a 128-bit intermediate where the compiler provides one. The Python side
/// separately guarantees the products fit in 64 bits, so this is belt and
/// braces on a cold path.
[[nodiscard]] inline bool ratio_less(std::int64_t a_num,
                                     std::int64_t a_den,
                                     std::int64_t b_num,
                                     std::int64_t b_den) noexcept {
#if defined(__SIZEOF_INT128__)
// __int128 is a compiler extension, so -Wpedantic objects. It is the right tool
// here and the objection is purely about ISO conformance, so it is silenced
// locally rather than by relaxing the project's warning set.
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
