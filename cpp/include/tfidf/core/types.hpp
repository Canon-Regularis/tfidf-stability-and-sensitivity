// Core scalar types and enumerations.
//
// Widths are pinned rather than left to the platform. `int32_t` identifiers
// support 2.1 billion documents and terms -- far beyond anything this project
// will index -- while halving the memory traffic of the postings loops against
// 64-bit alternatives, which is the dominant cost in term-at-a-time scoring.
// `int64_t` offsets are used where a count can exceed 2^31 (total non-zeros).
#pragma once

#include <cstddef>
#include <cstdint>

namespace tfidf {

/// Term (vocabulary) identifier. Assigned by UTF-8 byte order at vocabulary
/// freeze time, so it is a deterministic function of the token set alone.
using TermId = std::int32_t;

/// Document identifier: a row index into the corpus matrix.
using DocId = std::int32_t;

/// Offset into a sparse structure's flat index/value arrays.
using Offset = std::int64_t;

/// Every value this project computes is binary64. Named so the intent is
/// explicit at every call site rather than implied by `double`.
using Real = double;

/// A similarity score. Distinguished from Real purely for readability.
using Score = double;

/// How a sum of floating-point numbers is accumulated.
///
/// Mirrors `tfidf_stability.utils.numerics.Reduction` exactly, including the
/// integer values, which cross the language boundary.
enum class Reduction : std::int32_t {
    /// Plain left-to-right fold. The literal reading of the paper's formulas
    /// and the default for every published result.
    Naive = 0,
    /// Kahan-Babuska-Neumaier compensated summation.
    Neumaier = 1,
    /// Recursive pairwise summation, 128-element base case (matches numpy).
    Pairwise = 2,
    /// Correctly-rounded sum via Shewchuk expansion (matches Python's math.fsum).
    Exact = 3,
};

/// Which query-scoring algorithm to use. They must agree bit-for-bit under
/// `Reduction::Naive`; see `scoring.hpp` for why, and the test suite for the
/// assertion.
enum class ScoringAlgorithm : std::int32_t {
    /// Term-at-a-time over the inverted index. O(sum of df over query terms).
    Taat = 0,
    /// Document-at-a-time merge-intersect. O(sum of nnz over candidates).
    Daat = 1,
};

/// Size of the pairwise-summation base case. Matches numpy's, so the
/// `Pairwise` policy and the numpy cross-check agree.
inline constexpr std::size_t kPairwiseBlock = 128;

}  // namespace tfidf
