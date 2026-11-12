// Core scalar types and enumerations.
//
// Widths are fixed: `int32_t` identifiers halve postings-loop memory traffic
// against 64-bit, and `int64_t` offsets hold counts above 2^31 (total non-zeros).
// Rationale: docs/engineering_notes.md#core-types-integer-widths
#pragma once

#include <cstddef>
#include <cstdint>

namespace tfidf {

/// Term (vocabulary) identifier.
///
/// Assigned in UTF-8 byte order when the vocabulary is frozen, so it is a
/// deterministic function of the token set alone.
using TermId = std::int32_t;

/// Document identifier: a row index into the corpus matrix.
using DocId = std::int32_t;

/// Offset into a sparse structure's flat index/value arrays.
using Offset = std::int64_t;

/// Floating-point value type; every value this project computes is binary64.
///
/// Call sites use it so the precision is explicit, not implied by `double`.
using Real = double;

/// A similarity score, named separately from Real for readability only.
using Score = double;

/// How a sum of floating-point numbers is accumulated.
///
/// Mirrors `tfidf_stability.utils.numerics.Reduction`. The binding's `REDUCTION`
/// maps that enum's four strings to these integers at import, so the strings, not
/// the integers, are the binding contract. `save_load.py` has its own wire codes.
enum class Reduction : std::int32_t {
    /// Plain left-to-right fold, the default for every published result.
    ///
    /// The literal reading of the paper's formulas.
    Naive = 0,
    /// Kahan-Babuska-Neumaier compensated summation.
    Neumaier = 1,
    /// Pairwise summation of 128-element blocks, combined as a streaming tree.
    Pairwise = 2,
    /// Correctly-rounded sum via Shewchuk expansion.
    ///
    /// Agrees bit-for-bit with Python's `math.fsum` on finite inputs.
    Exact = 3,
};

/// Which query-scoring algorithm to use.
///
/// Both agree bit-for-bit under `Reduction::Naive`; the argument is in
/// `similarity/scoring.hpp` and the assertion in the test suite.
enum class ScoringAlgorithm : std::int32_t {
    /// Term-at-a-time over the inverted index, O(sum of df over query terms).
    Taat = 0,
    /// Document-at-a-time merge-intersect, O(sum of nnz over candidates).
    Daat = 1,
};

/// Size of the pairwise-summation base case, equal to numpy's.
///
/// The equal block size alone does not give agreement with numpy; see the
/// note on `reduce::Pairwise`.
inline constexpr std::size_t kPairwiseBlock = 128;

}  // namespace tfidf
