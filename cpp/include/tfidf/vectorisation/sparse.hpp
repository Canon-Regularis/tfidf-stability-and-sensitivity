// Sparse vector and matrix structures.
//
// Two invariants make the native backend bit-identical to the Python reference,
// and both are checked rather than assumed:
//
//   1. Indices are strictly ascending within every vector and every row, which
//      fixes the order products are accumulated in. binary64 addition is not
//      associative, so a different order is a different number.
//
//   2. Structure-of-arrays layout. A packed {int32, double} would be 16 bytes
//      with padding (25% waste) and would engage one prefetch stream instead of
//      two.
#pragma once

#include <tfidf/core/reduction.hpp>
#include <tfidf/core/types.hpp>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace tfidf {

/// A sparse vector: parallel arrays of strictly ascending indices and values.
///
/// Non-owning. The Python layer owns every buffer, so nothing is allocated or
/// freed across the language boundary.
struct SparseView {
    std::span<const TermId> indices;
    std::span<const Real> values;
    TermId dim = 0;

    [[nodiscard]] std::size_t nnz() const noexcept { return indices.size(); }
    [[nodiscard]] bool empty() const noexcept { return indices.empty(); }

    /// Whether indices are strictly ascending and in range.
    [[nodiscard]] bool is_canonical() const noexcept {
        if (indices.size() != values.size()) {
            return false;
        }
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= dim) {
                return false;
            }
            if (i > 0 && indices[i - 1] >= indices[i]) {
                return false;
            }
        }
        return true;
    }
};

// -----------------------------------------------------------------------------
// Dot product and norm
// -----------------------------------------------------------------------------

/// Inner product, accumulated in ascending term-identifier order.
///
/// A merge over the two ascending index lists, so the accumulation order is the
/// Python reference's merge order, and the two agree to the last bit rather
/// than to within rounding.
template <class Policy>
[[nodiscard]] Real dot_with(const SparseView& u, const SparseView& v) noexcept {
    Policy acc{};
    std::size_t i = 0;
    std::size_t j = 0;
    const std::size_t nu = u.nnz();
    const std::size_t nv = v.nnz();
    while (i < nu && j < nv) {
        const TermId a = u.indices[i];
        const TermId b = v.indices[j];
        if (a == b) {
            acc.add(u.values[i] * v.values[j]);
            // Both advance, though the sum does not need both to. Indices are
            // strictly ascending, so dropping either increment still gives the
            // same result: the next iteration compares the index that did not
            // move against its successor, falls into the branch below, and
            // advances it there. Disabling one is undetectable from outside,
            // which is why no test asserts it.
            ++i;
            ++j;
        } else if (a < b) {  // never reached for a == b, so `<=` would read the same
            ++i;
        } else {
            ++j;
        }
    }
    return acc.value();
}

[[nodiscard]] inline Real dot(const SparseView& u, const SparseView& v, Reduction p) noexcept {
    switch (p) {
        case Reduction::Naive:
            return dot_with<reduce::Naive>(u, v);
        case Reduction::Neumaier:
            return dot_with<reduce::Neumaier>(u, v);
        case Reduction::Pairwise:
            return dot_with<reduce::Pairwise>(u, v);
        case Reduction::Exact:
            return dot_with<reduce::Exact>(u, v);
    }
    return dot_with<reduce::Naive>(u, v);
}

/// Euclidean norm: sqrt of the sum of squares, in that order.
///
/// No hypot-style rescaling. It would be more robust to overflow and would
/// produce different digits, and README section 6 forbids stabilising
/// transformations.
///
/// IEEE-754 mandates that `sqrt` be correctly rounded, unlike `log`, which is
/// why idf is computed on the Python side and passed in as data
/// (spec_addenda G13).
template <class Policy>
[[nodiscard]] Real l2_norm_with(const SparseView& v) noexcept {
    Policy acc{};
    for (const Real x : v.values) {
        acc.add(x * x);
    }
    return std::sqrt(acc.value());
}

[[nodiscard]] inline Real l2_norm(const SparseView& v, Reduction p) noexcept {
    switch (p) {
        case Reduction::Naive:
            return l2_norm_with<reduce::Naive>(v);
        case Reduction::Neumaier:
            return l2_norm_with<reduce::Neumaier>(v);
        case Reduction::Pairwise:
            return l2_norm_with<reduce::Pairwise>(v);
        case Reduction::Exact:
            return l2_norm_with<reduce::Exact>(v);
    }
    return l2_norm_with<reduce::Naive>(v);
}

// -----------------------------------------------------------------------------
// Matrices
// -----------------------------------------------------------------------------

/// Compressed sparse row: one row per document, columns ascending within a row.
struct CsrView {
    std::span<const Offset> indptr;   ///< size n_rows + 1
    std::span<const TermId> indices;  ///< size nnz, ascending within each row
    std::span<const Real> values;     ///< size nnz
    DocId n_rows = 0;
    TermId n_cols = 0;

    [[nodiscard]] Offset nnz() const noexcept { return static_cast<Offset>(values.size()); }

    [[nodiscard]] SparseView row(DocId i) const noexcept {
        const auto lo = static_cast<std::size_t>(indptr[static_cast<std::size_t>(i)]);
        const auto hi = static_cast<std::size_t>(indptr[static_cast<std::size_t>(i) + 1]);
        return SparseView{indices.subspan(lo, hi - lo), values.subspan(lo, hi - lo), n_cols};
    }

    [[nodiscard]] bool is_canonical() const noexcept {
        if (indptr.size() != static_cast<std::size_t>(n_rows) + 1) {
            return false;
        }
        if (indptr.front() != 0 || indptr.back() != nnz()) {
            return false;
        }
        if (indices.size() != values.size()) {
            return false;
        }
        // Monotonicity, and it must come BEFORE the row loop below.
        //
        // `row(i)` computes `hi - lo` on `std::size_t`. A decreasing segment
        // makes that wrap: at `hi == lo - 1` it wraps to exactly
        // `std::dynamic_extent`, which `subspan` reads as "to the end", so the
        // row silently spans the rest of the arrays and overlaps its successors.
        // Larger jumps give a count past the end, which is undefined.
        //
        // The normative Python checks this explicitly -- `_check_csr` in
        // persistence/save_load.py raises "indptr decreases at row {i}" -- and
        // this mirror did not, so the two disagreed on the same input:
        // `indptr = [0, 3, 2, 4]` over four non-zeros was rejected there and
        // accepted here, since `front == 0` and `back == nnz` both hold and
        // nothing looked between them. Checked here rather than in `row()`
        // because canonical is what `row()` is allowed to assume; that is the
        // contract the Python states and this restores.
        for (std::size_t i = 0; i + 1 < indptr.size(); ++i) {
            if (indptr[i + 1] < indptr[i]) {
                return false;
            }
        }
        for (DocId i = 0; i < n_rows; ++i) {
            if (!row(i).is_canonical()) {
                return false;
            }
        }
        return true;
    }
};

/// Compressed sparse column: the inverted index.
///
/// Owns its storage, being derived rather than supplied. Column `t` holds the
/// postings list of term `t`: the documents containing it, in ascending
/// document order.
struct Csc {
    std::vector<Offset> colptr;  ///< size n_cols + 1
    std::vector<DocId> rowidx;   ///< size nnz, ascending within each column
    std::vector<Real> values;    ///< size nnz
    DocId n_rows = 0;
    TermId n_cols = 0;

    [[nodiscard]] std::size_t postings_begin(TermId t) const noexcept {
        return static_cast<std::size_t>(colptr[static_cast<std::size_t>(t)]);
    }
    [[nodiscard]] std::size_t postings_end(TermId t) const noexcept {
        return static_cast<std::size_t>(colptr[static_cast<std::size_t>(t) + 1]);
    }
    [[nodiscard]] std::size_t df(TermId t) const noexcept {
        return postings_end(t) - postings_begin(t);
    }
};

/// Transpose CSR to CSC by counting sort: O(nnz + n_cols), deterministic.
///
/// Source rows are visited in ascending document order and each column's
/// entries are appended in that order, so every postings list comes out
/// ascending in document id with no sort, and no dependence on a sort's
/// tie-breaking.
[[nodiscard]] inline Csc transpose(const CsrView& csr) {
    Csc out;
    out.n_rows = csr.n_rows;
    out.n_cols = csr.n_cols;
    const auto ncols = static_cast<std::size_t>(csr.n_cols);
    const auto nnz = static_cast<std::size_t>(csr.nnz());

    out.colptr.assign(ncols + 1, 0);
    for (const TermId t : csr.indices) {
        ++out.colptr[static_cast<std::size_t>(t) + 1];
    }
    // From c = 0, although that first iteration adds colptr[0], which `assign`
    // set to zero and the counting loop above never writes (it writes t + 1).
    // Starting from c = 1 would produce the same colptr.
    for (std::size_t c = 0; c < ncols; ++c) {
        out.colptr[c + 1] += out.colptr[c];
    }

    out.rowidx.resize(nnz);
    out.values.resize(nnz);
    std::vector<Offset> cursor(out.colptr.begin(), out.colptr.end() - 1);
    for (DocId d = 0; d < csr.n_rows; ++d) {
        const auto lo = static_cast<std::size_t>(csr.indptr[static_cast<std::size_t>(d)]);
        const auto hi = static_cast<std::size_t>(csr.indptr[static_cast<std::size_t>(d) + 1]);
        for (std::size_t k = lo; k < hi; ++k) {
            const auto t = static_cast<std::size_t>(csr.indices[k]);
            const auto pos = static_cast<std::size_t>(cursor[t]++);
            out.rowidx[pos] = d;
            out.values[pos] = csr.values[k];
        }
    }
    return out;
}

/// Per-row L2 norms, computed once and reused across every query.
[[nodiscard]] inline std::vector<Real> row_norms(const CsrView& csr, Reduction p) {
    std::vector<Real> out(static_cast<std::size_t>(csr.n_rows));
    for (DocId i = 0; i < csr.n_rows; ++i) {
        out[static_cast<std::size_t>(i)] = l2_norm(csr.row(i), p);
    }
    return out;
}

}  // namespace tfidf
