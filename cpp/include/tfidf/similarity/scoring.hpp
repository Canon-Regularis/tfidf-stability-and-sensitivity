// Query scoring: s_i = cos(q, w_i) for every document (README section 2.3).
//
// TAAT (the default) and DAAT agree bit for bit: TAAT's ascending term loop adds
// each document's products in the merge dot product's order, so it is not reordered.
// Rationale: docs/engineering_notes.md#scoring-taat-determinism
#pragma once

#include <tfidf/core/reduction.hpp>
#include <tfidf/core/types.hpp>
#include <tfidf/vectorisation/sparse.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tfidf {

namespace detail {

/// Preconditions shared by both scoring kernels.
///
/// The spans carry their sizes and nothing else checks them, so a short `out` or
/// `doc_norms` would index past the end. The Python reference `taat_scores`
/// refuses the first two mismatches and allocates its own output.
inline void check_scoring_shapes(TermId query_dim,
                                 TermId n_cols,
                                 DocId n_rows,
                                 std::span<const Real> doc_norms,
                                 std::span<Real> out) {
    if (query_dim != n_cols) {
        throw std::invalid_argument("dimension mismatch: query and corpus vocabularies differ");
    }
    const auto n = static_cast<std::size_t>(n_rows);
    if (doc_norms.size() != n) {
        throw std::invalid_argument("one norm per document is required");
    }
    if (out.size() != n) {
        throw std::invalid_argument("one output slot per document is required");
    }
}

}  // namespace detail

/// Reusable scratch for TAAT, sized once and reused across queries.
///
/// A buffer allocates only when a query outgrows the capacity it keeps across
/// queries; `Exact` accumulators also allocate as their expansions grow.
/// Rationale: docs/engineering_notes.md#scoring-scratch-reuse
struct ScoringScratch {
    static constexpr std::size_t kNoSlot = ~std::size_t{0};

    std::vector<Real> accumulator;           ///< dense, size n_docs
    std::vector<DocId> touched;              ///< which entries are live this query
    std::vector<std::size_t> slot_of;        ///< compensated policies: document to slot
    std::vector<reduce::Neumaier> neumaier;  ///< slot i accumulates touched[i]
    std::vector<reduce::Pairwise> pairwise;
    std::vector<reduce::Exact> exact;

    void reset(DocId n_docs) {
        accumulator.assign(static_cast<std::size_t>(n_docs), 0.0);
        touched.clear();
        touched.reserve(static_cast<std::size_t>(n_docs) / 8 + 16);
        slot_of.clear();
    }

    /// Zero only the entries the previous query touched.
    ///
    /// Costs O(|touched|) rather than O(n_docs), which matters for sparse queries.
    void clear_touched() {
        for (const DocId d : touched) {
            accumulator[static_cast<std::size_t>(d)] = 0.0;
        }
        if (!slot_of.empty()) {
            for (const DocId d : touched) {
                slot_of[static_cast<std::size_t>(d)] = kNoSlot;
            }
        }
        touched.clear();
    }

    /// The slot storage for one compensated policy.
    template <class Policy>
    [[nodiscard]] std::vector<Policy>& slots() noexcept {
        if constexpr (std::is_same_v<Policy, reduce::Neumaier>) {
            return neumaier;
        } else if constexpr (std::is_same_v<Policy, reduce::Pairwise>) {
            return pairwise;
        } else {
            static_assert(std::is_same_v<Policy, reduce::Exact>);
            return exact;
        }
    }
};

/// Term-at-a-time scoring over the inverted index.
///
/// Writes `out[i] = cos(query, w_i)` for every document, applying the
/// zero-vector convention of section 2.3.
template <class Policy>
void score_taat_with(const SparseView& query,
                     const Csc& index,
                     std::span<const Real> doc_norms,
                     Real query_norm,
                     std::span<Real> out,
                     ScoringScratch& scratch) {
    detail::check_scoring_shapes(query.dim, index.n_cols, index.n_rows, doc_norms, out);
    const auto n_docs = static_cast<std::size_t>(index.n_rows);
    std::fill(out.begin(), out.end(), 0.0);
    if (query_norm == 0.0 || query.empty()) {
        return;  // a zero query scores 0 everywhere (spec_addenda G3)
    }

    if (scratch.accumulator.size() != n_docs) {
        scratch.reset(index.n_rows);
    } else {
        scratch.clear_touched();
    }

    // Compensation state is per-document, so a compensated policy needs one
    // accumulator object per touched document. `Naive` holds no state beyond the
    // running sum and can share the dense array directly.
    if constexpr (std::is_same_v<Policy, reduce::Naive>) {
        // Ascending term id: this is what makes the result bit-identical to the
        // merge-based dot product. Do not reorder.
        for (std::size_t k = 0; k < query.nnz(); ++k) {
            const TermId t = query.indices[k];
            const Real qv = query.values[k];
            const std::size_t lo = index.postings_begin(t);
            const std::size_t hi = index.postings_end(t);
            for (std::size_t p = lo; p < hi; ++p) {
                const DocId d = index.rowidx[p];
                Real& slot = scratch.accumulator[static_cast<std::size_t>(d)];
                // TF-IDF values are non-negative and stored entries non-zero, so a slot stays
                // 0.0 after an addition only if every product so far underflowed. A duplicate
                // push only rewrites the same quotient; test_scoring.cpp asserts that each
                // document enters `touched` once.
                if (slot == 0.0) {
                    scratch.touched.push_back(d);
                }
                slot += qv * index.values[p];
            }
        }
    } else {
        std::vector<Policy>& accs = scratch.slots<Policy>();
        if (scratch.slot_of.size() != n_docs) {
            scratch.slot_of.assign(n_docs, ScoringScratch::kNoSlot);
        }
        for (std::size_t k = 0; k < query.nnz(); ++k) {
            const TermId t = query.indices[k];
            const Real qv = query.values[k];
            for (std::size_t p = index.postings_begin(t); p < index.postings_end(t); ++p) {
                const auto d = static_cast<std::size_t>(index.rowidx[p]);
                std::size_t& slot = scratch.slot_of[d];
                if (slot == ScoringScratch::kNoSlot) {
                    // A slot is reset at its first touch in each query, so it
                    // starts from the state a fresh accumulator has.
                    const std::size_t next = scratch.touched.size();
                    scratch.touched.push_back(index.rowidx[p]);
                    if (next == accs.size()) {
                        accs.emplace_back();
                    } else {
                        accs[next] = Policy{};
                    }
                    slot = next;
                }
                accs[slot].add(qv * index.values[p]);
            }
        }
        for (std::size_t i = 0; i < scratch.touched.size(); ++i) {
            scratch.accumulator[static_cast<std::size_t>(scratch.touched[i])] = accs[i].value();
        }
    }

    for (const DocId d : scratch.touched) {
        const auto i = static_cast<std::size_t>(d);
        const Real dn = doc_norms[i];
        // dot / (qn * dn), the form the Python reference pins. (dot/qn)/dn and
        // dot * (1/(qn*dn)) round differently.
        out[i] = (dn == 0.0) ? 0.0 : scratch.accumulator[i] / (query_norm * dn);
    }
}

/// Term-at-a-time scoring under a policy chosen at run time.
inline void score_taat(const SparseView& query,
                       const Csc& index,
                       std::span<const Real> doc_norms,
                       Real query_norm,
                       std::span<Real> out,
                       ScoringScratch& scratch,
                       Reduction policy) {
    switch (policy) {
        case Reduction::Naive:
            score_taat_with<reduce::Naive>(query, index, doc_norms, query_norm, out, scratch);
            return;
        case Reduction::Neumaier:
            score_taat_with<reduce::Neumaier>(query, index, doc_norms, query_norm, out, scratch);
            return;
        case Reduction::Pairwise:
            score_taat_with<reduce::Pairwise>(query, index, doc_norms, query_norm, out, scratch);
            return;
        case Reduction::Exact:
            score_taat_with<reduce::Exact>(query, index, doc_norms, query_norm, out, scratch);
            return;
    }
    // No `default:`, so `-Wswitch` still makes a new enumerator a compile error.
    // Reached only for a value outside the enumeration, which would otherwise
    // return leaving `out` unwritten, since the fill is inside the kernel.
    throw std::invalid_argument("unknown reduction policy");
}

/// Document-at-a-time scoring: an independent merge per document.
///
/// Shares no structure with TAAT (no inverted index, no dense accumulator) and
/// produces identical bits, so each kernel checks the other.
inline void score_daat(const SparseView& query,
                       const CsrView& corpus,
                       std::span<const Real> doc_norms,
                       Real query_norm,
                       std::span<Real> out,
                       Reduction policy) {
    detail::check_scoring_shapes(query.dim, corpus.n_cols, corpus.n_rows, doc_norms, out);
    if (query_norm == 0.0 || query.empty()) {
        std::fill(out.begin(), out.end(), 0.0);
        return;
    }
    for (DocId d = 0; d < corpus.n_rows; ++d) {
        const auto i = static_cast<std::size_t>(d);
        const Real dn = doc_norms[i];
        out[i] = (dn == 0.0) ? 0.0 : dot(query, corpus.row(d), policy) / (query_norm * dn);
    }
}

/// Score a query by the requested algorithm.
inline void score(const SparseView& query,
                  const CsrView& corpus,
                  const Csc& index,
                  std::span<const Real> doc_norms,
                  Real query_norm,
                  std::span<Real> out,
                  ScoringScratch& scratch,
                  Reduction policy,
                  ScoringAlgorithm algorithm) {
    if (algorithm == ScoringAlgorithm::Daat) {
        score_daat(query, corpus, doc_norms, query_norm, out, policy);
    } else {
        score_taat(query, index, doc_norms, query_norm, out, scratch, policy);
    }
}

/// An immutable fitted corpus: CSR, its inverted index and the row norms.
///
/// Construction refuses a corpus the kernels cannot read and fixes the one
/// policy every norm and dot product uses, so a score never mixes two.
/// Rationale: docs/engineering_notes.md#fitted-index
class FittedIndex {
  public:
    FittedIndex(std::vector<Offset> indptr,
                std::vector<TermId> indices,
                std::vector<Real> values,
                DocId n_docs,
                TermId n_terms,
                Reduction policy)
        : indptr_(std::move(indptr)),
          indices_(std::move(indices)),
          values_(std::move(values)),
          n_rows_(n_docs),
          n_cols_(n_terms),
          policy_(policy) {
        // Signs before the length: at n_docs = -1 the size_t sum n_docs + 1
        // wraps to 0, so an empty indptr would pass the next check.
        if (n_docs < 0 || n_terms < 0) {
            throw std::invalid_argument("n_docs and n_terms must be non-negative");
        }
        if (indptr_.size() != static_cast<std::size_t>(n_docs) + 1) {
            throw std::invalid_argument("indptr must have length n_docs + 1");
        }
        if (indices_.size() != values_.size()) {
            throw std::invalid_argument("indices and values must be the same length");
        }
        if (!csr().is_canonical()) {
            throw std::invalid_argument(
                "the CSR matrix is not canonical: indices must be strictly ascending "
                "within each row and lie in [0, n_terms)");
        }
        index_ = transpose(csr());
        norms_ = row_norms(csr(), policy_);
    }

    /// A view onto the owned CSR, built per call so a copy never borrows.
    [[nodiscard]] CsrView csr() const noexcept {
        return CsrView{indptr_, indices_, values_, n_rows_, n_cols_};
    }
    [[nodiscard]] const Csc& index() const noexcept { return index_; }
    [[nodiscard]] std::span<const Real> norms() const noexcept { return norms_; }
    [[nodiscard]] Reduction policy() const noexcept { return policy_; }
    [[nodiscard]] DocId n_documents() const noexcept { return n_rows_; }
    [[nodiscard]] TermId n_features() const noexcept { return n_cols_; }
    [[nodiscard]] Offset nnz() const noexcept { return static_cast<Offset>(values_.size()); }

    /// Document frequency of a term, read off the inverted index.
    [[nodiscard]] std::size_t df(TermId term) const {
        if (term < 0 || term >= n_cols_) {
            throw std::out_of_range("term identifier out of range");
        }
        return index_.df(term);
    }

    /// Score one query against every document: `out[i] = cos(query, w_i)`.
    ///
    /// Refuses a non-canonical query, then an algorithm outside the enum, and
    /// takes the query norm under the index's own policy.
    void score(const SparseView& query, std::span<Real> out, ScoringAlgorithm algorithm) {
        if (!query.is_canonical()) {
            throw std::invalid_argument(
                "the query is not canonical: indices must be strictly ascending "
                "and lie in [0, n_features)");
        }
        if (algorithm < ScoringAlgorithm::Taat || algorithm > ScoringAlgorithm::Daat) {
            throw std::invalid_argument("scoring algorithm out of range");
        }
        const Real query_norm = l2_norm(query, policy_);
        tfidf::score(query, csr(), index_, norms_, query_norm, out, scratch_, policy_, algorithm);
    }

  private:
    std::vector<Offset> indptr_;
    std::vector<TermId> indices_;
    std::vector<Real> values_;
    DocId n_rows_;
    TermId n_cols_;
    Reduction policy_;
    Csc index_;
    std::vector<Real> norms_;
    ScoringScratch scratch_;
};

}  // namespace tfidf
