// Query scoring: s_i = cos(q, w_i) for every document (README section 2.3).
//
// Two structurally different algorithms, required to agree bit for bit. Two
// independent traversals of the same data, with different memory access
// patterns and different loop nesting, emitting identical binary64 leaves
// little room for an indexing or accumulation bug to hide in.
//
// --- Why they agree -----------------------------------------------------------
//
// TAAT determinism theorem. The outer loop runs over query terms in ascending
// term identifier and each term contributes at most one addition to any given
// accumulator, so document d accumulates over
//
//     ascending term id over supp(q) INTERSECT supp(w_d), starting from 0.0
//
// which is the sequence the merge-based dot product performs. Hence TAAT is
// bit-identical to a naive row-wise dot product for any reduction that is a
// pure left fold.
//
// It holds only while the outer loop ascends. Blocking or reordering the term
// loop breaks it, so no such optimisation is applied on the normative path.
//
// --- Cost ---------------------------------------------------------------------
//
//   TAAT   O(sum of df(t) over query terms) multiply-adds, plus O(|touched|)
//          divisions rather than O(N), since untouched documents score 0.
//   DAAT   O(sum of nnz(d) over candidate documents).
//
// TAAT wins when the query's terms are individually rare, the usual case for a
// TF-IDF profile query, and is the default.
#pragma once

#include <tfidf/core/reduction.hpp>
#include <tfidf/core/types.hpp>
#include <tfidf/vectorisation/sparse.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace tfidf {

/// Reusable scratch for TAAT. Allocated once and reused across queries, so the
/// hot loop allocates nothing.
struct ScoringScratch {
    std::vector<Real> accumulator;  ///< dense, size n_docs
    std::vector<DocId> touched;     ///< which entries are live this query

    void reset(DocId n_docs) {
        accumulator.assign(static_cast<std::size_t>(n_docs), 0.0);
        touched.clear();
        touched.reserve(static_cast<std::size_t>(n_docs) / 8 + 16);
    }

    /// Zero only the entries touched by the previous query: O(|touched|) rather
    /// than O(n_docs), which matters when queries are sparse.
    void clear_touched() {
        for (const DocId d : touched) {
            accumulator[static_cast<std::size_t>(d)] = 0.0;
        }
        touched.clear();
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
                // A slot that accumulated back to 0.0 would be pushed twice.
                // Values are non-negative (TF-IDF lives in the non-negative
                // orthant) and a stored value is never zero in a canonical
                // sparse structure, so it cannot happen; the duplicate would
                // only rewrite the same quotient. Asserted in the test suite.
                if (slot == 0.0) {
                    scratch.touched.push_back(d);
                }
                slot += qv * index.values[p];
            }
        }
    } else {
        std::vector<Policy> accs(n_docs);
        std::vector<char> seen(n_docs, 0);
        for (std::size_t k = 0; k < query.nnz(); ++k) {
            const TermId t = query.indices[k];
            const Real qv = query.values[k];
            for (std::size_t p = index.postings_begin(t); p < index.postings_end(t); ++p) {
                const auto d = static_cast<std::size_t>(index.rowidx[p]);
                if (!seen[d]) {
                    seen[d] = 1;
                    scratch.touched.push_back(index.rowidx[p]);
                }
                accs[d].add(qv * index.values[p]);
            }
        }
        for (const DocId d : scratch.touched) {
            scratch.accumulator[static_cast<std::size_t>(d)] = accs[static_cast<std::size_t>(d)].value();
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
}

/// Document-at-a-time scoring: an independent merge per document.
///
/// Shares no structure with TAAT (no inverted index, no dense accumulator) and
/// must still produce identical bits, which is the strongest check in the
/// native suite.
inline void score_daat(const SparseView& query,
                       const CsrView& corpus,
                       std::span<const Real> doc_norms,
                       Real query_norm,
                       std::span<Real> out,
                       Reduction policy) {
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

}  // namespace tfidf
