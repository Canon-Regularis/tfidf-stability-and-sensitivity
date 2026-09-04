// =============================================================================
// module.cpp: the nanobind surface of the native backend.
//
// Argument marshalling and nothing else. No arithmetic lives here, so the
// mathematics in cpp/include/tfidf/ stays testable from C++ alone and the
// binding technology stays swappable.
//
// The Python side owns every input buffer. `NativeIndex` copies what it needs
// into owned storage at construction, so no C++ object holds a borrowed pointer
// into a numpy array that Python might collect.
// =============================================================================
#include <tfidf/core/build_config.hpp>
#include <tfidf/core/fp_guard.hpp>
#include <tfidf/core/reduction.hpp>
#include <tfidf/core/types.hpp>
#include <tfidf/ranking/attributes.hpp>
#include <tfidf/ranking/distances.hpp>
#include <tfidf/ranking/margins.hpp>
#include <tfidf/ranking/ranker.hpp>
#include <tfidf/ranking/sort_keys.hpp>
#include <tfidf/ranking/tie_groups.hpp>
#include <tfidf/similarity/scoring.hpp>
#include <tfidf/vectorisation/sparse.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace tfidf;

namespace {

using F64Array = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I32Array = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64Array = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

std::span<const double> as_span(const F64Array& a) {
    return {a.data(), a.shape(0)};
}

std::span<const DocId> as_ids(const I32Array& a) {
    return {a.data(), a.shape(0)};
}

/// Build a sparse view, checking the two halves describe one vector.
///
/// The free functions below take raw parallel arrays and were the only entry
/// points not checking that the halves agree. `dot` built its SparseView from
/// independently sized index and value spans, so 64 indices with 1 value read
/// 63 doubles past the end of the values buffer: undefined behaviour reachable
/// from pure Python with no unsafe API, returning a different answer on each
/// call. The reference rejects the same input (`SparseVector` raises "indices
/// and values differ in length") and the class methods here already checked it.
SparseView checked_view(const I32Array& idx, const F64Array& val, std::int32_t dim,
                        const char* what) {
    if (idx.shape(0) != val.shape(0)) {
        throw std::invalid_argument(std::string(what) +
                                    " indices and values must be the same length");
    }
    return SparseView{std::span<const TermId>(idx.data(), idx.shape(0)), as_span(val), dim};
}

/// Reject a reduction policy outside the enumeration.
///
/// `static_cast<Reduction>(999)` produced a value no switch handles and the sum
/// fell back to one of the policies instead of failing. The policy is recorded
/// in every run manifest and is never implicit, so substituting one silently is
/// the worst available outcome. Python's `Reduction(999)` raises; so does this.
Reduction checked_policy(std::int32_t policy) {
    if (policy < static_cast<std::int32_t>(Reduction::Naive) ||
        policy > static_cast<std::int32_t>(Reduction::Exact)) {
        throw std::invalid_argument("reduction policy out of range (expected 0..3)");
    }
    return static_cast<Reduction>(policy);
}

/// Reject non-finite scores before they reach a comparator.
///
/// G3 requires this re-check at the boundary, the only thing between an
/// arbitrary caller and undefined behaviour: a NaN makes `<` false in both
/// directions and destroys the strict weak ordering `std::sort` and
/// `std::min_element` require. `NativeRanker::rank` already did it, the free
/// functions below did not, and two consequences were measured. `std::sort`
/// over 65,536 scores containing NaN did not crash but is formally UB, and
/// `min_adjacent_margin_top` returned `inf` where the normative Python
/// reference returns `nan`, a bit-level divergence in a core contracted to be
/// bit-identical with that reference.
std::span<const double> checked_scores(const F64Array& scores) {
    const std::span<const double> span{scores.data(), scores.shape(0)};
    for (const double value : span) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("scores must all be finite");
        }
    }
    return span;
}

/// Reject a tolerance that is negative or NaN.
///
/// `!(tau >= 0.0)` rather than `tau < 0.0`: every comparison with NaN is false,
/// so the second form passes NaN through a guard whose message says
/// non-negative.
///
/// Applied to all four tie-group entry points. An earlier sweep added the check
/// to `tie_ball_interval` alone, leaving `tie_chains`, `tie_cliques` and
/// `chain_inflation_ratio` to accept a negative tau that the normative Python
/// rejects, and to answer it: the ratio comes back as 1.0, its minimum, so an
/// invalid tolerance serialises as the healthiest possible tie structure.
double checked_tau(double tau) {
    if (!(tau >= 0.0)) {
        throw std::invalid_argument("tau must be non-negative");
    }
    return tau;
}

/// Shape of this binding surface, checked against
/// `tfidf_stability._native.REQUIRED_ABI` on import.
///
/// Distinct from the project version. A release can leave the surface untouched
/// and a new binding can land inside one release, so tying the two would make
/// the staleness check fire needlessly, or fail to fire when a rebuilt .pyd is
/// required, which is the dangerous direction.
constexpr const char* kAbi = "0.4.0";

/// Return an owning numpy array built from a vector, without copying twice.
nb::ndarray<nb::numpy, double> to_numpy(std::vector<double>&& v) {
    auto* held = new std::vector<double>(std::move(v));
    nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return nb::ndarray<nb::numpy, double>(held->data(), {held->size()}, owner);
}

/// An immutable, fitted corpus index: CSR, its transpose, and the row norms.
///
/// Every quantity is computed once at construction under a single declared
/// reduction policy, so a scored result cannot mix norms computed one way with
/// dot products computed another.
class NativeIndex {
  public:
    NativeIndex(const I64Array& indptr,
                const I32Array& indices,
                const F64Array& values,
                std::int32_t n_docs,
                std::int32_t n_terms,
                std::int32_t reduction)
        : indptr_(indptr.data(), indptr.data() + indptr.shape(0)),
          indices_(indices.data(), indices.data() + indices.shape(0)),
          values_(values.data(), values.data() + values.shape(0)),
          n_rows_(n_docs),
          n_cols_(n_terms),
          reduction_(checked_policy(reduction)) {
        // Dimensions first, before any size arithmetic. `n_docs = -1` with an
        // empty indptr passes the length check below (0 == -1 + 1), and then
        // `is_canonical()` calls front()/back() on an empty span and
        // `transpose()` sizes a colptr from a negative count. Four such inputs
        // segfaulted the interpreter (SIGSEGV, rc 139) from pure Python with no
        // unsafe API. Every later check casts to std::size_t, where a negative
        // value wraps to something enormous.
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
        csc_ = transpose(csr());
        norms_ = row_norms(csr(), reduction_);
    }

    [[nodiscard]] CsrView csr() const {
        return CsrView{indptr_, indices_, values_, n_rows_, n_cols_};
    }

    [[nodiscard]] std::int32_t n_documents() const noexcept { return n_rows_; }
    [[nodiscard]] std::int32_t n_features() const noexcept { return n_cols_; }
    [[nodiscard]] std::int64_t nnz() const noexcept {
        return static_cast<std::int64_t>(values_.size());
    }
    [[nodiscard]] std::int32_t reduction() const noexcept {
        return static_cast<std::int32_t>(reduction_);
    }
    [[nodiscard]] const std::vector<double>& norms() const noexcept { return norms_; }

    /// Document frequency of a term, read straight off the inverted index.
    [[nodiscard]] std::int64_t df(std::int32_t term) const {
        if (term < 0 || term >= n_cols_) {
            throw std::out_of_range("term identifier out of range");
        }
        return static_cast<std::int64_t>(csc_.df(term));
    }

    /// Score one query against every document: `s_i = cos(q, w_i)`.
    nb::ndarray<nb::numpy, double> score(const I32Array& q_indices,
                                         const F64Array& q_values,
                                         std::int32_t algorithm) {
        if (q_indices.shape(0) != q_values.shape(0)) {
            throw std::invalid_argument("query indices and values must be the same length");
        }
        const SparseView query{
            std::span<const TermId>(q_indices.data(), q_indices.shape(0)),
            as_span(q_values),
            n_cols_,
        };
        if (!query.is_canonical()) {
            throw std::invalid_argument(
                "the query is not canonical: indices must be strictly ascending "
                "and lie in [0, n_features)");
        }

        const double q_norm = l2_norm(query, reduction_);
        std::vector<double> out(static_cast<std::size_t>(n_rows_));
        tfidf::score(query, csr(), csc_, norms_, q_norm, out, scratch_, reduction_,
                     static_cast<ScoringAlgorithm>(algorithm));
        return to_numpy(std::move(out));
    }

  private:
    std::vector<Offset> indptr_;
    std::vector<TermId> indices_;
    std::vector<double> values_;
    std::int32_t n_rows_;
    std::int32_t n_cols_;
    Reduction reduction_;
    Csc csc_;
    std::vector<double> norms_;
    ScoringScratch scratch_;
};

/// Return an owning numpy int32 array built from a vector.
nb::ndarray<nb::numpy, std::int32_t> to_numpy_i32(std::vector<std::int32_t>&& v) {
    auto* held = new std::vector<std::int32_t>(std::move(v));
    nb::capsule owner(held,
                      [](void* p) noexcept { delete static_cast<std::vector<std::int32_t>*>(p); });
    return nb::ndarray<nb::numpy, std::int32_t>(held->data(), {held->size()}, owner);
}

/// A corpus's tie-break ranks plus one operator's attribute priority.
///
/// The rank matrix is built in Python, in exact unbounded-precision integer
/// arithmetic, and crosses the boundary as data. Nothing here re-derives it, so
/// the cross-language check is integer equality rather than a question about
/// two rational-comparison implementations agreeing.
class NativeRanker {
  public:
    NativeRanker(const I32Array& ranks,
                 const I32Array& id_ranks,
                 const I32Array& priority,
                 std::int32_t n_attrs) {
        const auto n_docs = static_cast<std::int32_t>(id_ranks.shape(0));
        if (n_attrs < 0 || (n_attrs > 0 && ranks.shape(0) % static_cast<std::size_t>(n_attrs))) {
            throw std::invalid_argument("the rank matrix is not n_attrs * n_docs");
        }
        const auto expected_ranks =
            static_cast<std::size_t>(n_attrs) * static_cast<std::size_t>(n_docs);
        if (n_attrs > 0 && ranks.shape(0) != expected_ranks) {
            throw std::invalid_argument("the rank matrix does not match n_docs");
        }
        if (priority.shape(0) > ranking::kMaxAttributes) {
            throw std::invalid_argument("more attributes than the sort key can carry");
        }

        table_.n_docs = n_docs;
        table_.n_attrs = n_attrs;
        table_.ranks.assign(ranks.data(), ranks.data() + ranks.shape(0));
        table_.id_ranks.assign(id_ranks.data(), id_ranks.data() + id_ranks.shape(0));
        priority_.assign(priority.data(), priority.data() + priority.shape(0));

        // Injectivity of the key, and hence uniqueness of the sorted
        // permutation, rests entirely on this.
        if (!table_.id_ranks_are_a_bijection()) {
            throw std::invalid_argument(
                "identifier ranks must be a bijection onto 0..n_docs-1; without that the "
                "sort key is not injective and the ranking is not uniquely determined");
        }
        for (const std::int32_t p : priority_) {
            if (p < 0 || p >= n_attrs) {
                throw std::invalid_argument("priority names an attribute that does not exist");
            }
        }
    }

    [[nodiscard]] std::int32_t n_documents() const noexcept { return table_.n_docs; }

    nb::ndarray<nb::numpy, std::int32_t> rank(const F64Array& scores, std::int32_t selection) {
        prepare(scores);
        std::vector<DocId> out(static_cast<std::size_t>(table_.n_docs));
        ranking::rank_full(keys_, out, static_cast<ranking::Selection>(selection));
        return to_numpy_i32(std::vector<std::int32_t>(out.begin(), out.end()));
    }

    nb::ndarray<nb::numpy, std::int32_t> top_k(const F64Array& scores,
                                               std::int32_t m,
                                               std::int32_t selection) {
        prepare(scores);
        const auto take = std::min<std::size_t>(static_cast<std::size_t>(std::max(m, 0)),
                                                keys_.size());
        std::vector<DocId> out(take);
        ranking::select_top(keys_, take, out, static_cast<ranking::Selection>(selection));
        return to_numpy_i32(std::vector<std::int32_t>(out.begin(), out.end()));
    }

  private:
    void prepare(const F64Array& scores) {
        if (static_cast<std::int32_t>(scores.shape(0)) != table_.n_docs) {
            throw std::invalid_argument("score count does not match the ranker's document count");
        }
        const std::span<const Real> s(scores.data(), scores.shape(0));
        // G3 requires this re-check at the boundary: a NaN in a sort key is
        // undefined behaviour in std::sort, a real out-of-bounds write, and
        // this is the last line of defence against an arbitrary caller.
        if (!ranking::all_finite(s)) {
            throw std::invalid_argument("scores must all be finite");
        }
        keys_.resize(static_cast<std::size_t>(table_.n_docs));
        ranking::build_keys(s, table_, priority_, keys_);
    }

    ranking::RankTable table_;
    std::vector<std::int32_t> priority_;
    std::vector<ranking::SortKey> keys_;
};

}  // namespace

NB_MODULE(_tfidf_native, m) {
    m.doc() = "Native (C++20) evaluator for the tfidf-stability pipeline.";

    m.attr("__version__") = tfidf::build::kVersion;
    m.attr("__abi__") = kAbi;

    // --- provenance ----------------------------------------------------------
    m.def(
        "build_info",
        [] {
            nb::dict d;
            d["version"] = tfidf::build::kVersion;
            d["git_sha"] = tfidf::build::kGitSha;
            d["compiler_id"] = tfidf::build::kCompilerId;
            d["compiler_ver"] = tfidf::build::kCompilerVer;
            d["build_type"] = tfidf::build::kBuildType;
            d["numeric_flags"] = tfidf::build::kNumericFlags;
            d["system"] = tfidf::build::kSystem;
            d["processor"] = tfidf::build::kProcessor;
            d["arch_tune"] = tfidf::build::kArchTune;
            d["fast_math"] = tfidf::build::kFastMath;
            d["reproducible"] = tfidf::build::kReproducible;
            return d;
        },
        "Compiler, flags and revision that produced this binary. Embedded verbatim\n"
        "into every run manifest so any published number is traceable to its build.");

    // --- floating-point environment -----------------------------------------
    m.def("fp_selftest", [] { return tfidf::fp::selftest(); },
          "Probe the live floating-point environment. Returns a bitmask; 0 means ok.");
    m.def("fp_describe", [](std::uint32_t f) { return tfidf::fp::describe(f); }, nb::arg("flags"),
          "Human-readable rendering of an fp_selftest() bitmask.");
    m.def("fp_restore_subnormals", [] { return tfidf::fp::restore_subnormals(); },
          "Clear MXCSR FTZ/DAZ if a third-party BLAS has set them. True if changed.");

    nb::dict failure;
    failure["ok"] = static_cast<std::uint32_t>(tfidf::fp::kOk);
    failure["constant_folding"] = static_cast<std::uint32_t>(tfidf::fp::kConstantFolding);
    failure["reassociation"] = static_cast<std::uint32_t>(tfidf::fp::kReassociation);
    failure["fma_contraction"] = static_cast<std::uint32_t>(tfidf::fp::kFmaContraction);
    failure["rounding_mode"] = static_cast<std::uint32_t>(tfidf::fp::kRoundingMode);
    failure["flush_to_zero"] = static_cast<std::uint32_t>(tfidf::fp::kFlushToZero);
    failure["denormals_are_zero"] = static_cast<std::uint32_t>(tfidf::fp::kDenormalsAreZero);
    m.attr("FP_FAILURE") = failure;

    // --- reduction primitives, exposed for the differential tests ------------
    m.def(
        "reduce_sum",
        [](const F64Array& a, std::int32_t policy) {
            return reduce::sum(a.data(), a.shape(0), checked_policy(policy));
        },
        nb::arg("values"), nb::arg("policy"),
        "Sum under the given reduction policy. Exposed so the Python and C++\n"
        "implementations of each policy can be compared bit for bit.");

    m.def(
        "dot",
        [](const I32Array& ai, const F64Array& av, const I32Array& bi, const F64Array& bv,
           std::int32_t dim, std::int32_t policy) {
            const SparseView a = checked_view(ai, av, dim, "a");
            const SparseView b = checked_view(bi, bv, dim, "b");
            return dot(a, b, checked_policy(policy));
        },
        nb::arg("a_indices"), nb::arg("a_values"), nb::arg("b_indices"), nb::arg("b_values"),
        nb::arg("dim"), nb::arg("policy"));

    m.def(
        "l2_norm",
        [](const I32Array& i, const F64Array& v, std::int32_t dim, std::int32_t policy) {
            const SparseView s = checked_view(i, v, dim, "vector");
            return l2_norm(s, checked_policy(policy));
        },
        nb::arg("indices"), nb::arg("values"), nb::arg("dim"), nb::arg("policy"));

    // --- the index -----------------------------------------------------------
    nb::class_<NativeIndex>(m, "NativeIndex",
                            "An immutable fitted corpus: CSR, inverted index and row norms.")
        .def(nb::init<const I64Array&, const I32Array&, const F64Array&, std::int32_t,
                      std::int32_t, std::int32_t>(),
             nb::arg("indptr"), nb::arg("indices"), nb::arg("values"), nb::arg("n_docs"),
             nb::arg("n_terms"), nb::arg("reduction"))
        .def_prop_ro("n_documents", &NativeIndex::n_documents)
        .def_prop_ro("n_features", &NativeIndex::n_features)
        .def_prop_ro("nnz", &NativeIndex::nnz)
        .def_prop_ro("reduction", &NativeIndex::reduction)
        // `move` rather than the default `reference_internal`: to_numpy hands
        // back an array that already owns its buffer through a capsule, and
        // nanobind refuses to re-parent an ndarray that has an owner.
        .def_prop_ro(
            "norms",
            [](const NativeIndex& self) { return to_numpy(std::vector<double>(self.norms())); },
            nb::rv_policy::move)
        .def("df", &NativeIndex::df, nb::arg("term"))
        .def("score", &NativeIndex::score, nb::arg("q_indices"), nb::arg("q_values"),
             nb::arg("algorithm") = 0, nb::rv_policy::move,
             "Score one query against every document. algorithm: 0 = TAAT, 1 = DAAT.");

    // Mirrors of the C++ enumerations, so Python never hard-codes the integers.
    nb::dict reductions;
    reductions["naive"] = static_cast<std::int32_t>(Reduction::Naive);
    reductions["neumaier"] = static_cast<std::int32_t>(Reduction::Neumaier);
    reductions["pairwise"] = static_cast<std::int32_t>(Reduction::Pairwise);
    reductions["exact"] = static_cast<std::int32_t>(Reduction::Exact);
    m.attr("REDUCTION") = reductions;

    nb::dict algorithms;
    algorithms["taat"] = static_cast<std::int32_t>(ScoringAlgorithm::Taat);
    algorithms["daat"] = static_cast<std::int32_t>(ScoringAlgorithm::Daat);
    m.attr("ALGORITHM") = algorithms;

    // --- ranking -------------------------------------------------------------
    nb::class_<NativeRanker>(m, "NativeRanker",
                             "A corpus's tie-break ranks plus one operator's priority.")
        .def(nb::init<const I32Array&, const I32Array&, const I32Array&, std::int32_t>(),
             nb::arg("ranks"), nb::arg("id_ranks"), nb::arg("priority"), nb::arg("n_attrs"))
        .def_prop_ro("n_documents", &NativeRanker::n_documents)
        .def("rank", &NativeRanker::rank, nb::arg("scores"), nb::arg("selection") = 0,
             nb::rv_policy::move, "Rank every document; returns the permutation.")
        .def("top_k", &NativeRanker::top_k, nb::arg("scores"), nb::arg("m"),
             nb::arg("selection") = 3, nb::rv_policy::move);

    m.def(
        "sorted_scores_desc",
        [](const F64Array& scores) {
            return to_numpy(ranking::sorted_scores_desc(checked_scores(scores)));
        },
        nb::arg("scores"), nb::rv_policy::move);

    m.def(
        "boundary_margin",
        [](const F64Array& sorted_scores, std::int32_t k) {
            const auto mg = ranking::boundary_margin(checked_scores(sorted_scores), k);
            return nb::make_tuple(mg.value, mg.defined, mg.k_effective);
        },
        nb::arg("sorted_scores"), nb::arg("k"),
        "Returns (value, defined, k_effective); value is NaN when undefined.");

    m.def(
        "min_adjacent_margin_top",
        [](const F64Array& sorted_scores, std::int32_t k) {
            const auto mg = ranking::min_adjacent_margin_top(checked_scores(sorted_scores), k);
            return nb::make_tuple(mg.value, mg.defined, mg.k_effective);
        },
        nb::arg("sorted_scores"), nb::arg("k"));

    m.def(
        "tie_ball_interval",
        [](const F64Array& sorted_scores, std::int32_t j, double tau) {
            if (j < 0 || j >= static_cast<std::int32_t>(sorted_scores.shape(0))) {
                throw std::out_of_range("rank index out of range");
            }
            checked_tau(tau);
            const auto [lo, hi] =
                ranking::tie_ball_interval(checked_scores(sorted_scores), j, tau);
            return nb::make_tuple(lo, hi);
        },
        nb::arg("sorted_scores"), nb::arg("j"), nb::arg("tau"));

    m.def(
        "tie_chains",
        [](const F64Array& sorted_scores, double tau) {
            std::vector<std::int32_t> flat;
            for (const auto& [lo, hi] :
                 ranking::tie_chains(checked_scores(sorted_scores), checked_tau(tau))) {
                flat.push_back(lo);
                flat.push_back(hi);
            }
            return to_numpy_i32(std::move(flat));
        },
        nb::arg("sorted_scores"), nb::arg("tau"), nb::rv_policy::move,
        "Flattened [lo, hi) pairs.");

    m.def(
        "tie_cliques",
        [](const F64Array& sorted_scores, double tau) {
            std::vector<std::int32_t> flat;
            for (const auto& [lo, hi] :
                 ranking::tie_cliques(checked_scores(sorted_scores), checked_tau(tau))) {
                flat.push_back(lo);
                flat.push_back(hi);
            }
            return to_numpy_i32(std::move(flat));
        },
        nb::arg("sorted_scores"), nb::arg("tau"), nb::rv_policy::move);

    m.def(
        "chain_inflation_ratio",
        [](const F64Array& sorted_scores, double tau) {
            return ranking::chain_inflation_ratio(checked_scores(sorted_scores),
                                                  checked_tau(tau));
        },
        nb::arg("sorted_scores"), nb::arg("tau"));

    // --- ordering distances --------------------------------------------------
    m.def(
        "inversion_count",
        [](const I32Array& sequence) {
            return ranking::inversion_count({sequence.data(), sequence.shape(0)});
        },
        nb::arg("sequence"), "Pairs i < j with sequence[i] > sequence[j], by merge sort.");

    m.def(
        "kendall_tau_distance",
        [](const I32Array& a, const I32Array& b) {
            return ranking::kendall_tau_distance(as_ids(a), as_ids(b));
        },
        nb::arg("a"), nb::arg("b"),
        "Normalised Kendall tau between two orderings of the same set.\n"
        "Raises ValueError when the sets differ -- that is the signal that\n"
        "kendall_fks is the function wanted (spec_addenda G2).");

    m.def(
        "fks_max", [](std::int32_t k, double penalty) { return ranking::fks_max(k, penalty); },
        nb::arg("k"), nb::arg("penalty") = ranking::kFksPenalty,
        "The FKS ceiling k^2 + p*k*(k-1), attained by disjoint lists. 1.0 at k = 1.");

    m.def(
        "kendall_fks",
        [](const I32Array& a, const I32Array& b, double penalty, bool normalise) {
            return ranking::kendall_fks(as_ids(a), as_ids(b), penalty, normalise);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("penalty") = ranking::kFksPenalty,
        nb::arg("normalise") = true,
        "Generalised Kendall distance between two top-k lists that may rank\n"
        "different sets. A near-metric, not a metric (spec_addenda G2).");

    m.def(
        "top_k_disagreement",
        [](const I32Array& a, const I32Array& b) {
            return ranking::top_k_disagreement(as_ids(a), as_ids(b));
        },
        nb::arg("a"), nb::arg("b"));

    m.def(
        "jaccard_distance",
        [](const I32Array& a, const I32Array& b) {
            return ranking::jaccard_distance(as_ids(a), as_ids(b));
        },
        nb::arg("a"), nb::arg("b"));

    m.def(
        "compare_top_k",
        [](const I32Array& a, const I32Array& b, std::int32_t k) {
            // Refused rather than clamped: a negative k is a Python slice
            // counting from the end, a different question, and answering the
            // wrong one silently is what this boundary exists to prevent.
            if (k < 0) {
                throw std::invalid_argument("k must be non-negative");
            }
            const auto c = ranking::compare_top_k(as_ids(a), as_ids(b), k);
            return nb::make_tuple(c.k, c.sets_differ, c.fks, c.kendall_intersection,
                                  c.intersection_size, c.jaccard, c.swapped);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("k"),
        "Returns (k, sets_differ, fks, kendall_intersection, intersection_size,\n"
        "jaccard, swapped). kendall_intersection is NaN -- undefined, never a\n"
        "value -- when fewer than two elements are shared.");

    m.attr("FKS_PENALTY") = ranking::kFksPenalty;

    nb::dict selections;
    selections["full_sort"] = static_cast<std::int32_t>(ranking::Selection::FullSort);
    selections["stable_sort"] = static_cast<std::int32_t>(ranking::Selection::StableSort);
    selections["partial_sort"] = static_cast<std::int32_t>(ranking::Selection::PartialSort);
    selections["nth_element"] = static_cast<std::int32_t>(ranking::Selection::NthElement);
    selections["bounded_heap"] = static_cast<std::int32_t>(ranking::Selection::BoundedHeap);
    m.attr("SELECTION") = selections;
}
