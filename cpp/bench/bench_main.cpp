// Native microbenchmarks of the kernels, without FFI overhead.
//
// They measure the cost of each reduction policy. Timings compare only within one run of
// one binary, because the bench preset enables arch tuning.
// Rationale: docs/engineering_notes.md#native-bench-scope
#define ANKERL_NANOBENCH_IMPLEMENT
#include <tfidf/core/build_config.hpp>
#include <tfidf/core/fp_guard.hpp>
#include <tfidf/core/reduction.hpp>
#include <tfidf/similarity/scoring.hpp>
#include <tfidf/vectorisation/sparse.hpp>

#include <nanobench.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace {

using namespace tfidf;

/// A corpus in CSR, owning its buffers so the views below stay valid.
struct Corpus {
    std::vector<Offset> indptr;
    std::vector<TermId> indices;
    std::vector<Real> values;
    DocId n_rows = 0;
    TermId n_cols = 0;

    [[nodiscard]] CsrView view() const noexcept {
        return CsrView{indptr, indices, values, n_rows, n_cols};
    }
};

/// Builds a deterministic CSR corpus with `per_doc` sampled terms per row.
///
/// Terms are sorted and deduplicated. Weights span several orders of magnitude, so
/// compensated summation has rounding error to recover.
Corpus make_corpus(DocId docs, TermId vocab, std::size_t per_doc, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int32_t> term(0, vocab - 1);
    std::uniform_real_distribution<Real> mag(0.25, 4.0);
    std::uniform_int_distribution<int> scale(-6, 2);

    Corpus c;
    c.n_rows = docs;
    c.n_cols = vocab;
    c.indptr.reserve(static_cast<std::size_t>(docs) + 1);
    c.indptr.push_back(0);
    std::vector<TermId> row;
    for (DocId d = 0; d < docs; ++d) {
        row.clear();
        for (std::size_t k = 0; k < per_doc; ++k) {
            row.push_back(term(rng));
        }
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (const TermId t : row) {
            c.indices.push_back(t);
            c.values.push_back(mag(rng) * std::pow(10.0, scale(rng)));
        }
        c.indptr.push_back(static_cast<Offset>(c.indices.size()));
    }
    return c;
}

const char* policy_name(Reduction p) {
    switch (p) {
        case Reduction::Naive:
            return "Naive";
        case Reduction::Neumaier:
            return "Neumaier";
        case Reduction::Pairwise:
            return "Pairwise";
        case Reduction::Exact:
            return "Exact";
    }
    return "?";
}

constexpr Reduction kPolicies[] = {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                                   Reduction::Exact};

/// Minimum epoch time for every benchmark.
///
/// Millisecond-scale kernels need more than one iteration per epoch; with fewer,
/// nanobench reports them as unstable and results vary by tens of percent between runs.
constexpr auto kEpoch = std::chrono::milliseconds(20);

/// Untimed iterations run before measurement starts.
///
/// They keep cold caches and the first scratch allocation out of the kernel's cost.
constexpr std::size_t kWarmup = 10;

}  // namespace

int main() {
    const std::uint32_t f = tfidf::fp::selftest();
    if (f != tfidf::fp::kOk) {
        std::fprintf(stderr, "refusing to benchmark an untrustworthy build: %s\n",
                     tfidf::fp::describe(f));
        return 1;
    }
    std::printf("tfidf %s (%s %s, %s)\n", tfidf::build::kVersion, tfidf::build::kCompilerId,
                tfidf::build::kCompilerVer, tfidf::build::kBuildType);
    std::printf("reproducible build: %s\n", tfidf::build::kReproducible ? "yes" : "NO");

    ankerl::nanobench::Bench().minEpochTime(kEpoch).warmup(kWarmup).run("fp_selftest", [&] {
        ankerl::nanobench::doNotOptimizeAway(tfidf::fp::selftest());
    });

    // ------------------------------------------------------------------
    // The reduction policies on a flat array: the cost of compensation with
    // no sparse structure in the way.
    // ------------------------------------------------------------------
    {
        std::mt19937_64 rng(20260903);
        std::uniform_real_distribution<Real> mag(-1.0, 1.0);
        std::uniform_int_distribution<int> scale(-8, 4);
        std::vector<Real> flat(1u << 16);
        for (Real& x : flat) {
            x = mag(rng) * std::pow(10.0, scale(rng));
        }

        ankerl::nanobench::Bench bench;
        bench.title("reduce::sum, 65536 doubles")
            .relative(true)
            .batch(flat.size())
            .minEpochTime(kEpoch)
            .warmup(kWarmup);
        for (const Reduction p : kPolicies) {
            bench.run(policy_name(p), [&] {
                ankerl::nanobench::doNotOptimizeAway(reduce::sum(flat, p));
            });
        }
    }

    const Corpus corpus = make_corpus(2000, 20000, 120, 7);
    const CsrView csr = corpus.view();

    // ------------------------------------------------------------------
    // The sparse kernels. `dot` is the inner loop of DAAT scoring and of
    // every norm, so its policy cost is the one that multiplies out.
    // ------------------------------------------------------------------
    {
        const SparseView u = csr.row(0);
        const SparseView v = csr.row(1);
        ankerl::nanobench::Bench bench;
        bench.title("sparse kernels").relative(true).minEpochTime(kEpoch).warmup(kWarmup);
        for (const Reduction p : kPolicies) {
            bench.run(std::string("dot ") + policy_name(p), [&] {
                ankerl::nanobench::doNotOptimizeAway(dot(u, v, p));
            });
        }
        for (const Reduction p : kPolicies) {
            bench.run(std::string("l2_norm ") + policy_name(p), [&] {
                ankerl::nanobench::doNotOptimizeAway(l2_norm(u, p));
            });
        }
        bench.run("transpose 2000x20000", [&] {
            ankerl::nanobench::doNotOptimizeAway(transpose(csr));
        });
    }

    // ------------------------------------------------------------------
    // Scoring. TAAT and DAAT return identical bits, so any difference in
    // these timings is cost alone.
    // ------------------------------------------------------------------
    {
        const Csc index = transpose(csr);
        const SparseView query = csr.row(3);
        std::vector<Real> out(static_cast<std::size_t>(csr.n_rows));
        ScoringScratch scratch;
        scratch.reset(csr.n_rows);

        ankerl::nanobench::Bench bench;
        bench.title("score, 2000 docs x 20000 terms")
            .relative(true)
            .minEpochTime(kEpoch)
            .warmup(kWarmup);
        for (const Reduction p : kPolicies) {
            const std::vector<Real> norms = row_norms(csr, p);
            const Real qn = l2_norm(query, p);
            bench.run(std::string("taat ") + policy_name(p), [&] {
                score(query, csr, index, norms, qn, out, scratch, p, ScoringAlgorithm::Taat);
                ankerl::nanobench::doNotOptimizeAway(out[0]);
            });
            bench.run(std::string("daat ") + policy_name(p), [&] {
                score(query, csr, index, norms, qn, out, scratch, p, ScoringAlgorithm::Daat);
                ankerl::nanobench::doNotOptimizeAway(out[0]);
            });
        }
        bench.run("row_norms Naive", [&] {
            ankerl::nanobench::doNotOptimizeAway(row_norms(csr, Reduction::Naive));
        });
        bench.run("row_norms Exact", [&] {
            ankerl::nanobench::doNotOptimizeAway(row_norms(csr, Reduction::Exact));
        });
    }
    return 0;
}
