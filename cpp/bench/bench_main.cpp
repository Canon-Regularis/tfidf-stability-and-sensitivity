// Native microbenchmarks. Kernel-level costs, free of FFI overhead.
//
// Populated alongside the kernels themselves in Stage 2; for now this exists so
// the benchmark target compiles and the build graph is complete from Stage 0.
#define ANKERL_NANOBENCH_IMPLEMENT
#include <tfidf/core/build_config.hpp>
#include <tfidf/core/fp_guard.hpp>

#include <nanobench.h>

#include <cstdio>

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

    ankerl::nanobench::Bench().run("fp_selftest", [&] {
        ankerl::nanobench::doNotOptimizeAway(tfidf::fp::selftest());
    });
    return 0;
}
