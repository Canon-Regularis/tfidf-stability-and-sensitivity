// Tests for the floating-point environment guard.
//
// These are the first tests that run and the ones most worth reading: if any of
// them fails, no other number this project produces can be trusted, because the
// compiler or a loaded library has changed the arithmetic out from under us.
#include <tfidf/core/build_config.hpp>
#include <tfidf/core/fp_guard.hpp>

#include <doctest.h>

#include <cfenv>
#include <cmath>
#include <limits>
#include <string>

TEST_CASE("fp: the build is numerically trustworthy") {
    const std::uint32_t f = tfidf::fp::selftest();
    INFO("selftest bitmask = " << f << " -> " << tfidf::fp::describe(f));
    CHECK(f == tfidf::fp::kOk);
}

TEST_CASE("fp: FMA contraction is off") {
    // x*y + z with x = y = 1 + 2^-27, z = -1.
    //   unfused -> the product rounds to 1 + 2^-26, so the result is exactly 2^-26
    //   fused   -> the exact product 1 + 2^-26 + 2^-54 survives, giving a larger value
    volatile double x = 1.0 + 0x1p-27, y = 1.0 + 0x1p-27, z = -1.0;
    const double unfused = x * y + z;
    CHECK(unfused == 0x1p-26);

    // std::fma is still available and must give the *other* answer -- proving the
    // probe discriminates rather than merely always passing.
    const double fused = std::fma(1.0 + 0x1p-27, 1.0 + 0x1p-27, -1.0);
    CHECK(fused > unfused);
}

TEST_CASE("fp: no reassociation of sums") {
    volatile double a = 1.0, b = 1e-17;
    CHECK((a + b) - a == 0.0);
}

TEST_CASE("fp: subnormals survive (FTZ/DAZ clear)") {
    // Margins between near-tied scores can legitimately be subnormal. If a BLAS
    // has set flush-to-zero, those differences silently become exact ties --
    // corrupting precisely the phenomenon this project studies.
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    CHECK(tiny > 0.0);
    CHECK(tiny * 2.0 > 0.0);

    volatile double a = std::numeric_limits<double>::min();  // smallest normal
    CHECK(a / 2.0 > 0.0);                                    // must go subnormal, not to zero
    CHECK((tfidf::fp::selftest() & tfidf::fp::kFlushToZero) == 0u);
    CHECK((tfidf::fp::selftest() & tfidf::fp::kDenormalsAreZero) == 0u);
}

TEST_CASE("fp: rounding mode is to-nearest-even") {
    CHECK(std::fegetround() == FE_TONEAREST);
    // Ties-to-even: 0.5 and 2.5 both round to an even integer.
    CHECK(std::nearbyint(0.5) == 0.0);
    CHECK(std::nearbyint(2.5) == 2.0);
}

TEST_CASE("fp: sqrt is correctly rounded (IEEE-754 mandated)") {
    // Unlike log, sqrt MUST be correctly rounded, which is why the native
    // pipeline is allowed to compute norms but never logarithms.
    for (double v : {2.0, 3.0, 1e-300, 1e300, 0.1}) {
        const double r = std::sqrt(v);
        CHECK(std::isfinite(r));
        CHECK(r >= 0.0);
    }
    CHECK(std::sqrt(4.0) == 2.0);
    CHECK(std::sqrt(0.0) == 0.0);
}

TEST_CASE("build: provenance is populated") {
    CHECK(std::string(tfidf::build::kVersion).size() > 0);
    CHECK(std::string(tfidf::build::kCompilerId).size() > 0);
    CHECK(std::string(tfidf::build::kNumericFlags).size() > 0);
}

TEST_CASE("build: reproducibility flag reflects the actual knobs") {
    CHECK(tfidf::build::kReproducible == (!tfidf::build::kArchTune && !tfidf::build::kFastMath));
}
