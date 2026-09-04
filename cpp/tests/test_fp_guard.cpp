// Tests for the floating-point environment guard.
//
// These run first. If any fails, the compiler or a loaded library has changed
// the arithmetic underneath us and no other number here can be trusted.
#include <tfidf/core/build_config.hpp>
#include <tfidf/core/fp_guard.hpp>

#include <doctest.h>

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

TEST_CASE("fp: the build is numerically trustworthy") {
    const std::uint32_t f = tfidf::fp::selftest();
    INFO("selftest bitmask = " << f << " -> " << tfidf::fp::describe(f));
    CHECK(f == tfidf::fp::kOk);
}

TEST_CASE("fp: FMA contraction is off") {
    // x*y + z with x = y = 1 + 2^-27, z = -1.
    //   unfused -> the product rounds to 1 + 2^-26, so the result is 2^-26
    //   fused   -> the exact product 1 + 2^-26 + 2^-54 survives, so it is larger
    volatile double x = 1.0 + 0x1p-27, y = 1.0 + 0x1p-27, z = -1.0;
    const double unfused = x * y + z;
    CHECK(unfused == 0x1p-26);

    // std::fma is still available and must give the other answer, so the probe
    // discriminates rather than always passing.
    const double fused = std::fma(1.0 + 0x1p-27, 1.0 + 0x1p-27, -1.0);
    CHECK(fused > unfused);
}

TEST_CASE("fp: no reassociation of sums") {
    // Asserted against the compiler's own report, not against arithmetic. An
    // arithmetic probe needs `volatile`, which makes each read an observable
    // access and forbids the very fold it is written to detect. `selftest()`
    // reads __ASSOCIATIVE_MATH__ instead, and this asserts that bit.
    const std::uint32_t f = tfidf::fp::selftest();
    INFO("selftest bitmask = " << f << " -> " << tfidf::fp::describe(f));
    CHECK((f & tfidf::fp::kReassociation) == 0u);
}

TEST_CASE("fp: subnormals survive (FTZ/DAZ clear)") {
    // Margins between near-tied scores can legitimately be subnormal. Under a
    // BLAS-set flush-to-zero those differences become exact ties, which is the
    // phenomenon this project studies.
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    CHECK(tiny > 0.0);
    CHECK(tiny * 2.0 > 0.0);

    volatile double a = std::numeric_limits<double>::min();  // smallest normal
    CHECK(a / 2.0 > 0.0);                                    // subnormal rather than zero
    CHECK((tfidf::fp::selftest() & tfidf::fp::kFlushToZero) == 0u);
    CHECK((tfidf::fp::selftest() & tfidf::fp::kDenormalsAreZero) == 0u);
}

// The portable probe, exercised under the modes it exists to catch.
//
// `subnormals_survive()` is the ONLY flush detection on targets without MXCSR
// (aarch64, shipped as a macOS wheel). Those targets cannot set the modes from
// a test, so the probe went unexercised there and was wrong: it asked only
// whether a subnormal RESULT collapses, guarded by whether a subnormal INPUT
// survives, so a mode zeroing both short-circuited to "clean".
//
// x86 can set FTZ and DAZ independently, so the aarch64-only code path is
// tested here, on the one architecture able to produce every combination.
// AArch64's FPCR.FZ is the FTZ+DAZ row.
#if TFIDF_HAS_MXCSR
TEST_CASE("fp: the portable probe detects every flushing mode") {
    const unsigned int original = _mm_getcsr();
    const unsigned int clean = original & ~0x8040u;

    SUBCASE("clean: no false positive") {
        _mm_setcsr(clean);
        const auto s = tfidf::fp::subnormals_survive();
        CHECK(s.inputs);
        CHECK(s.results);
        _mm_setcsr(original);
    }

    SUBCASE("FTZ only: results are flushed") {
        _mm_setcsr(clean | 0x8000u);
        const auto s = tfidf::fp::subnormals_survive();
        CHECK(s.inputs);        // inputs still load
        CHECK_FALSE(s.results); // but a subnormal result collapses
        _mm_setcsr(original);
    }

    SUBCASE("DAZ only: inputs are zeroed, which the old probe missed") {
        _mm_setcsr(clean | 0x0040u);
        const auto s = tfidf::fp::subnormals_survive();
        CHECK_FALSE(s.inputs);
        _mm_setcsr(original);
    }

    SUBCASE("FTZ and DAZ together, the mode a BLAS sets and AArch64's FZ bit") {
        _mm_setcsr(clean | 0x8040u);
        const auto s = tfidf::fp::subnormals_survive();
        CHECK_FALSE(s.inputs);
        _mm_setcsr(original);
    }

    // Restored, or every later test in this binary runs under a flushing mode.
    CHECK((_mm_getcsr() & 0x8040u) == (original & 0x8040u));
}

TEST_CASE("fp: selftest raises a flag for every flushing mode") {
    const unsigned int original = _mm_getcsr();
    const unsigned int clean = original & ~0x8040u;
    const std::uint32_t flushing = tfidf::fp::kFlushToZero | tfidf::fp::kDenormalsAreZero;

    for (const unsigned int bits : {0x8000u, 0x0040u, 0x8040u}) {
        _mm_setcsr(clean | bits);
        const std::uint32_t f = tfidf::fp::selftest();
        INFO("MXCSR flush bits = " << bits << " -> " << tfidf::fp::describe(f));
        CHECK((f & flushing) != 0u);
    }

    _mm_setcsr(original);
    CHECK((tfidf::fp::selftest() & flushing) == 0u);
}
#endif

TEST_CASE("fp: rounding mode is to-nearest-even") {
    CHECK(std::fegetround() == FE_TONEAREST);
    // Ties-to-even: 0.5 and 2.5 both round to an even integer.
    CHECK(std::nearbyint(0.5) == 0.0);
    CHECK(std::nearbyint(2.5) == 2.0);
}

TEST_CASE("fp: sqrt is correctly rounded (IEEE-754 mandated)") {
    // IEEE-754 mandates correct rounding for sqrt and not for log, which is why
    // the native pipeline computes norms and never logarithms.
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

TEST_CASE("fp: every failure flag is its own single bit") {
    // The flags are shift expressions, and nothing checked the shift amounts.
    // Changing any one of them makes two flags equal: `kConstantFolding` at
    // `1u << 1` collides with `kReassociation`, and `selftest()` then reports a
    // failure under the wrong name while `f & kConstantFolding` becomes true
    // for a build that only reassociates. Three such mutations survived the
    // suite.
    const std::pair<const char*, std::uint32_t> flags[] = {
        {"kConstantFolding", tfidf::fp::kConstantFolding},
        {"kReassociation", tfidf::fp::kReassociation},
        {"kFmaContraction", tfidf::fp::kFmaContraction},
        {"kRoundingMode", tfidf::fp::kRoundingMode},
        {"kFlushToZero", tfidf::fp::kFlushToZero},
        {"kDenormalsAreZero", tfidf::fp::kDenormalsAreZero},
    };

    CHECK(tfidf::fp::kOk == 0u);

    std::uint32_t seen = 0u;
    for (const auto& [name, bit] : flags) {
        INFO("flag " << name);
        CHECK(bit != 0u);
        // A single bit, so a mask test names exactly one failure.
        CHECK((bit & (bit - 1u)) == 0u);
        // And one no other flag has already claimed.
        CHECK((seen & bit) == 0u);
        seen |= bit;
    }
    // Six flags, so six bits: the loop above would also pass if two names
    // referred to the same constant and the check ran over five distinct bits.
    CHECK(std::popcount(seen) == 6);
}

TEST_CASE("fp: describe names the failure it was handed") {
    // `describe` is what an operator reads when a build is refused, and it was
    // untested. Its first arm is `f == kOk`; inverted, every broken environment
    // is described as "ok" and the only failure path in the shipped binary
    // reports success.
    CHECK(std::string(tfidf::fp::describe(tfidf::fp::kOk)) == "ok");

    const std::uint32_t flags[] = {
        tfidf::fp::kConstantFolding, tfidf::fp::kReassociation,
        tfidf::fp::kFmaContraction,  tfidf::fp::kRoundingMode,
        tfidf::fp::kFlushToZero,     tfidf::fp::kDenormalsAreZero,
    };
    std::vector<std::string> messages;
    for (const std::uint32_t f : flags) {
        const std::string message = tfidf::fp::describe(f);
        CHECK(message != "ok");
        CHECK_FALSE(message.empty());
        messages.push_back(message);
    }
    // Distinct, or two failures would be indistinguishable in the one place
    // they are reported.
    std::sort(messages.begin(), messages.end());
    CHECK(std::adjacent_find(messages.begin(), messages.end()) == messages.end());

    // An unknown bit still says something rather than falling off the end.
    CHECK(std::string(tfidf::fp::describe(1u << 31)) != "ok");
}

#if TFIDF_HAS_MXCSR
TEST_CASE("fp: restore_subnormals reports whether it actually changed anything") {
    // The return value is the caller's record that a third-party library had
    // set FTZ/DAZ. Reported the wrong way round, a clean run logs a correction
    // that never happened and a corrected run stays silent.
    const unsigned int original = _mm_getcsr();
    const unsigned int clean = original & ~0x8040u;

    _mm_setcsr(clean);
    CHECK_FALSE(tfidf::fp::restore_subnormals());  // nothing to do
    CHECK((_mm_getcsr() & 0x8040u) == 0u);

    _mm_setcsr(clean | 0x8040u);
    CHECK(tfidf::fp::restore_subnormals());  // and it says so when there was
    CHECK((_mm_getcsr() & 0x8040u) == 0u);   // having actually cleared them

    // A second call has nothing left to change.
    CHECK_FALSE(tfidf::fp::restore_subnormals());

    _mm_setcsr(original);
    CHECK((_mm_getcsr() & 0x8040u) == (original & 0x8040u));
}
#endif
