// fp_guard.hpp: refuse to produce numbers from an untrustworthy build.
//
// Layer 1, compile time: #error on flags that break IEEE-754 semantics.
// Layer 2, build time: build_config.hpp compiles in the flags; every manifest records them.
// Layer 3, run time: selftest(). Rationale: docs/engineering_notes.md#fp-guard-layers
#pragma once

#include <cfenv>
#include <cfloat>
#include <cstdint>
#include <limits>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  define TFIDF_HAS_MXCSR 1
#  include <xmmintrin.h>
#  include <pmmintrin.h>
#else
#  define TFIDF_HAS_MXCSR 0
#endif

// -----------------------------------------------------------------------------
// Layer 1: compile time
// -----------------------------------------------------------------------------
#if defined(__FAST_MATH__) && !defined(TFIDF_ALLOW_FAST_MATH)
#  error "tfidf: built with -ffast-math. Numerical results would not be trustworthy. \
Define TFIDF_ALLOW_FAST_MATH only in the CI job that deliberately proves this guard fires."
#endif

#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ && !defined(TFIDF_ALLOW_FAST_MATH)
#  error "tfidf: -ffinite-math-only is enabled; NaN/Inf handling would be undefined."
#endif

// -fno-signed-zeros licenses the compiler to treat -0.0 and +0.0 as
// interchangeable. Every score here is compared on its raw bit pattern, so the
// two must stay distinct. The flag does not define __FAST_MATH__; GCC and Clang
// define __NO_SIGNED_ZEROS__ for it instead.
#if defined(__NO_SIGNED_ZEROS__) && !defined(TFIDF_ALLOW_FAST_MATH)
#  error "tfidf: -fno-signed-zeros is enabled; +0.0 and -0.0 would be interchangeable."
#endif

// MSVC has no negative form of /fp:contract: `/fp:contract-` draws `command
// line warning D9002: ignoring unknown option` and contraction continues. The
// message therefore names /fp:strict, the documented way to forbid it. See
// cpp/cmake/NumericsFlags.cmake.
#if defined(_M_FP_FAST) && !defined(TFIDF_ALLOW_FAST_MATH)
#  error "tfidf: MSVC /fp:fast is enabled. Use /fp:strict."
#endif

namespace tfidf::fp {

static_assert(std::numeric_limits<double>::is_iec559,
              "tfidf: IEEE-754 binary64 is required for every published result.");
static_assert(std::numeric_limits<double>::round_style == std::round_to_nearest,
              "tfidf: round-to-nearest-even is required.");
static_assert(sizeof(double) == 8, "tfidf: double must be exactly 64 bits.");

/// Bit flags returned by selftest(); zero means a trustworthy environment.
enum Failure : std::uint32_t {
    kOk                 = 0u,
    kConstantFolding    = 1u << 0,  ///< literals folded at higher precision
    kReassociation      = 1u << 1,  ///< (a+b)+c rewritten as a+(b+c)
    kFmaContraction     = 1u << 2,  ///< a*b+c contracted to fma(a,b,c)
    kRoundingMode       = 1u << 3,  ///< rounding mode is not to-nearest
    kFlushToZero        = 1u << 4,  ///< subnormals flushed; corrupts tiny margins
    kDenormalsAreZero   = 1u << 5,  ///< subnormal inputs treated as zero
};

/// Whether a subnormal survived being loaded, and being produced.
struct SubnormalSurvival {
    bool inputs;   ///< false when subnormal operands are read as zero (DAZ-like)
    bool results;  ///< false when a subnormal result collapses to zero (FTZ-like)
};

/// Probe whether subnormals survive this process, on both sides.
///
/// The only detection on targets without MXCSR, such as aarch64. Each half is
/// computed independently, so a combined FTZ+DAZ mode is reported.
/// Rationale: docs/engineering_notes.md#subnormal-survival-probe
[[nodiscard]] inline SubnormalSurvival subnormals_survive() noexcept {
    // `denorm_min()` is the smallest positive subnormal, or `min()` where the
    // type has no subnormals at all -- non-zero either way, so a platform
    // without them is not misreported as flushing.
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    const bool inputs = (tiny != 0.0);
    // Doubling the smallest subnormal gives another subnormal, so a flushed
    // result collapses to zero. Computed unconditionally: guarding it behind
    // `inputs` would hide the combined FTZ+DAZ mode.
    volatile double doubled = tiny * 2.0;
    const bool results = (doubled != 0.0);
    return SubnormalSurvival{inputs, results};
}

/// Probe the live floating-point behaviour of this process.
///
/// A handful of arithmetic ops plus one `stmxcsr`, so it is cheap enough to run
/// at import and again at the head of every batch scoring call.
[[nodiscard]] inline std::uint32_t selftest() noexcept {
    std::uint32_t f = kOk;

    // Excess intermediate precision: FLT_EVAL_METHOD is 0 when each operation is
    // performed in its own type, 2 when everything is evaluated as long double.
    // Rationale: docs/engineering_notes.md#excess-precision-detection
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
    f |= kConstantFolding;
#endif

    // Reassociation is read off the compiler, not probed for: an arithmetic
    // probe needs `volatile`, which makes each read an observable access and
    // forbids the very fold it looks for. GCC and Clang define
    // __ASSOCIATIVE_MATH__ whenever -fassociative-math is in effect.
#if defined(__ASSOCIATIVE_MATH__)
    f |= kReassociation;
#endif

    // FMA contraction: with x = y = 1 + 2^-27 and z = -1,
    //   unfused  x*y + z  ==  2^-26                (the product is rounded first)
    //   fused    fma(x,y,z) == 2^-26 + 2^-54       (single rounding keeps the tail)
    {
        volatile double x = 1.0 + 0x1p-27, y = 1.0 + 0x1p-27, z = -1.0;
        const double r = x * y + z;
        if (r != 0x1p-26) f |= kFmaContraction;
    }

    if (std::fegetround() != FE_TONEAREST) f |= kRoundingMode;

#if TFIDF_HAS_MXCSR
    {
        const unsigned int mxcsr = _mm_getcsr();
        if (mxcsr & 0x8000u) f |= kFlushToZero;       // FTZ  (bit 15)
        if (mxcsr & 0x0040u) f |= kDenormalsAreZero;  // DAZ  (bit 6)
    }
#else
    // Portable fallback. See `subnormals_survive()` for why both halves are
    // needed and why neither may guard the other.
    {
        const SubnormalSurvival s = subnormals_survive();
        // Inputs zeroed is DAZ's defining behaviour; results flushed is FTZ's.
        // AArch64 has one bit doing both, so it reports as DAZ, which is
        // accurate -- inputs really are being zeroed -- and enough to warn.
        if (!s.inputs) f |= kDenormalsAreZero;
        else if (!s.results) f |= kFlushToZero;
    }
#endif

    return f;
}

/// Clear FTZ/DAZ if a third-party library (typically a BLAS) has set them.
///
/// Returns true if a change was made, so the caller can record it.
inline bool restore_subnormals() noexcept {
#if TFIDF_HAS_MXCSR
    const unsigned int before = _mm_getcsr();
    const unsigned int after  = before & ~0x8040u;
    if (before != after) {
        _mm_setcsr(after);
        return true;
    }
#endif
    return false;
}

/// Human-readable rendering of a selftest() result, for error messages.
[[nodiscard]] inline const char* describe(std::uint32_t f) noexcept {
    if (f == kOk)
        return "ok";
    if (f & kFmaContraction)
        return "FMA contraction is active (need -ffp-contract=off, or /fp:strict on MSVC)";
    if (f & kFlushToZero)
        return "flush-to-zero is set (a BLAS may have set MXCSR.FTZ)";
    if (f & kDenormalsAreZero)
        return "denormals-are-zero is set (a BLAS may have set MXCSR.DAZ)";
    if (f & kReassociation)
        return "the compiler is reassociating floating-point sums";
    if (f & kConstantFolding)
        return "constants are folded at extended precision";
    if (f & kRoundingMode)
        return "the rounding mode is not round-to-nearest-even";
    return "unknown floating-point environment failure";
}

}  // namespace tfidf::fp
