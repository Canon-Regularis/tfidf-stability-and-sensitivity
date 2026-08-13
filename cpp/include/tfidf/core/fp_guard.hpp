// =============================================================================
// fp_guard.hpp -- refuse to produce numbers from an untrustworthy build.
//
// This project publishes bit-reproducibility claims. A single stray compiler
// flag silently invalidates all of them, and the failure mode is the worst kind:
// the code still runs, still looks right, and quietly returns different digits.
//
// Three layers of defence:
//   Layer 1  compile time -- #error on flags known to break IEEE-754 semantics
//   Layer 2  build time   -- the resolved flag string is baked into the binary
//                            (build_config.hpp) and recorded in every manifest
//   Layer 3  run time     -- fp_selftest() probes for behaviour the flags were
//                            supposed to prevent, catching cases the compiler
//                            macros miss (notably FTZ/DAZ set by a third-party
//                            library at load time -- see below)
//
// Layer 3 is not paranoia. numpy's BLAS (MKL, and some OpenBLAS builds) sets
// flush-to-zero / denormals-are-zero process-wide when it loads. In a project
// whose entire subject is near-tie margins, silently flushing subnormal score
// differences to zero would corrupt precisely the phenomenon under study.
// =============================================================================
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
// Layer 1 -- compile time
// -----------------------------------------------------------------------------
#if defined(__FAST_MATH__) && !defined(TFIDF_ALLOW_FAST_MATH)
#  error "tfidf: built with -ffast-math. Numerical results would not be trustworthy. \
Define TFIDF_ALLOW_FAST_MATH only in the CI job that deliberately proves this guard fires."
#endif

#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ && !defined(TFIDF_ALLOW_FAST_MATH)
#  error "tfidf: -ffinite-math-only is enabled; NaN/Inf handling would be undefined."
#endif

#if defined(_M_FP_FAST) && !defined(TFIDF_ALLOW_FAST_MATH)
#  error "tfidf: MSVC /fp:fast is enabled. Use /fp:precise /fp:contract-."
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
    kFlushToZero        = 1u << 4,  ///< subnormals flushed -- corrupts tiny margins
    kDenormalsAreZero   = 1u << 5,  ///< subnormal inputs treated as zero
};

/// Probe the *actual* floating-point behaviour of this process, right now.
///
/// Cheap enough (a handful of arithmetic ops plus one `stmxcsr`) to call at
/// import and again at the head of every batch scoring call.
[[nodiscard]] inline std::uint32_t selftest() noexcept {
    std::uint32_t f = kOk;

    // Excess intermediate precision.
    //
    // The previous probe was `volatile double a = 0.1, b = 0.2; if (a + b ==
    // 0.3)`, and it could not fire under any configuration tested -- including
    // `-mfpmath=387 -fexcess-precision=fast`, which is precisely the hazard it
    // names, and `-ffast-math`. The reason is arithmetic, not optimisation: the
    // exact sum of the doubles 0.1 and 0.2 differs from the double 0.3 at every
    // precision, so the comparison is false whether or not intermediates are
    // widened. `kConstantFolding` was therefore a guard bit that could never be
    // set, and `describe()`'s branch for it was unreachable.
    //
    // FLT_EVAL_METHOD reports the property directly: 0 when each operation is
    // performed in its own type, 2 when everything is evaluated as long double.
    // Verified to read 0 under `-mfpmath=sse -fexcess-precision=standard` and 2
    // under `-mfpmath=387 -fexcess-precision=fast`.
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
    f |= kConstantFolding;
#endif

    // Reassociation would preserve the tiny addend instead of losing it.
    {
        volatile double a = 1.0, b = 1e-17;
        if ((a + b) - a != 0.0) f |= kReassociation;
    }

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
    // Portable fallback: if subnormals survive arithmetic, FTZ cannot be on.
    {
        volatile double tiny = std::numeric_limits<double>::denorm_min();
        if (tiny > 0.0 && tiny * 2.0 == 0.0) f |= kFlushToZero;
    }
#endif

    return f;
}

/// Clear FTZ/DAZ if a third-party library (typically a BLAS) has set them.
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
    if (f == kOk)                 return "ok";
    if (f & kFmaContraction)      return "FMA contraction is active (need -ffp-contract=off / /fp:contract-)";
    if (f & kFlushToZero)         return "flush-to-zero is set (a BLAS may have set MXCSR.FTZ)";
    if (f & kDenormalsAreZero)    return "denormals-are-zero is set (a BLAS may have set MXCSR.DAZ)";
    if (f & kReassociation)       return "the compiler is reassociating floating-point sums";
    if (f & kConstantFolding)     return "constants are folded at extended precision";
    if (f & kRoundingMode)        return "the rounding mode is not round-to-nearest-even";
    return "unknown floating-point environment failure";
}

}  // namespace tfidf::fp
