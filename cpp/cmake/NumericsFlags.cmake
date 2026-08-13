# =============================================================================
# NumericsFlags.cmake -- THE single source of truth for floating-point policy.
#
# This project's subject matter is numerical stability. Every result it publishes
# is claimed to be bit-reproducible across compilers, optimisation levels and
# operating systems. That claim is only meaningful if the compiler is forbidden
# from rewriting our arithmetic, so the flags below are load-bearing, not
# stylistic. Do not add flags anywhere else.
#
# The three rules:
#   1. No contraction.   a*b+c must NOT become fma(a,b,c) -- a different rounding.
#   2. No reassociation. (a+b)+c must NOT become a+(b+c).
#   3. No excess precision. Intermediates are binary64, never x87 80-bit.
#
# Anything that violates these is confined to the explicitly non-reproducible
# `Fast` reduction policy, which is never used for published results.
# =============================================================================

add_library(tfidf_numerics_strict INTERFACE)
add_library(tfidf::numerics_strict ALIAS tfidf_numerics_strict)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  target_compile_options(tfidf_numerics_strict INTERFACE
    -ffp-contract=off                 # rule 1 -- the single most important flag
    -fno-fast-math
    -fno-associative-math             # rule 2
    -fno-reciprocal-math              # x/y must not become x*(1/y)
    -fno-unsafe-math-optimizations
    -fno-finite-math-only             # NaN/Inf semantics must be honoured
    # NOT -fno-signed-zeros. That is a *fast-math sub-option* -- it licenses the
    # compiler to ignore the sign of zero -- and it sat in this list of
    # disables for years reading like one of them. Measured: with it,
    # (-0.0) + 0.0 compiles to -0.0 (bits 8000000000000000); with
    # -fsigned-zeros it gives +0.0, which is what IEEE 754 requires. This
    # repository compares every score on its raw bit pattern and
    # ranking/margins.py reasons explicitly that -0.0 cannot occur, so the
    # relaxed form is exactly the wrong one. fp_guard.hpp cannot catch it
    # either: -fno-signed-zeros alone does not define __FAST_MATH__.
    -fsigned-zeros
    -fexcess-precision=standard       # rule 3
  )
  # x87 carries 80-bit intermediates; SSE2 is exactly binary64. Irrelevant on
  # x86-64 (SSE2 is the default ABI) but decisive on 32-bit hosts.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mfpmath=sse" TFIDF_HAS_MFPMATH_SSE)
  if(TFIDF_HAS_MFPMATH_SSE)
    target_compile_options(tfidf_numerics_strict INTERFACE -mfpmath=sse)
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(tfidf_numerics_strict INTERFACE
    /fp:precise
    # Easy to miss and actively dangerous: since VS2022, MSVC permits FMA
    # contraction *under /fp:precise* by default. This flag turns it back off.
    /fp:contract-
    /fp:except-
  )
else()
  message(WARNING "tfidf: unrecognised compiler '${CMAKE_CXX_COMPILER_ID}'. "
                  "Floating-point reproducibility is NOT guaranteed.")
endif()

# -----------------------------------------------------------------------------
# Deliberate violation, used only to prove the runtime guards actually fire.
# `cpp/tests/test_fp_guard.cpp` expects a fast-math build to FAIL to compile;
# CI additionally builds a shared library this way and asserts the runtime
# self-test rejects it.
# -----------------------------------------------------------------------------
if(TFIDF_FAST_MATH)
  message(WARNING "tfidf: TFIDF_FAST_MATH=ON -- results from this build are NOT trustworthy.")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tfidf_numerics_strict INTERFACE -ffast-math)
  elseif(MSVC)
    target_compile_options(tfidf_numerics_strict INTERFACE /fp:fast)
  endif()
endif()

# -----------------------------------------------------------------------------
# Architecture tuning. OFF by default: -march=native changes vectorisation, can
# re-enable FMA contraction despite -ffp-contract=off in some GCC versions, and
# makes the binary machine-specific -- all three fatal to reproducibility.
# -----------------------------------------------------------------------------
if(TFIDF_ARCH_TUNE)
  message(WARNING "tfidf: TFIDF_ARCH_TUNE=ON -- binary is machine-specific and NOT reproducible.")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tfidf_numerics_strict INTERFACE -march=native)
  elseif(MSVC)
    target_compile_options(tfidf_numerics_strict INTERFACE /arch:AVX2)
  endif()
endif()

# -----------------------------------------------------------------------------
# Record the exact policy so build_config.hpp, and hence every run manifest,
# can report what the numbers were actually produced with.
# -----------------------------------------------------------------------------
get_target_property(TFIDF_NUMERIC_FLAGS tfidf_numerics_strict INTERFACE_COMPILE_OPTIONS)
string(REPLACE ";" " " TFIDF_NUMERIC_FLAGS "${TFIDF_NUMERIC_FLAGS}")
set(TFIDF_NUMERIC_FLAGS "${TFIDF_NUMERIC_FLAGS}" CACHE INTERNAL "resolved FP flags")
