# =============================================================================
# NumericsFlags.cmake: the single source of floating-point policy.
# =============================================================================

# Published results are bit-reproducible across compilers, optimisation levels
# and operating systems only if the compiler may not rewrite the arithmetic. All
# four reduction policies (Naive, Neumaier, Pairwise, Exact) build under these
# flags. Floating-point flags belong in this file and nowhere else.

# Rules: 1. No contraction: a*b+c never becomes fma(a,b,c), which rounds once.
#        2. No reassociation: (a+b)+c never becomes a+(b+c).
#        3. No excess precision: intermediates are binary64, never x87 80-bit.
# The `fastmath-guard-check` preset breaks them to prove the guards reject it.

add_library(tfidf_numerics_strict INTERFACE)
add_library(tfidf::numerics_strict ALIAS tfidf_numerics_strict)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  target_compile_options(tfidf_numerics_strict INTERFACE
    -ffp-contract=off                 # rule 1, the one that matters most
    -fno-fast-math
    -fno-associative-math             # rule 2
    -fno-reciprocal-math              # x/y must not become x*(1/y)
    -fno-unsafe-math-optimizations
    -fno-finite-math-only             # NaN/Inf semantics must be honoured
    # -fsigned-zeros, never -fno-signed-zeros: the negative form is a fast-math
    # sub-option under which (-0.0) + 0.0 may compile to -0.0. fp_guard.hpp also
    # refuses __NO_SIGNED_ZEROS__, for a build that bypasses this file.
    # Rationale: docs/engineering_notes.md#signed-zeros
    -fsigned-zeros
  )
  include(CheckCXXCompilerFlag)

  # rule 3, probed rather than assumed: GCC accepts the flag for C++ only from
  # GCC 13, and manylinux_2_28 ships GCC 12. Where the probe fails, rule 3 rests
  # on the target evaluating each operation in its own type, which fp_guard.hpp
  # reads off FLT_EVAL_METHOD and `fp_selftest` reports.
  check_cxx_compiler_flag("-fexcess-precision=standard" TFIDF_HAS_EXCESS_PRECISION)
  if(TFIDF_HAS_EXCESS_PRECISION)
    target_compile_options(tfidf_numerics_strict INTERFACE -fexcess-precision=standard)
  endif()

  # x87 carries 80-bit intermediates; SSE2 is exactly binary64. Irrelevant on
  # x86-64 (SSE2 is the default ABI) but decisive on 32-bit hosts.
  check_cxx_compiler_flag("-mfpmath=sse" TFIDF_HAS_MFPMATH_SSE)
  if(TFIDF_HAS_MFPMATH_SSE)
    target_compile_options(tfidf_numerics_strict INTERFACE -mfpmath=sse)
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(tfidf_numerics_strict INTERFACE
    # /fp:strict, not /fp:precise: some MSVC versions contract to FMA under
    # /fp:precise, and `/fp:contract-` does not exist (MSVC ignores it, D9002).
    # /fp:strict forbids contraction and pins exception and rounding semantics.
    # Rationale: docs/engineering_notes.md#msvc-fp-strict
    /fp:strict
  )
else()
  message(WARNING "tfidf: unrecognised compiler '${CMAKE_CXX_COMPILER_ID}'. "
                  "Floating-point reproducibility is NOT guaranteed.")
endif()

# -----------------------------------------------------------------------------
# TFIDF_FAST_MATH: an unsound build that proves the compile-time guard fires.
# -----------------------------------------------------------------------------

# The `fastmath-guard-check` preset sets it, and `determinism.yml` asserts that
# the build fails at the fp_guard.hpp #error (Layer 1). The runtime self-test
# checks for reassociation and constant folding are therefore not reached in CI.
# -ffast-math comes after the strict flags, or GCC leaves __FAST_MATH__ unset.
if(TFIDF_FAST_MATH)
  message(WARNING "tfidf: TFIDF_FAST_MATH=ON, results from this build are NOT trustworthy.")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tfidf_numerics_strict INTERFACE -ffast-math)
  elseif(MSVC)
    target_compile_options(tfidf_numerics_strict INTERFACE /fp:fast)
  endif()
endif()

# -----------------------------------------------------------------------------
# Architecture tuning. OFF by default.
# -----------------------------------------------------------------------------

# -march=native changes vectorisation, can re-enable FMA contraction despite
# -ffp-contract=off in some GCC versions, and makes the binary machine-specific.
# Each of these breaks reproducibility.
if(TFIDF_ARCH_TUNE)
  message(WARNING "tfidf: TFIDF_ARCH_TUNE=ON, binary is machine-specific and NOT reproducible.")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tfidf_numerics_strict INTERFACE -march=native)
  elseif(MSVC)
    target_compile_options(tfidf_numerics_strict INTERFACE /arch:AVX2)
  endif()
endif()

# -----------------------------------------------------------------------------
# Record the exact policy so build_config.hpp, and hence every run manifest,
# can report what produced the numbers.
# -----------------------------------------------------------------------------
get_target_property(TFIDF_NUMERIC_FLAGS tfidf_numerics_strict INTERFACE_COMPILE_OPTIONS)
string(REPLACE ";" " " TFIDF_NUMERIC_FLAGS "${TFIDF_NUMERIC_FLAGS}")
set(TFIDF_NUMERIC_FLAGS "${TFIDF_NUMERIC_FLAGS}" CACHE INTERNAL "resolved FP flags")
