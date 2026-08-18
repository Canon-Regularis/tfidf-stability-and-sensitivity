# =============================================================================
# tfidf-stability, top-level build.
#
# The native backend is an OPTIONAL accelerator. The normative implementation is
# the pure-Python `reference` backend, which needs no compiler. Configure with
# -DTFIDF_BUILD_NATIVE=OFF for a fully functional reference-only install on a
# machine with no C++ toolchain.
# =============================================================================
cmake_minimum_required(VERSION 3.20)

project(tfidf_stability
        VERSION 0.2.0
        DESCRIPTION "Numerical stability and perturbation behaviour in TF-IDF similarity systems"
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(TFIDF_BUILD_NATIVE "Build the native (C++/nanobind) backend"          ON)
option(TFIDF_BUILD_TESTS  "Build the C++ unit-test binary"                   OFF)
option(TFIDF_BUILD_BENCH  "Build the C++ microbenchmark binary"              OFF)
option(TFIDF_WERROR       "Treat compiler warnings as errors"                OFF)

# NON-REPRODUCIBLE knobs. Both OFF for every published result. The manifest
# records their state and refuses to label a run reproducible when either is on.
option(TFIDF_ARCH_TUNE    "Enable -march=native (NON-REPRODUCIBLE)"          OFF)
option(TFIDF_FAST_MATH    "Enable fast-math on purpose, to prove the numeric \
guard rails actually fire. Never use for results." OFF)

set(TFIDF_SANITIZERS "" CACHE STRING
    "Semicolon/comma list of sanitizers, e.g. 'address;undefined' or 'thread'")

# enable_testing() has to be called from the TOP-LEVEL list file. Called deeper,
# CTest writes no root CTestTestfile.cmake and `ctest` reports "No tests were
# found" while still exiting 0.
if(TFIDF_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(cpp)

# -----------------------------------------------------------------------------
# Configuration summary. Printed so a build's numeric policy is never implicit.
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== tfidf-stability ${PROJECT_VERSION} =====================================")
message(STATUS "  compiler         : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  native backend   : ${TFIDF_BUILD_NATIVE}")
message(STATUS "  C++ tests        : ${TFIDF_BUILD_TESTS}")
message(STATUS "  benchmarks       : ${TFIDF_BUILD_BENCH}")
message(STATUS "  sanitizers       : '${TFIDF_SANITIZERS}'")
if(TFIDF_ARCH_TUNE OR TFIDF_FAST_MATH)
  message(STATUS "  reproducible     : *** NO *** (arch-tune=${TFIDF_ARCH_TUNE} \
fast-math=${TFIDF_FAST_MATH})")
else()
  message(STATUS "  reproducible     : yes (strict IEEE-754, no FMA contraction)")
endif()
message(STATUS "==============================================================")
message(STATUS "")
