# =============================================================================
# cpp/: the native core, its bindings, tests and benchmarks.
# =============================================================================

# Each C++ subpackage under cpp/include/tfidf/, except core/, mirrors the Python
# subpackage of the same name under src/tfidf_stability/. Only the hot paths are
# ported, so most Python subpackages have no C++ counterpart.
# `scripts/check_layout.py` checks the C++-to-Python direction only.
include(NumericsFlags)
include(Warnings)
include(Sanitizers)

# -----------------------------------------------------------------------------
# Build metadata baked into the binary.
# -----------------------------------------------------------------------------

# build_config.hpp records the compiler, flags and git revision, so that
# `tfidf_stability.build_info()` and every run manifest state what produced a
# given number.
find_package(Git QUIET)
set(TFIDF_GIT_SHA "unknown")
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
                  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                  OUTPUT_VARIABLE TFIDF_GIT_SHA
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

# CMake booleans are ON/OFF; C++ needs true/false.
set(TFIDF_ARCH_TUNE_CXX "false")
set(TFIDF_FAST_MATH_CXX "false")
if(TFIDF_ARCH_TUNE)
  set(TFIDF_ARCH_TUNE_CXX "true")
endif()
if(TFIDF_FAST_MATH)
  set(TFIDF_FAST_MATH_CXX "true")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tfidf/core/build_config.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/tfidf/core/build_config.hpp"
               @ONLY)

# -----------------------------------------------------------------------------
# tfidf_core: the mathematics.
# -----------------------------------------------------------------------------

# Header-dominant and template-heavy, with no reference to Python, so the core
# is testable on its own and the binding layer is replaceable.
add_library(tfidf_core INTERFACE)
add_library(tfidf::core ALIAS tfidf_core)

target_include_directories(tfidf_core INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_compile_features(tfidf_core INTERFACE cxx_std_20)

# Strict floating point is a usage requirement: any target that compiles these
# headers must use IEEE-754 semantics. Warnings are not: through the INTERFACE
# they would apply the pedantic style checks to third-party headers (nanobind,
# Python.h). The tests and benchmarks link tfidf::warnings explicitly.
target_link_libraries(tfidf_core INTERFACE tfidf::numerics_strict)
tfidf_apply_sanitizers(tfidf_core INTERFACE)

# -----------------------------------------------------------------------------
# The nanobind extension module.
# -----------------------------------------------------------------------------
if(TFIDF_BUILD_NATIVE)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_CMAKE_DIR
                  RESULT_VARIABLE NB_RC ERROR_QUIET)
  if(NOT NB_RC EQUAL 0)
    message(FATAL_ERROR "tfidf: nanobind not importable by ${Python_EXECUTABLE}.\n"
                        "Install it (`pip install nanobind`) or configure with "
                        "-DTFIDF_BUILD_NATIVE=OFF for a reference-only build.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_tfidf_native NB_STATIC
    bindings/module.cpp
  )
  target_link_libraries(_tfidf_native PRIVATE tfidf::core)

  # module.cpp is pure marshalling, but it still gets the substantive warnings.
  # The pedantic style checks (-Wold-style-cast, -Wshadow) are omitted because
  # nanobind's macros and Python.h trip them hundreds of times, drowning out any
  # real diagnostic. tfidf::core and the test binary keep the full set.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_tfidf_native PRIVATE -Wall -Wextra)
  elseif(MSVC)
    target_compile_options(_tfidf_native PRIVATE /W3)
  endif()

  # MinGW: GCC runtime DLLs are not on the DLL search path of the MSVC-built
  # CPython. Plain -static links libgcc, libstdc++ and libwinpthread statically,
  # so the .pyd imports only python3xx.dll, KERNEL32 and the UCRT.
  # Rationale: docs/engineering_notes.md#mingw-static-link
  if(MINGW)
    target_link_options(_tfidf_native PRIVATE -static -static-libgcc -static-libstdc++)
  endif()

  # Land the module next to its Python loader, for both editable and wheel installs.
  install(TARGETS _tfidf_native LIBRARY DESTINATION tfidf_stability/_native
                                RUNTIME DESTINATION tfidf_stability/_native)
  set_target_properties(_tfidf_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/tfidf_stability/_native"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/tfidf_stability/_native")
endif()

# -----------------------------------------------------------------------------
# Tests and benchmarks (vendored single-header frameworks -> hermetic, offline).
# -----------------------------------------------------------------------------
if(TFIDF_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(TFIDF_BUILD_BENCH)
  add_subdirectory(bench)
endif()
