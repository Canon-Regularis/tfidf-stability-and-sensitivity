# =============================================================================
# cpp/ -- the native core, its bindings, tests and benchmarks.
#
# Layout mirrors the Python package one-for-one:
#     cpp/include/tfidf/<sub>/   <->   src/tfidf_stability/<sub>/
# A CI check asserts the correspondence, so the two trees cannot drift.
# =============================================================================
include(NumericsFlags)
include(Warnings)
include(Sanitizers)

# -----------------------------------------------------------------------------
# build_config.hpp -- bakes the exact compiler, flags and git revision into the
# binary so `tfidf_stability.build_info()` (and hence every run manifest) can
# state precisely what produced a given number.
# -----------------------------------------------------------------------------
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
# tfidf_core -- the mathematics. Header-dominant, template-heavy, and containing
# NO reference to Python whatsoever, so the core stays independently testable
# and the binding layer stays swappable.
# -----------------------------------------------------------------------------
add_library(tfidf_core INTERFACE)
add_library(tfidf::core ALIAS tfidf_core)

target_include_directories(tfidf_core INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_compile_features(tfidf_core INTERFACE cxx_std_20)

# Strict floating-point IS a usage requirement: anything that compiles these
# headers must do so under IEEE-754 semantics, or the results it produces are
# meaningless. Warnings are NOT a usage requirement -- propagating them through
# the INTERFACE would force our pedantic style checks onto third-party headers
# (nanobind, Python.h) that legitimately do not follow them. Each of our own
# targets opts into tfidf::warnings explicitly instead.
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

  # ---------------------------------------------------------------------------
  # MinGW note (verified empirically on this project's reference machine):
  #
  # CPython on Windows is built with MSVC, so a MinGW-produced .pyd must not
  # depend on any GCC runtime DLL -- those are not on the interpreter's DLL
  # search path and the import fails with a bare, unhelpful
  # "DLL load failed while importing ...".
  #
  # -static-libgcc/-static-libstdc++ alone are NOT sufficient: the posix thread
  # model also pulls in libwinpthread-1.dll. Plain -static covers all three.
  # With it the import list reduces to python3xx.dll + KERNEL32 + UCRT, which is
  # exactly what a self-contained extension should need.
  # ---------------------------------------------------------------------------
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
