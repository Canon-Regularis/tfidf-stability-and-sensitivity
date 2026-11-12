# =============================================================================
# Native microbenchmarks (nanobench, vendored).
# =============================================================================

# These time individual kernels; pytest-benchmark measures across the FFI
# boundary and cannot separate kernel cost from marshalling cost. Built only with
# -DTFIDF_BUILD_BENCH=ON, normally through the `bench` preset, which enables
# arch tuning and so is not reproducible.
add_library(tfidf_nanobench INTERFACE)
target_include_directories(tfidf_nanobench INTERFACE
  "${CMAKE_SOURCE_DIR}/cpp/third_party/nanobench")

add_executable(tfidf_bench bench_main.cpp)
target_link_libraries(tfidf_bench PRIVATE tfidf::core tfidf::warnings tfidf_nanobench)

if(MINGW)
  target_link_options(tfidf_bench PRIVATE -static)
endif()
