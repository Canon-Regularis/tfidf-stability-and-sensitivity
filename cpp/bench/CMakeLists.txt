# =============================================================================
# Native microbenchmarks (nanobench, vendored).
#
# These measure individual kernels. pytest-benchmark measures across the FFI
# boundary and so cannot separate scoring cost from marshalling cost; this
# binary can. Built only with -DTFIDF_BUILD_BENCH=ON, and normally with the
# `bench` preset, which enables arch tuning and is therefore NOT reproducible.
# =============================================================================
add_library(tfidf_nanobench INTERFACE)
target_include_directories(tfidf_nanobench INTERFACE
  "${CMAKE_SOURCE_DIR}/cpp/third_party/nanobench")

add_executable(tfidf_bench bench_main.cpp)
target_link_libraries(tfidf_bench PRIVATE tfidf::core tfidf::warnings tfidf_nanobench)

if(MINGW)
  target_link_options(tfidf_bench PRIVATE -static)
endif()
