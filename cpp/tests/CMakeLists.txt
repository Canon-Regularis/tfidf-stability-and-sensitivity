# =============================================================================
# Native unit tests (doctest, vendored -> hermetic and offline).
#
# These test the C++ core directly, without Python in the loop. They mirror the
# same golden values the Python reference is checked against, so a divergence
# between the two implementations shows up here first and with a much shorter
# debugging path than a cross-language differential failure would give.
# =============================================================================
add_library(tfidf_doctest INTERFACE)
target_include_directories(tfidf_doctest INTERFACE
  "${CMAKE_SOURCE_DIR}/cpp/third_party/doctest")

add_executable(tfidf_tests
  main.cpp
  test_fp_guard.cpp
  test_reduction.cpp
  test_scoring.cpp
  test_ranking.cpp
  test_distances.cpp
)

target_link_libraries(tfidf_tests PRIVATE tfidf::core tfidf::warnings tfidf_doctest)
tfidf_apply_sanitizers(tfidf_tests PRIVATE)

if(MINGW)
  target_link_options(tfidf_tests PRIVATE -static)
endif()

# Register each doctest TEST_CASE individually with CTest so a failure names the
# specific case rather than just "the test binary".
include("${CMAKE_SOURCE_DIR}/cpp/third_party/doctest/doctest.cmake" OPTIONAL
        RESULT_VARIABLE _doctest_cmake)
add_test(NAME cpp.unit COMMAND tfidf_tests --reporters=console --no-intro=false)
set_tests_properties(cpp.unit PROPERTIES TIMEOUT 300)
