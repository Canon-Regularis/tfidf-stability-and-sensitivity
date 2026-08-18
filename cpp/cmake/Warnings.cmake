# =============================================================================
# Warnings.cmake: a noisy warning set, on purpose.
#
# Several of these are chosen specifically for a numerics codebase:
#   -Wfloat-equal        forces every intentional `==` on doubles to be justified
#                        (doubles are compared exactly here; bit-exactness is
#                        the contract, so each site is silenced locally, which
#                        makes the intent reviewable rather than accidental)
#   -Wdouble-promotion   catches a float sneaking into a double computation
#   -Wconversion         catches silent int/float narrowing in index arithmetic
# =============================================================================
add_library(tfidf_warnings INTERFACE)
add_library(tfidf::warnings ALIAS tfidf_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  target_compile_options(tfidf_warnings INTERFACE
    -Wall -Wextra -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
    -Wnull-dereference
    -Wimplicit-fallthrough
  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tfidf_warnings INTERFACE
      -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
  endif()
elseif(MSVC)
  target_compile_options(tfidf_warnings INTERFACE /W4 /permissive- /Zc:__cplusplus)
endif()

if(TFIDF_WERROR)
  if(MSVC)
    target_compile_options(tfidf_warnings INTERFACE /WX)
  else()
    target_compile_options(tfidf_warnings INTERFACE -Werror)
  endif()
endif()
