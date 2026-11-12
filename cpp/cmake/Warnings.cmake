# =============================================================================
# Warnings.cmake: a strict warning set.
# =============================================================================

# Chosen for a numerics codebase:
#   -Wdouble-promotion   catches a float promoted inside a double computation
#   -Wconversion         catches silent int/float narrowing in index arithmetic
#   -Wsign-conversion    catches a negative index reaching an unsigned span

# -Wfloat-equal is excluded. Bit-exactness is the contract, so exact `==` on
# doubles is normal here: the warning fires on every comparison in the core and
# every `CHECK(x == y)` in the tests, and silencing it at each site would need a
# pragma with no justification.
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
