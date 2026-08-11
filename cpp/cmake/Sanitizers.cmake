# =============================================================================
# Sanitizers.cmake
#
# Usage:  -DTFIDF_SANITIZERS="address;undefined"   or   -DTFIDF_SANITIZERS=thread
#
# `float-divide-by-zero` and `float-cast-overflow` are included with UBSan
# deliberately: in this codebase a division by a zero vector norm is a real
# logic error (the zero-vector convention of README section 2.3 must be applied
# *before* the division, never discovered by producing an Inf).
# =============================================================================
function(tfidf_apply_sanitizers target scope)
  if(NOT TFIDF_SANITIZERS)
    return()
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "tfidf: sanitizers requested but ${CMAKE_CXX_COMPILER_ID} is unsupported.")
    return()
  endif()

  string(REPLACE "," ";" _sans "${TFIDF_SANITIZERS}")
  string(REPLACE ";" "," _san_arg "${_sans}")

  set(_opts -fsanitize=${_san_arg} -fno-omit-frame-pointer -g)
  if("undefined" IN_LIST _sans)
    list(APPEND _opts -fsanitize=float-divide-by-zero -fno-sanitize-recover=all)
  endif()

  target_compile_options(${target} ${scope} ${_opts})
  target_link_options(${target} ${scope} ${_opts})
  message(STATUS "tfidf: sanitizers '${_san_arg}' enabled for ${target}")
endfunction()
