# tests/CMakeLists.txt
#
# Phase 3 — Testing Infrastructure
#
# Each test is a standalone executable linked against the blas1 static
# library. Tests are registered with CTest via add_test() so the whole
# suite can be run with `ctest` or `cmake --build build -t test`.
#
# Math library note:
#   Test files use <math.h> functions directly (fabs, sqrt, sqrtl, isnan,
#   etc.) in their own verification code, separate from whatever blas1
#   itself links. Some of these (notably sqrtl) are not always inlined
#   as compiler builtins the way sqrt/fabs often are, so they require an
#   explicit link against libm on Linux. find_library() resolves this
#   portably — on platforms where libm is merged into libc (no separate
#   library to find), MATH_LIBRARY simply comes back empty and the
#   target_link_libraries() call below is a harmless no-op.
#
# Pattern for adding a new test file:
#   add_executable(test_X test_X.c)
#   target_link_libraries(test_X PRIVATE blas1 ${MATH_LIBRARY})
#   target_include_directories(test_X PRIVATE ${CMAKE_SOURCE_DIR}/include)
#   add_test(NAME X COMMAND test_X)

message(STATUS "Configuring test suite (Phase 3)")

find_library(MATH_LIBRARY m)

# ---- test_dot ------------------------------------------------------
add_executable(test_dot test_dot.c)
target_link_libraries(test_dot PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(test_dot PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME dot COMMAND test_dot)

# ---- test_axpy -------------------------------------------------------
add_executable(test_axpy test_axpy.c)
target_link_libraries(test_axpy PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(test_axpy PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME axpy COMMAND test_axpy)

# ---- test_scal -------------------------------------------------------
add_executable(test_scal test_scal.c)
target_link_libraries(test_scal PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(test_scal PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME scal COMMAND test_scal)

# ---- test_nrm2 -------------------------------------------------------
# nrm2 internally calls blas_iamax (see src/nrm2.c) — both are already
# part of the blas1 static library, so linking against blas1 alone
# covers that dependency; ${MATH_LIBRARY} below is for this test file's
# own use of sqrtl() in its long-double reference computation.
add_executable(test_nrm2 test_nrm2.c)
target_link_libraries(test_nrm2 PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(test_nrm2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME nrm2 COMMAND test_nrm2)

# ---- test_asum -------------------------------------------------------
add_executable(test_asum test_asum.c)
target_link_libraries(test_asum PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(test_asum PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME asum COMMAND test_asum)

# ---- test_iamax ------------------------------------------------------
add_executable(test_iamax test_iamax.c)
target_link_libraries(test_iamax PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(test_iamax PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME iamax COMMAND test_iamax)

# ---- test_correctness --------------------------------------------------
# Cross-function tests (step 3.7): exercises compositions of dot, axpy,
# scal, nrm2, asum, and iamax together, rather than any one function in
# isolation. Linking against blas1 covers all six.
add_executable(test_correctness test_correctness.c)
target_link_libraries(test_correctness PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(test_correctness PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME correctness COMMAND test_correctness)