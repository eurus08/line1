# bench/CMakeLists.txt
#
# Phase 4 — Benchmarking Infrastructure
#
# Benchmarks are built ONLY when BLAS1_BUILD_BENCH=ON (see root
# CMakeLists.txt, section 8 — disabled by default since they take much
# longer to run than the test suite and aren't needed for a normal build).
#
# Each benchmark is:
#   1. A normal executable (so it can be run directly: ./build/bench/bench_dot)
#   2. ALSO registered with CTest via add_test(), with a name beginning
#      "bench_" so the Makefile's `make bench` target — which runs
#      `ctest -R bench -V` — picks it up automatically (CTest's -R flag
#      matches test names against a regex; "bench" matches "bench_dot",
#      "bench_axpy", etc.)
#
# Benchmarks are always built with optimised flags regardless of
# CMAKE_BUILD_TYPE, since the entire point is to measure optimised
# performance — see target_apply_compiler_flags() below, which applies
# the same warning flags from CompilerFlags.cmake the library uses, plus
# an explicit Release-equivalent optimisation flag set added directly.
#
# Pattern for adding a new benchmark:
#   add_executable(bench_X bench_X.c)
#   target_link_libraries(bench_X PRIVATE blas1 ${MATH_LIBRARY})
#   target_include_directories(bench_X PRIVATE ${CMAKE_SOURCE_DIR}/include
#                                               ${CMAKE_CURRENT_SOURCE_DIR})
#   target_apply_compiler_flags(bench_X)
#   target_compile_options(bench_X PRIVATE -O3 -march=native -funroll-loops
#                                           -fomit-frame-pointer -ffast-math)
#   add_test(NAME bench_X COMMAND bench_X)
#   set_tests_properties(bench_X PROPERTIES LABELS "bench")

message(STATUS "Configuring benchmark suite (Phase 4)")

find_library(MATH_LIBRARY m)

# ---- bench_dot -------------------------------------------------------
# Step 4.1: sweeps vector sizes 1K-100M, measures warm-cache and
# cold-cache median times over 100 trials each, reports GFlop/s and GB/s.
# See bench/bench_dot.c for full methodology notes.
add_executable(bench_dot bench_dot.c)
target_link_libraries(bench_dot PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(bench_dot PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Benchmarks must be compiled optimised even if the overall build type is
# Debug — an unoptimised benchmark number is meaningless. target_apply_compiler_flags()
# applies the shared warning flags; we additionally force the optimisation
# flag set directly rather than relying on CMAKE_BUILD_TYPE, since a developer
# might reasonably run
#   cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBLAS1_BUILD_BENCH=ON
# and would still expect real numbers out of `make bench`.
# target_apply_arch_flags() gives this executable the same USE_AVX2 /
# USE_NEON / USE_GENERIC define DetectArch.cmake selected for the
# library itself, purely so bench_dot.c can print which backend is
# actually active in its banner. This is why the DetectArch bug (fixed
# separately -- see cmake/DetectArch.cmake) went unnoticed for so long
# in the committed results/ numbers: nothing benchmark-side ever
# reported which kernel was really running.
target_apply_compiler_flags(bench_dot)
target_apply_arch_flags(bench_dot)
target_compile_options(bench_dot PRIVATE
    -O3 -march=native -funroll-loops -fomit-frame-pointer -ffast-math
)
add_test(NAME bench_dot COMMAND bench_dot)
set_tests_properties(bench_dot PROPERTIES LABELS "bench")

# ---- bench_axpy ------------------------------------------------------
# Step 4.2: same structure as bench_dot, reusing timing.h as-is. See
# bench/bench_axpy.c for what's different about axpy's methodology
# (3x bytes/element instead of 2x, void return value, in-place mutation
# of y across repeated warm-cache trials).
add_executable(bench_axpy bench_axpy.c)
target_link_libraries(bench_axpy PRIVATE blas1 ${MATH_LIBRARY})
target_include_directories(bench_axpy PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_apply_compiler_flags(bench_axpy)
target_apply_arch_flags(bench_axpy)
target_compile_options(bench_axpy PRIVATE
    -O3 -march=native -funroll-loops -fomit-frame-pointer -ffast-math
)
add_test(NAME bench_axpy COMMAND bench_axpy)
set_tests_properties(bench_axpy PROPERTIES LABELS "bench")
