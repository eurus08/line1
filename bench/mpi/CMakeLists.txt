# bench/mpi/CMakeLists.txt
#
# Phase 6, step 6.8 — MPI Benchmarks
#
# bench_mpi_dot is a single-data-point tool (see bench_mpi_dot.c for
# why) -- it measures ONE (rank count, n_local) combination per run.
# It's registered with CTest the same way bench_dot/bench_axpy are
# (Phase 4 pattern: name begins "bench" so `make bench`'s
# `ctest -R bench -V` picks it up), but that registration is only a
# SMOKE TEST at LINE1_MPI_TEST_NPROCS ranks with a modest fixed size
# -- confirming the tool runs and produces sane numbers. The actual
# strong/weak SCALING STUDIES (running this at many different rank
# counts and assembling a table) are done by run_scaling.sh, invoked
# manually -- CTest tests are not the right tool for an open-ended
# parameter sweep across process counts.
#
# Links against line1_mpi (brings in line1 + MPI::MPI_C transitively,
# same as the MPI test executables in tests/mpi/CMakeLists.txt).
# Needs bench/ on its include path too, for timing.h (this file lives
# in bench/mpi/, one directory below bench/ where timing.h is kept —
# reused as-is, no MPI-specific timing utility needed since bench_now()
# and bench_median() are already precision-agnostic).

message(STATUS "Configuring MPI benchmark suite (Phase 6, step 6.8)")

# Same cache variable tests/mpi/CMakeLists.txt defines. set(... CACHE ...)
# is idempotent -- if tests/mpi already defined it, this is a no-op; if
# LINE1_BUILD_TESTS=OFF and tests/mpi never ran, this defines it here
# instead, so this file doesn't silently break depending on build option
# ordering.
set(LINE1_MPI_TEST_NPROCS "4" CACHE STRING "Number of MPI ranks used to run MPI tests")

find_library(MATH_LIBRARY m)

add_executable(bench_mpi_dot bench_mpi_dot.c)
target_link_libraries(bench_mpi_dot PRIVATE line1_mpi ${MATH_LIBRARY})
target_include_directories(bench_mpi_dot PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/bench
)
# Same reasoning as bench/CMakeLists.txt: force optimised flags
# regardless of CMAKE_BUILD_TYPE, since an unoptimised benchmark
# number is meaningless.
target_apply_compiler_flags(bench_mpi_dot)
target_compile_options(bench_mpi_dot PRIVATE
    -O3 -march=native -funroll-loops -fomit-frame-pointer -ffast-math
)

# Smoke test only -- one rank count (LINE1_MPI_TEST_NPROCS, same
# variable the MPI unit tests use), one modest fixed n_local so it
# runs quickly as part of `make bench` / `make test`.
add_test(NAME bench_mpi_dot
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${LINE1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:bench_mpi_dot> 1000000 ${MPIEXEC_POSTFLAGS})
set_tests_properties(bench_mpi_dot PROPERTIES LABELS "bench")
# See tests/mpi/CMakeLists.txt for why this is set (OpenMPI-internal
# leaks unrelated to this project's code, tripping LeakSanitizer if
# this benchmark is ever built and run in a Debug+sanitizer config).
set_tests_properties(bench_mpi_dot PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
