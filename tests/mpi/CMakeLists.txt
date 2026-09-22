# tests/mpi/CMakeLists.txt
#
# Phase 6, step 6.7 — MPI Tests
#
# Each test is a standalone MPI executable linked against line1_mpi
# (which transitively brings in line1 and MPI::MPI_C -- see
# src/mpi/CMakeLists.txt). Registered with CTest the same way the
# serial tests are (tests/CMakeLists.txt), with one necessary
# difference: the add_test() COMMAND has to be the mpirun/mpiexec
# invocation, not the bare executable, since these programs call
# MPI_Init and will not run correctly (or at all, depending on the
# MPI implementation) launched any other way.
#
# LINE1_MPI_TEST_NPROCS controls how many ranks every MPI test runs
# with. Override on the command line if the default doesn't suit
# your machine, e.g.:
#   cmake -B build -DLINE1_BUILD_MPI=ON -DLINE1_MPI_TEST_NPROCS=8
# If your machine has fewer physical cores than this and your MPI
# implementation refuses to oversubscribe by default (OpenMPI does
# this on some configurations), also pass:
#   -DMPIEXEC_PREFLAGS=--oversubscribe
#
# ASan/LeakSanitizer note (Phase 7, added after a real CI failure):
# in a Debug build (sanitizers on), OpenMPI's own runtime -- hwloc's
# topology discovery, libevent -- allocates memory it does not free
# until well after MPI_Finalize, in ways LeakSanitizer flags as leaks
# at process exit. None of these leaks are in this project's code
# (confirmed while building mpi_dot.c: the leak stack traces point
# into libmpi.so/libhwloc.so, never into any line1 source file). This
# was worked around by hand during development
# (ASAN_OPTIONS=detect_leaks=0), but that workaround was never wired
# into the actual CTest execution environment -- so it silently did
# not protect real `ctest` runs, including CI's Debug+sanitizers job,
# until that gap actually surfaced there. ENVIRONMENT is set on every
# MPI test below to close that gap for good. Leak detection is
# disabled only for these tests specifically -- AddressSanitizer's
# other checks (heap corruption, use-after-free, etc.) and UBSan
# remain fully active; this narrows the check, it does not weaken it.
#
# Pattern for adding a new MPI test file:
#   add_executable(test_mpi_X test_mpi_X.c)
#   target_link_libraries(test_mpi_X PRIVATE line1_mpi ${MATH_LIBRARY})
#   target_include_directories(test_mpi_X PRIVATE ${CMAKE_SOURCE_DIR}/include)
#   add_test(NAME mpi_X
#       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${LINE1_MPI_TEST_NPROCS}
#               ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_X> ${MPIEXEC_POSTFLAGS})
#   set_tests_properties(mpi_X PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

set(LINE1_MPI_TEST_NPROCS "4" CACHE STRING "Number of MPI ranks used to run MPI tests")

message(STATUS "Configuring MPI test suite (Phase 6, step 6.7) -- ${LINE1_MPI_TEST_NPROCS} ranks")

find_library(MATH_LIBRARY m)

# ---- test_mpi_dot ------------------------------------------------------
add_executable(test_mpi_dot test_mpi_dot.c)
target_link_libraries(test_mpi_dot PRIVATE line1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_dot PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_dot
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${LINE1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_dot> ${MPIEXEC_POSTFLAGS})
set_tests_properties(mpi_dot PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

# ---- test_mpi_axpy -------------------------------------------------------
add_executable(test_mpi_axpy test_mpi_axpy.c)
target_link_libraries(test_mpi_axpy PRIVATE line1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_axpy PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_axpy
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${LINE1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_axpy> ${MPIEXEC_POSTFLAGS})
set_tests_properties(mpi_axpy PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

# ---- test_mpi_nrm2 -------------------------------------------------------
add_executable(test_mpi_nrm2 test_mpi_nrm2.c)
target_link_libraries(test_mpi_nrm2 PRIVATE line1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_nrm2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_nrm2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${LINE1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_nrm2> ${MPIEXEC_POSTFLAGS})
set_tests_properties(mpi_nrm2 PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

# ---- test_mpi_iamax -------------------------------------------------------
add_executable(test_mpi_iamax test_mpi_iamax.c)
target_link_libraries(test_mpi_iamax PRIVATE line1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_iamax PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_iamax
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${LINE1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_iamax> ${MPIEXEC_POSTFLAGS})
set_tests_properties(mpi_iamax PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
