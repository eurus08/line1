# tests/mpi/CMakeLists.txt
#
# Phase 6, step 6.7 — MPI Tests
#
# Each test is a standalone MPI executable linked against blas1_mpi
# (which transitively brings in blas1 and MPI::MPI_C -- see
# src/mpi/CMakeLists.txt). Registered with CTest the same way the
# serial tests are (tests/CMakeLists.txt), with one necessary
# difference: the add_test() COMMAND has to be the mpirun/mpiexec
# invocation, not the bare executable, since these programs call
# MPI_Init and will not run correctly (or at all, depending on the
# MPI implementation) launched any other way.
#
# BLAS1_MPI_TEST_NPROCS controls how many ranks every MPI test runs
# with. Override on the command line if the default doesn't suit
# your machine, e.g.:
#   cmake -B build -DBLAS1_BUILD_MPI=ON -DBLAS1_MPI_TEST_NPROCS=8
# If your machine has fewer physical cores than this and your MPI
# implementation refuses to oversubscribe by default (OpenMPI does
# this on some configurations), also pass:
#   -DMPIEXEC_PREFLAGS=--oversubscribe
#
# Pattern for adding a new MPI test file:
#   add_executable(test_mpi_X test_mpi_X.c)
#   target_link_libraries(test_mpi_X PRIVATE blas1_mpi ${MATH_LIBRARY})
#   target_include_directories(test_mpi_X PRIVATE ${CMAKE_SOURCE_DIR}/include)
#   add_test(NAME mpi_X
#       COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${BLAS1_MPI_TEST_NPROCS}
#               ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_X> ${MPIEXEC_POSTFLAGS})

set(BLAS1_MPI_TEST_NPROCS "4" CACHE STRING "Number of MPI ranks used to run MPI tests")

message(STATUS "Configuring MPI test suite (Phase 6, step 6.7) -- ${BLAS1_MPI_TEST_NPROCS} ranks")

find_library(MATH_LIBRARY m)

# ---- test_mpi_dot ------------------------------------------------------
add_executable(test_mpi_dot test_mpi_dot.c)
target_link_libraries(test_mpi_dot PRIVATE blas1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_dot PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_dot
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${BLAS1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_dot> ${MPIEXEC_POSTFLAGS})

# ---- test_mpi_axpy -------------------------------------------------------
add_executable(test_mpi_axpy test_mpi_axpy.c)
target_link_libraries(test_mpi_axpy PRIVATE blas1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_axpy PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_axpy
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${BLAS1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_axpy> ${MPIEXEC_POSTFLAGS})

# ---- test_mpi_nrm2 -------------------------------------------------------
add_executable(test_mpi_nrm2 test_mpi_nrm2.c)
target_link_libraries(test_mpi_nrm2 PRIVATE blas1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_nrm2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_nrm2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${BLAS1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_nrm2> ${MPIEXEC_POSTFLAGS})

# ---- test_mpi_iamax -------------------------------------------------------
add_executable(test_mpi_iamax test_mpi_iamax.c)
target_link_libraries(test_mpi_iamax PRIVATE blas1_mpi ${MATH_LIBRARY})
target_include_directories(test_mpi_iamax PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mpi_iamax
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${BLAS1_MPI_TEST_NPROCS}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_iamax> ${MPIEXEC_POSTFLAGS})
