# src/mpi/CMakeLists.txt
#
# Phase 6 — MPI Parallel Layer
#
# This subdirectory is only entered when BLAS1_BUILD_MPI is ON (see root
# CMakeLists.txt section 9, which also runs find_package(MPI) before
# add_subdirectory(src/mpi) is called — so the MPI::MPI_C imported target
# already exists by the time this file executes).
#
# Design: each MPI function is a thin wrapper, not a reimplementation.
# It calls the existing serial blas1 kernel on its local slice of the
# vector, then performs exactly one MPI collective to combine results
# across ranks. This is why blas1_mpi links against blas1 (PUBLIC) —
# every wrapper .c file needs the serial function declarations.
#
# Step 6.2 (this commit): scaffolding only. BLAS1_MPI_SOURCES starts
# empty and add_library() is guarded behind it, so `cmake --build`
# succeeds with zero MPI object files compiled — mirroring the empty
# skeleton deliverable from Phase 1, step 1.1. Steps 6.3-6.6 append one
# source file each as mpi_dot.c, mpi_axpy.c, mpi_nrm2.c, and
# mpi_iamax.c are written.
#
# Pattern for adding a new MPI wrapper (used in steps 6.3-6.6):
#   list(APPEND BLAS1_MPI_SOURCES mpi_X.c)

set(BLAS1_MPI_SOURCES
    mpi_dot.c        # step 6.3 — done
    mpi_axpy.c       # step 6.4 — done
    # mpi_nrm2.c     <- step 6.5
    # mpi_iamax.c    <- step 6.6
)

if(BLAS1_MPI_SOURCES)
    add_library(blas1_mpi STATIC ${BLAS1_MPI_SOURCES})

    target_include_directories(blas1_mpi
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    # PUBLIC: anything linking blas1_mpi (our MPI test/bench executables)
    # needs blas1's declarations and MPI's headers transitively too.
    target_link_libraries(blas1_mpi
        PUBLIC
            blas1
            MPI::MPI_C
    )

    # Same warning set as the serial library — Wmissing-prototypes etc.
    # apply here too. Defined once in cmake/CompilerFlags.cmake,
    # included at the root, so the function is already in scope.
    target_apply_compiler_flags(blas1_mpi)

    message(STATUS "MPI layer: blas1_mpi target created (${BLAS1_MPI_SOURCES})")
else()
    message(STATUS "MPI layer: scaffolding only, no source files yet (step 6.2)")
endif()
