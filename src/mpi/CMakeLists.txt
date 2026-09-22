# src/mpi/CMakeLists.txt
#
# Phase 6 — MPI Parallel Layer
#
# This subdirectory is only entered when LINE1_BUILD_MPI is ON (see root
# CMakeLists.txt section 9, which also runs find_package(MPI) before
# add_subdirectory(src/mpi) is called — so the MPI::MPI_C imported target
# already exists by the time this file executes).
#
# Design: each MPI function is a thin wrapper, not a reimplementation.
# It calls the existing serial line1 kernel on its local slice of the
# vector, then performs exactly one MPI collective to combine results
# across ranks. This is why line1_mpi links against line1 (PUBLIC) —
# every wrapper .c file needs the serial function declarations.
#
# Step 6.2 (this commit): scaffolding only. LINE1_MPI_SOURCES starts
# empty and add_library() is guarded behind it, so `cmake --build`
# succeeds with zero MPI object files compiled — mirroring the empty
# skeleton deliverable from Phase 1, step 1.1. Steps 6.3-6.6 append one
# source file each as mpi_dot.c, mpi_axpy.c, mpi_nrm2.c, and
# mpi_iamax.c are written.
#
# Pattern for adding a new MPI wrapper (used in steps 6.3-6.6):
#   list(APPEND LINE1_MPI_SOURCES mpi_X.c)

set(LINE1_MPI_SOURCES
    mpi_dot.c        # step 6.3 — done
    mpi_axpy.c       # step 6.4 — done
    mpi_nrm2.c       # step 6.5 — done
    mpi_iamax.c      # step 6.6 — done
)

if(LINE1_MPI_SOURCES)
    add_library(line1_mpi STATIC ${LINE1_MPI_SOURCES})

    target_include_directories(line1_mpi
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    # PUBLIC: anything linking line1_mpi (our MPI test/bench executables)
    # needs line1's declarations and MPI's headers transitively too.
    target_link_libraries(line1_mpi
        PUBLIC
            line1
            MPI::MPI_C
    )

    # Same warning set as the serial library — Wmissing-prototypes etc.
    # apply here too. Defined once in cmake/CompilerFlags.cmake,
    # included at the root, so the function is already in scope.
    target_apply_compiler_flags(line1_mpi)

    # ------------------------------------------------------------------
    # Installation (Phase 7, step 7.4) — mirrors the root CMakeLists.txt's
    # section 11 for the serial line1 target, but for line1_mpi. Kept
    # here rather than in the root file since line1_mpi is defined here
    # and only exists at all when this block runs.
    # ------------------------------------------------------------------
    install(TARGETS line1_mpi
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Turn MPI_C_INCLUDE_DIRS / MPI_C_LIBRARIES (populated by the
    # find_package(MPI) call in the root CMakeLists.txt) into plain
    # space-separated strings line1-mpi.pc.in can substitute directly.
    # A consumer using pkg-config has no access to CMake imported
    # targets like MPI::MPI_C -- the actual flags have to be spelled
    # out literally in the .pc file itself.
    set(LINE1_MPI_PC_CFLAGS "")
    foreach(_dir ${MPI_C_INCLUDE_DIRS})
        string(APPEND LINE1_MPI_PC_CFLAGS "-I${_dir} ")
    endforeach()

    set(LINE1_MPI_PC_LIBS "")
    foreach(_lib ${MPI_C_LIBRARIES})
        string(APPEND LINE1_MPI_PC_LIBS "${_lib} ")
    endforeach()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/line1-mpi.pc.in
        ${CMAKE_BINARY_DIR}/line1-mpi.pc
        @ONLY
    )
    install(FILES ${CMAKE_BINARY_DIR}/line1-mpi.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )

    message(STATUS "MPI layer: line1_mpi target created (${LINE1_MPI_SOURCES})")
else()
    message(STATUS "MPI layer: scaffolding only, no source files yet (step 6.2)")
endif()
