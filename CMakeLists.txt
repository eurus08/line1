# CMakeLists.txt — Root build file for the BLAS1 library
#
# Quick reference:
#   cmake -B build              <- configure (creates the build folder)
#   cmake --build build         <- compile
#   cmake --build build -t test <- run tests
#   cmake --build build -t bench <- run benchmarks (added later)
#
# Build types:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release   <- optimised (default)
#   cmake -B build -DCMAKE_BUILD_TYPE=Debug     <- debug symbols, no optimisation

cmake_minimum_required(VERSION 3.18)

project(blas1
    VERSION     1.0.0
    DESCRIPTION "BLAS Level 1 library — serial C + MPI parallel"
    LANGUAGES   C
)

# GNUInstallDirs sets portable install locations (CMAKE_INSTALL_LIBDIR,
# CMAKE_INSTALL_INCLUDEDIR, ...) -- e.g. "lib" vs "lib64" depending on
# the distro's own convention, rather than hardcoding one. Needed by
# section 11 (Installation) below, and included here near the top
# since it's used across the file.
include(GNUInstallDirs)


# ------------------------------------------------------------------ #
#  0. Let CMake tell vscode where /include is                       #
# ------------------------------------------------------------------ #

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ------------------------------------------------------------------ #
#  1. C standard                                                       #
#     C11 gives us: _Generic, static_assert, aligned_alloc            #
# ------------------------------------------------------------------ #
set(CMAKE_C_STANDARD          11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS        OFF)   # use -std=c11, not -std=gnu11

# ------------------------------------------------------------------ #
#  2. Default build type                                               #
#     If the user doesn't specify, build in Release mode.             #
# ------------------------------------------------------------------ #
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified — defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------ #
#  3. Load our custom CMake modules                                    #
#     These live in cmake/ and are written in steps 1.6 and 1.7       #
# ------------------------------------------------------------------ #
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CompilerFlags)   # sets -O3, -march=native, -Wall, etc.
include(DetectArch)      # sets USE_AVX2 or USE_NEON define.

# ------------------------------------------------------------------ #
#  4. Precision option                                                 #
#     OFF = double (default, HPC standard)                            #
#     ON  = float  (single precision, for special use cases)          #
# ------------------------------------------------------------------ #
option(BLAS1_USE_FLOAT "Build library in single precision (float)" OFF)

if(BLAS1_USE_FLOAT)
    message(STATUS "Precision: float (single)")
    add_compile_definitions(BLAS_USE_FLOAT)
else()
    message(STATUS "Precision: double (default)")
endif()

# ------------------------------------------------------------------ #
#  5. Collect source files                                             #
#     We list them explicitly — never use GLOB for sources.           #
#     (GLOB won't detect newly added files without re-running CMake)  #
# ------------------------------------------------------------------ #
set(BLAS1_SOURCES
    src/dot.c
    src/axpy.c
    src/scal.c
    src/nrm2.c
    src/asum.c
    src/iamax.c
)

# ------------------------------------------------------------------ #
#  5b. Kernel sources — selected based on SIMD backend               #
#      DetectArch.cmake already set BLAS1_SIMD_BACKEND above.        #
# ------------------------------------------------------------------ #
if(BLAS1_SIMD_BACKEND STREQUAL "avx2")
    list(APPEND BLAS1_SOURCES
        kernel/x86/dot_avx2.c
        kernel/x86/axpy_avx2.c
    )
    set_source_files_properties(kernel/x86/dot_avx2.c
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
    )
    set_source_files_properties(kernel/x86/axpy_avx2.c
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
    )
elseif(BLAS1_SIMD_BACKEND STREQUAL "neon")
    list(APPEND BLAS1_SOURCES
        kernel/arm/dot_neon.c
        kernel/arm/axpy_neon.c
    )
else()
    list(APPEND BLAS1_SOURCES
        kernel/generic/dot_generic.c
        kernel/generic/axpy_generic.c
    )
endif()
# ------------------------------------------------------------------ #
#  6. Library targets — static (always) + shared (opt-in, step 7.5)   #
#     This produces:  build/libblas1.a   (and libblas1.so if enabled) #
# ------------------------------------------------------------------ #

# The real target is named blas1_static, not blas1 -- this is what
# lets a static AND a shared library coexist as build artifacts that
# are both, correctly, called "blas1" (libblas1.a / libblas1.so): a
# single CMake target can only be one of STATIC or SHARED, so two
# distinct targets are unavoidable once both are wanted "alongside"
# each other rather than as an either/or choice.
#
# The `blas1` ALIAS below is what keeps every existing
# target_link_libraries(... blas1) reference across tests/,
# bench/, src/mpi/, tests/mpi/, and bench/mpi/ working completely
# unchanged -- none of those files needed to be touched for this
# step. Internal consumers (tests, benchmarks, the MPI layer) always
# link the static library specifically, on purpose: it keeps their
# executables self-contained with no runtime dependency on finding a
# co-located .so via RPATH. The shared library, when built, exists
# purely as an additional installable artifact for external
# consumers who want dynamic linking -- nothing in this repository
# itself links against it.
add_library(blas1_static STATIC ${BLAS1_SOURCES})
add_library(blas1 ALIAS blas1_static)

target_include_directories(blas1_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Apply compiler flags defined in cmake/CompilerFlags.cmake
target_apply_compiler_flags(blas1_static)
target_apply_arch_flags(blas1_static)

# OUTPUT_NAME (not the target name itself) is what controls the
# actual file on disk -- this is what makes libblas1_static.a
# instead be written as libblas1.a.
set_target_properties(blas1_static PROPERTIES
    OUTPUT_NAME blas1
)

option(BLAS1_BUILD_SHARED "Also build a shared library (libblas1.so) alongside the static one" OFF)

if(BLAS1_BUILD_SHARED)
    add_library(blas1_shared SHARED ${BLAS1_SOURCES})

    target_include_directories(blas1_shared
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    target_apply_compiler_flags(blas1_shared)
    target_apply_arch_flags(blas1_shared)

    # CMake automatically compiles SHARED library sources with -fPIC --
    # no POSITION_INDEPENDENT_CODE property needed here, that part is
    # handled unconditionally by CMake itself for any SHARED target.

    set_target_properties(blas1_shared PROPERTIES
        OUTPUT_NAME    blas1
        # Standard shared-library versioning: the real file is
        # libblas1.so.1.0.0, with libblas1.so.1 and libblas1.so as
        # symlinks pointing to it. VERSION is the full version;
        # SOVERSION (just the major number) is what a binary built
        # against this library actually records as its runtime
        # dependency -- so a future 1.1.0 stays load-compatible with
        # anything linked against 1.0.0, while a future 2.0.0
        # (a new SOVERSION) correctly would not.
        VERSION        ${PROJECT_VERSION}
        SOVERSION      ${PROJECT_VERSION_MAJOR}
    )

    message(STATUS "Shared library: ON  (libblas1.so.${PROJECT_VERSION})")
else()
    message(STATUS "Shared library: OFF (static only -- enable with -DBLAS1_BUILD_SHARED=ON)")
endif()

# ------------------------------------------------------------------ #
#  7. Tests (Phase 3)                                                  #
#     Enabled by default. Disable with -DBLAS1_BUILD_TESTS=OFF        #
# ------------------------------------------------------------------ #
option(BLAS1_BUILD_TESTS "Build the test suite" ON)

if(BLAS1_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------ #
#  8. Benchmarks (Phase 4)                                             #
#     Disabled by default — only built when explicitly requested.     #
# ------------------------------------------------------------------ #
option(BLAS1_BUILD_BENCH "Build the benchmark suite" OFF)

if(BLAS1_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# ------------------------------------------------------------------ #
#  9. MPI parallel layer (Phase 6)                                    #
#     Disabled by default — only probed when explicitly requested.    #
#     find_package(MPI REQUIRED) here means: if the user turns this   #
#     option on, a missing MPI install is a hard configure error,     #
#     not a silent skip.                                              #
#                                                                      #
#     COMPONENTS C restricts the probe to the C bindings only — the   #
#     bare form also checks for Fortran MPI bindings, which triggers  #
#     a Fortran compiler search we don't need (this is a C-only       #
#     project per LANGUAGES C above) and can fail on machines with    #
#     no Fortran toolchain installed.                                 #
# ------------------------------------------------------------------ #
option(BLAS1_BUILD_MPI "Build the MPI parallel layer" OFF)

if(BLAS1_BUILD_MPI)
    find_package(MPI REQUIRED COMPONENTS C)

    message(STATUS "MPI found:")
    message(STATUS "  MPI_C_COMPILER      : ${MPI_C_COMPILER}")
    message(STATUS "  MPI_C_INCLUDE_DIRS  : ${MPI_C_INCLUDE_DIRS}")
    message(STATUS "  MPI_C_LIBRARIES     : ${MPI_C_LIBRARIES}")

    # src/mpi/CMakeLists.txt defines the blas1_mpi target. It must come
    # after find_package(MPI) above so the MPI::MPI_C imported target
    # it links against already exists.
    add_subdirectory(src/mpi)

    # tests/mpi/CMakeLists.txt registers MPI test executables with
    # CTest (step 6.7), launched via mpirun. Gated on BLAS1_BUILD_TESTS
    # too -- someone who explicitly disabled the test suite shouldn't
    # get MPI test executables built anyway.
    if(BLAS1_BUILD_TESTS)
        add_subdirectory(tests/mpi)
    endif()

    # bench/mpi/CMakeLists.txt registers the MPI scaling benchmark
    # (step 6.8). Gated on BLAS1_BUILD_BENCH for the same reason the
    # serial benchmarks are (section 8 above) -- benchmarks take much
    # longer to run than tests and aren't needed for a normal build.
    if(BLAS1_BUILD_BENCH)
        add_subdirectory(bench/mpi)
    endif()
endif()

# ------------------------------------------------------------------ #
#  11. Installation (Phase 7, step 7.4)                               #
#      Enables `cmake --install build` and, via the generated .pc    #
#      file, `pkg-config --cflags --libs blas1` for any project      #
#      that wants to consume this library once installed.            #
# ------------------------------------------------------------------ #

# The blas1_static target's own target_include_directories (section 6)
# already uses $<INSTALL_INTERFACE:include> -- this install() call is
# what makes that interface true after `cmake --install`, by actually
# placing blas1 where that generator expression promised it would be.
#
# Note: this installs blas1_static, not the `blas1` ALIAS from
# section 6 -- ALIAS targets cannot be passed to install(TARGETS) at
# all (CMake rejects it outright), so the real target name has to be
# used here regardless of which name the rest of the build refers to
# it by.
install(TARGETS blas1_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(BLAS1_BUILD_SHARED)
    # Three destination types, covering every platform's convention:
    # ARCHIVE is only reached here on Windows (a SHARED target there
    # also produces a .lib import library); LIBRARY is where the
    # actual libblas1.so lands on Linux/macOS; RUNTIME is where a
    # Windows .dll would go. Harmless to list all three unconditionally
    # -- each only applies on the platforms where that artifact exists.
    install(TARGETS blas1_shared
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(DIRECTORY include/blas1
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Generate blas1.pc from the template at configure time (values like
# CMAKE_INSTALL_PREFIX are only known then) and install it to the
# standard pkg-config search location under the chosen libdir.
configure_file(
    ${CMAKE_SOURCE_DIR}/blas1.pc.in
    ${CMAKE_BINARY_DIR}/blas1.pc
    @ONLY
)
install(FILES ${CMAKE_BINARY_DIR}/blas1.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ------------------------------------------------------------------ #
#  12. Summary print — shown every time cmake configures              #
# ------------------------------------------------------------------ #
message(STATUS "-------------------------------")
message(STATUS "BLAS1 configuration summary")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "  C standard : C${CMAKE_C_STANDARD}")
message(STATUS "  MPI layer  : ${BLAS1_BUILD_MPI}")
message(STATUS "  Shared lib : ${BLAS1_BUILD_SHARED}")
message(STATUS "  Install to : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "-------------------------------")

