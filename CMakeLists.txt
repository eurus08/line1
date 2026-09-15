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
#  6. Static library target                                            #
#     This produces:  build/libblas1.a                                #
# ------------------------------------------------------------------ #
add_library(blas1 STATIC ${BLAS1_SOURCES})

# Tell CMake where our public headers live.
# SYSTEM suppresses warnings from our own headers in consuming targets.
target_include_directories(blas1
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Apply compiler flags defined in cmake/CompilerFlags.cmake
target_apply_compiler_flags(blas1)
target_apply_arch_flags(blas1) 

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
endif()

# ------------------------------------------------------------------ #
#  10. Summary print — shown every time cmake configures              #
# ------------------------------------------------------------------ #
message(STATUS "-------------------------------")
message(STATUS "BLAS1 configuration summary")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "  C standard : C${CMAKE_C_STANDARD}")
message(STATUS "  MPI layer  : ${BLAS1_BUILD_MPI}")
message(STATUS "  Install to : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "-------------------------------")

