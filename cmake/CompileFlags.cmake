# cmake/CompilerFlags.cmake
#
# Defines a function:  target_apply_compiler_flags(<target>)
#
# Call it on any CMake target (library or executable) to apply
# the standard BLAS1 compiler flags for that build type.
#
# Usage in CMakeLists.txt:
#   include(CompilerFlags)
#   target_apply_compiler_flags(blas1)

# ------------------------------------------------------------------ #
#  Guard — only process this file once per CMake run                  #
# ------------------------------------------------------------------ #
if(DEFINED BLAS1_COMPILER_FLAGS_INCLUDED)
    return()
endif()
set(BLAS1_COMPILER_FLAGS_INCLUDED TRUE)

# ------------------------------------------------------------------ #
#  Detect compiler family                                              #
# ------------------------------------------------------------------ #
set(IS_GCC    FALSE)
set(IS_CLANG  FALSE)
set(IS_MSVC   FALSE)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(IS_GCC TRUE)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(IS_CLANG TRUE)
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set(IS_MSVC TRUE)
endif()

message(STATUS "Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

# ------------------------------------------------------------------ #
#  Flag sets                                                           #
#                                                                      #
#  We define four lists:                                               #
#    _FLAGS_COMMON  — always applied (warnings, standards conformance) #
#    _FLAGS_RELEASE — applied in Release / RelWithDebInfo builds       #
#    _FLAGS_DEBUG   — applied in Debug builds                          #
#    _FLAGS_MATH    — floating-point model (documented separately)     #
# ------------------------------------------------------------------ #

# --- Common flags (all build types) -------------------------------- #
set(_BLAS1_FLAGS_COMMON "")

if(IS_GCC OR IS_CLANG)
    list(APPEND _BLAS1_FLAGS_COMMON
        -Wall                  # standard warnings
        -Wextra                # extra warnings (unused params, etc.)
        -Wpedantic             # strict ISO C conformance warnings
        -Wconversion           # warn on implicit type conversions
        -Wshadow               # warn when a local var shadows an outer one
        -Wdouble-promotion     # warn when float silently promotes to double
        -Wmissing-prototypes   # every function must have a declaration header
        -Wstrict-prototypes    # function declarations must have typed params
    )
endif()

# --- Release flags -------------------------------------------------- #
set(_BLAS1_FLAGS_RELEASE "")

if(IS_GCC OR IS_CLANG)
    list(APPEND _BLAS1_FLAGS_RELEASE
        -O3                    # maximum optimisation
        -march=native          # tune for the CPU this is compiled on
                               # (uses AVX2/FMA/NEON if available)
                               # WARNING: binaries are not portable —
                               # do not distribute a -march=native build
        -funroll-loops         # unroll small loops (helps BLAS kernels)
        -fomit-frame-pointer   # free up one register in tight loops
    )
endif()

# --- Math flags ----------------------------------------------------- #
#
# -ffast-math allows the compiler to:
#   - reorder floating-point operations (breaks strict IEEE 754)
#   - assume no NaN / Inf in input
#   - replace division with multiply-by-reciprocal
#
# This is standard practice in HPC — OpenBLAS and MKL both use it.
# We enable it for Release only and document it clearly so users
# know their results may differ slightly from a strict IEEE build.
#
# If you need reproducible bit-exact results across compilers,
# build with -DBLAS1_STRICT_IEEE=ON to disable this flag.
#
option(BLAS1_STRICT_IEEE
    "Disable -ffast-math for strict IEEE 754 compliance" OFF)

set(_BLAS1_FLAGS_MATH "")

if(NOT BLAS1_STRICT_IEEE)
    if(IS_GCC OR IS_CLANG)
        list(APPEND _BLAS1_FLAGS_MATH -ffast-math)
    endif()
    message(STATUS "Fast math: ON  (use -DBLAS1_STRICT_IEEE=ON to disable)")
else()
    message(STATUS "Fast math: OFF (strict IEEE 754 mode)")
endif()

# --- Debug flags ---------------------------------------------------- #
set(_BLAS1_FLAGS_DEBUG "")

if(IS_GCC OR IS_CLANG)
    list(APPEND _BLAS1_FLAGS_DEBUG
        -O0                    # no optimisation — debugger sees real code
        -g3                    # maximum debug info (includes macros)
        -fsanitize=address     # AddressSanitizer: catches buffer overflows,
                               # use-after-free, stack overflows
        -fsanitize=undefined   # UBSanitizer: catches signed overflow,
                               # null pointer deref, misaligned access
        -fno-omit-frame-pointer  # required for sanitizer stack traces
    )
endif()

# ------------------------------------------------------------------ #
#  The public function                                                 #
#                                                                      #
#  target_apply_compiler_flags(<target>)                              #
#                                                                      #
#  Applies the right flags for the current build type to <target>.    #
#  Uses PRIVATE so flags don't leak to consumers of the library.      #
# ------------------------------------------------------------------ #
function(target_apply_compiler_flags target)

    # Common flags — always
    target_compile_options(${target} PRIVATE ${_BLAS1_FLAGS_COMMON})

    # Release and RelWithDebInfo
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Release>:${_BLAS1_FLAGS_RELEASE}>
        $<$<CONFIG:Release>:${_BLAS1_FLAGS_MATH}>
        $<$<CONFIG:RelWithDebInfo>:${_BLAS1_FLAGS_RELEASE}>
        $<$<CONFIG:RelWithDebInfo>:${_BLAS1_FLAGS_MATH}>
    )

    # Debug
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Debug>:${_BLAS1_FLAGS_DEBUG}>
    )

endfunction()