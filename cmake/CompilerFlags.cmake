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
# This is standard practice in HPC for raw throughput -- OpenBLAS and
# MKL both use it -- but it is NOT this library's default, and that is
# a deliberate choice, not an oversight: this library's whole pitch is
# careful, numerically robust arithmetic (Kahan-compensated summation,
# a scaled overflow/underflow-safe nrm2, explicit Inf/NaN handling --
# see dot.c/asum.c/nrm2.c). -ffast-math actively undermines exactly
# that: -ffinite-math-only (part of -ffast-math) is entitled to assume
# no value is ever Inf/NaN, and confirmed here (by inspecting the
# actual generated assembly, not just reasoned about) to act on that
# assumption by eliminating the Inf/NaN-handling branches this library
# depends on -- and floating-point reassociation is confirmed to
# reduce blas_dot_kahan()'s Kahan compensation to bit-identical output
# with the uncompensated blas_dot(), silently making the "Kahan"
# variant do nothing useful for the extra arithmetic it costs. Shipping
# that as the default, with no clear signal to the caller, would
# contradict this project's own correctness pitch. Benchmarks (bench/)
# still build with -ffast-math unconditionally, since raw best-case
# throughput is exactly what a benchmark should report -- this default
# is about the LIBRARY build, which callers actually link against.
#
# Opt in to -ffast-math with -DBLAS1_STRICT_IEEE=OFF once you have
# specifically decided the throughput is worth those trade-offs for
# your use case. If you need reproducible bit-exact results across
# compilers, the default (strict IEEE 754) already gives you that.
#
option(BLAS1_STRICT_IEEE
    "Strict IEEE 754 compliance (no -ffast-math). Set to OFF to opt in to -ffast-math for maximum throughput." ON)

set(_BLAS1_FLAGS_MATH "")

if(NOT BLAS1_STRICT_IEEE)
    if(IS_GCC OR IS_CLANG)
        list(APPEND _BLAS1_FLAGS_MATH -ffast-math)
    endif()
    message(STATUS "Fast math: ON  (opted in via -DBLAS1_STRICT_IEEE=OFF -- "
                   "Kahan compensation and Inf/NaN handling are not guaranteed "
                   "under this flag; see CompilerFlags.cmake)")
else()
    message(STATUS "Fast math: OFF (strict IEEE 754 mode -- default; "
                   "use -DBLAS1_STRICT_IEEE=OFF to opt in to -ffast-math)")
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

    # Sanitizer flags (-fsanitize=address,undefined) are unlike ordinary
    # compiler flags: the LINKER needs them too, not just the compiler,
    # because they pull in libasan/libubsan's runtime support at link
    # time.
    #
    # PUBLIC (not PRIVATE) is required here, for a subtle reason: PRIVATE
    # link options on a STATIC library have NO EFFECT AT ALL, because
    # CMake never invokes a linker for a static archive (it calls `ar`,
    # not `ld`) -- there is no link step for a PRIVATE flag to attach to.
    # PUBLIC link options, in contrast, are propagated as a USAGE
    # REQUIREMENT to any target that links against this one, even though
    # this target itself never "links". That propagation is exactly what
    # makes -fsanitize=... appear on, say, test_dot's link command when
    # test_dot links the blas1 static library -- without it, every
    # executable linking a sanitizer-instrumented static library fails
    # with "undefined reference to __asan_report_load8" and similar,
    # regardless of what flags that executable's own source was compiled
    # with.
    target_link_options(${target} PUBLIC
        $<$<CONFIG:Debug>:${_BLAS1_FLAGS_DEBUG}>
    )

endfunction()