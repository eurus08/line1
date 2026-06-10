# cmake/DetectArch.cmake
#
# Detects the host CPU architecture and sets compile definitions so
# the right SIMD kernel is selected at build time (Phase 5).
#
# What it sets (as preprocessor defines on the blas1 target):
#
#   USE_AVX2   — x86_64 CPU with AVX2 + FMA support (Intel Haswell+,
#                AMD Ryzen+). Processes 4 doubles / 8 floats per cycle.
#
#   USE_NEON   — ARM CPU with NEON support (Apple M-series, AWS Graviton,
#                Raspberry Pi 4+). Processes 2 doubles / 4 floats per cycle.
#
#   Neither    — generic scalar fallback (works everywhere, no SIMD)
#
# The detection works in two stages:
#   1. Check CMAKE_SYSTEM_PROCESSOR to identify the architecture family
#   2. Try to compile a small test program using the SIMD instructions
#      to confirm the compiler + CPU actually support them
#
# This file only sets variables/defines — it does not create targets.
# Call it with:  include(DetectArch)
# Then apply with: target_apply_arch_flags(<target>)

# ------------------------------------------------------------------ #
#  Guard                                                               #
# ------------------------------------------------------------------ #
if(DEFINED BLAS1_DETECT_ARCH_INCLUDED)
    return()
endif()
set(BLAS1_DETECT_ARCH_INCLUDED TRUE)

include(CheckCSourceCompiles)   # built-in CMake module for compile tests

# ------------------------------------------------------------------ #
#  Stage 1 — Identify architecture family                              #
# ------------------------------------------------------------------ #
set(BLAS1_ARCH_X86   FALSE)
set(BLAS1_ARCH_ARM   FALSE)

message(STATUS "Detecting CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
    set(BLAS1_ARCH_X86 TRUE)
    message(STATUS "Architecture family: x86 / x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8|ARM64")
    set(BLAS1_ARCH_ARM TRUE)
    message(STATUS "Architecture family: ARM / AArch64")
else()
    message(STATUS "Architecture family: unknown (${CMAKE_SYSTEM_PROCESSOR}) — using generic fallback")
endif()

# ------------------------------------------------------------------ #
#  Stage 2 — Compile tests                                             #
#                                                                      #
#  Checking CMAKE_SYSTEM_PROCESSOR alone is not enough.               #
#  A CPU might be x86_64 but pre-Haswell (no AVX2).                   #
#  We try to compile a tiny program that uses the actual instructions. #
#  If it compiles, the compiler supports them.                         #
#  Combined with -march=native, the CPU will support them too.        #
# ------------------------------------------------------------------ #

set(BLAS1_HAS_AVX2  FALSE)
set(BLAS1_HAS_NEON  FALSE)

# --- AVX2 + FMA test (x86 only) ------------------------------------ #
if(BLAS1_ARCH_X86)

    # Save current flags, temporarily add AVX2 flag for the test
    set(_SAVED_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")

    check_c_source_compiles("
        #include <immintrin.h>
        int main(void) {
            __m256d a = _mm256_set1_pd(1.0);
            __m256d b = _mm256_set1_pd(2.0);
            __m256d c = _mm256_set1_pd(0.0);
            /* FMA: c = a*b + c  (fused multiply-add) */
            c = _mm256_fmadd_pd(a, b, c);
            (void)c;
            return 0;
        }
    " BLAS1_HAS_AVX2)

    # Restore original flags
    set(CMAKE_C_FLAGS "${_SAVED_CMAKE_C_FLAGS}")

    if(BLAS1_HAS_AVX2)
        message(STATUS "SIMD support: AVX2 + FMA detected")
    else()
        message(STATUS "SIMD support: AVX2 not available — using generic fallback")
    endif()

endif()

# --- NEON test (ARM only) ------------------------------------------ #
if(BLAS1_ARCH_ARM)

    check_c_source_compiles("
        #include <arm_neon.h>
        int main(void) {
            float64x2_t a = vdupq_n_f64(1.0);
            float64x2_t b = vdupq_n_f64(2.0);
            /* multiply two double vectors */
            float64x2_t c = vmulq_f64(a, b);
            (void)c;
            return 0;
        }
    " BLAS1_HAS_NEON)

    if(BLAS1_HAS_NEON)
        message(STATUS "SIMD support: NEON (AArch64) detected")
    else()
        message(STATUS "SIMD support: NEON not available — using generic fallback")
    endif()

endif()

# ------------------------------------------------------------------ #
#  Stage 3 — Summarise result                                          #
# ------------------------------------------------------------------ #
set(BLAS1_SIMD_BACKEND "generic")

if(BLAS1_HAS_AVX2)
    set(BLAS1_SIMD_BACKEND "avx2")
elseif(BLAS1_HAS_NEON)
    set(BLAS1_SIMD_BACKEND "neon")
endif()

message(STATUS "SIMD backend selected: ${BLAS1_SIMD_BACKEND}")

# ------------------------------------------------------------------ #
#  Public function                                                      #
#                                                                      #
#  target_apply_arch_flags(<target>)                                  #
#                                                                      #
#  Adds the correct USE_AVX2 / USE_NEON / USE_GENERIC preprocessor    #
#  define to the target. Your .c files check these with #if defined(). #
# ------------------------------------------------------------------ #
function(target_apply_arch_flags target)

    if(BLAS1_HAS_AVX2)
        target_compile_definitions(${target} PRIVATE USE_AVX2=1)
        # AVX2 requires these flags to actually emit the instructions
        target_compile_options(${target} PRIVATE -mavx2 -mfma)

    elseif(BLAS1_HAS_NEON)
        target_compile_definitions(${target} PRIVATE USE_NEON=1)
        # NEON is always available on AArch64 — no extra flag needed

    else()
        target_compile_definitions(${target} PRIVATE USE_GENERIC=1)
    endif()

endfunction()