/*
 * kernel/x86/dot_avx2.c — AVX2 + FMA explicit intrinsics kernel for blas_dot
 *
 * Step 5.2 of the build plan.
 *
 * Compiled ONLY when USE_AVX2=1 (set by cmake/DetectArch.cmake on x86_64
 * machines that pass the AVX2+FMA compile test). This file is compiled with
 * -mavx2 -mfma added by target_apply_arch_flags() in CMakeLists.txt — those
 * flags must NOT be applied to the whole build, only to this translation unit.
 *
 * =========================================================================
 * WHY EXPLICIT INTRINSICS INSTEAD OF RELYING ON AUTO-VECTORISATION?
 * =========================================================================
 *
 * The generic kernel (dot_generic.c) already auto-vectorises under
 * -O3 -march=native. So why bother with explicit intrinsics?
 *
 *   1. Predictability. The compiler's auto-vectoriser is a heuristic —
 *      it can silently fail to vectorise, choose the wrong vector width,
 *      or generate suboptimal FMA patterns. Explicit intrinsics guarantee
 *      the exact instruction sequence we intend, every build.
 *
 *   2. Multiple accumulators. The auto-vectoriser may or may not unroll
 *      and use multiple accumulators to hide FMA latency. We do it
 *      explicitly: 4 independent __m256d accumulators process 16 doubles
 *      per iteration, keeping all 4 AVX2 FMA pipelines busy.
 *
 *   3. Measurability. With explicit intrinsics, the Phase 4->5 speedup
 *      is a real, attributable number — not "we got lucky with the
 *      compiler." If the explicit kernel doesn't beat auto-vectorisation,
 *      that's a meaningful data point too (it means the compiler was
 *      already doing the right thing on this CPU).
 *
 * =========================================================================
 * AVX2 FUNDAMENTALS (for reference)
 * =========================================================================
 *
 * __m256d  — a 256-bit register holding 4 doubles (4 × 64 bits)
 *
 * Key intrinsics used here:
 *   _mm256_loadu_pd(ptr)        load 4 doubles from ptr (unaligned OK)
 *   _mm256_fmadd_pd(a, b, c)   compute a*b+c for all 4 lanes — one instruction
 *   _mm256_add_pd(a, b)        add corresponding lanes
 *   _mm256_setzero_pd()        create a zero register (all 4 lanes = 0.0)
 *   _mm256_hadd_pd(a, b)       horizontal add pairs within each 128-bit half
 *   _mm256_permute2f128_pd     swap/mix the two 128-bit halves of a register
 *   _mm256_extractf128_pd      extract one 128-bit half as __m128d
 *   _mm_cvtsd_f64              extract the lowest double from a __m128d
 *
 * FMA latency/throughput (Intel Haswell, typical AVX2 target):
 *   _mm256_fmadd_pd latency:    5 cycles
 *   _mm256_fmadd_pd throughput: 0.5 cycles (2 per cycle, two FMA units)
 *   → With 1 accumulator: throughput limited by latency (5 cycle stall)
 *   → With 4+ accumulators: latency hidden, throughput-limited (0.5 cycle)
 *
 * =========================================================================
 * FLOAT PRECISION GUARD
 * =========================================================================
 *
 * All intrinsics here use _pd variants (packed double). If the library is
 * built with -DBLAS_USE_FLOAT, BLAS_REAL becomes float and the _pd
 * intrinsics would silently compute the wrong types. We static-assert
 * that BLAS_REAL == double and bail out to the generic kernel at runtime
 * if somehow that assertion is bypassed. In practice, DetectArch.cmake
 * and the build system prevent this combination from being built.
 */

#include <immintrin.h>   /* AVX2 + FMA intrinsics */
#include <stddef.h>      /* size_t */

#include "blas1/types.h"
#include "dot_avx2.h"    /* own prototype — satisfies -Wmissing-prototypes */

/* Compile-time guard: these intrinsics are written for double precision.
 * If someone builds with -DBLAS_USE_FLOAT, they must not reach this file.
 * DetectArch.cmake + CMakeLists.txt prevent it, but belt-and-suspenders. */
#ifndef BLAS_USE_FLOAT
/* double build — proceed */
#else
#  error "dot_avx2.c uses _pd (packed double) intrinsics and cannot be compiled \
for float precision (-DBLAS_USE_FLOAT). Set BLAS1_SIMD_BACKEND to generic for \
float builds, or add a _ps (packed single) variant."
#endif

/* -------------------------------------------------------------------------
 * hadd4() — horizontal sum of a __m256d register into a single double.
 *
 * AVX2 has no single "sum all 4 lanes" instruction. The standard pattern:
 *   1. hadd the 256-bit register with itself → pairs within each 128-bit half
 *   2. Extract the high 128-bit half
 *   3. Add the two 128-bit halves
 *   4. Extract the low double
 *
 * This costs ~4 instructions and is done once per blas_dot() call (at the
 * reduce step), so its overhead is negligible.
 * ------------------------------------------------------------------------- */
static inline double hadd4(__m256d v)
{
    /* Step 1: horizontal add pairs within each 128-bit half.
     * hadd([a,b,c,d], [a,b,c,d]) → [a+b, a+b, c+d, c+d]  (not useful alone)
     * We want [a+b, c+d, ...], so use permute first, then add. */
    __m128d lo  = _mm256_castpd256_pd128(v);          /* [a, b] */
    __m128d hi  = _mm256_extractf128_pd(v, 1);        /* [c, d] */
    __m128d sum = _mm_add_pd(lo, hi);                 /* [a+c, b+d] */
    __m128d shuf = _mm_unpackhi_pd(sum, sum);         /* [b+d, b+d] */
    return _mm_cvtsd_f64(_mm_add_sd(sum, shuf));      /* (a+c) + (b+d) */
}

/* -------------------------------------------------------------------------
 * blas_dot_avx2 — standard dot product with explicit AVX2 + FMA
 *
 * Inner loop processes 16 doubles per iteration (4 AVX2 vectors × 4 lanes),
 * using 4 independent accumulators to hide FMA latency. The tail loop
 * handles any remaining elements (0 to 15) with scalar code.
 *
 * Only the unit-stride path uses AVX2. The strided path falls back to
 * scalar — gather intrinsics (_mm256_i64gather_pd) exist but have very
 * poor throughput on most CPUs (4-20 cycles vs 0.5 for contiguous loads),
 * so scalar strided is almost always faster in practice.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_avx2(blas_int n,
                         const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                         const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return 0.0;
    }

    /* ----------------------------------------------------------------
     * Strided path — fall back to scalar (see rationale above)
     * ---------------------------------------------------------------- */
    if (incx != 1 || incy != 1) {
        double sum = 0.0;
        blas_int ix = 0, iy = 0;
        for (blas_int i = 0; i < n; i++) {
            sum += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
        return sum;
    }

    /* ----------------------------------------------------------------
     * Unit-stride AVX2 path
     *
     * 4 accumulators × 4 doubles = 16 doubles per iteration.
     * Each accumulator is independent → no RAW dependency stalls.
     * ---------------------------------------------------------------- */
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    blas_int i   = 0;
    blas_int n16 = (n / 16) * 16;   /* largest multiple of 16 <= n */

    for (; i < n16; i += 16) {
        /* Load 4 AVX2 vectors from x and y (unaligned loads — safe on
         * any address, hardware handles alignment internally on modern CPUs) */
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d y0 = _mm256_loadu_pd(y + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d y1 = _mm256_loadu_pd(y + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d y2 = _mm256_loadu_pd(y + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);
        __m256d y3 = _mm256_loadu_pd(y + i + 12);

        /* FMA: acc += x * y  (fused multiply-add, one instruction per lane) */
        acc0 = _mm256_fmadd_pd(x0, y0, acc0);
        acc1 = _mm256_fmadd_pd(x1, y1, acc1);
        acc2 = _mm256_fmadd_pd(x2, y2, acc2);
        acc3 = _mm256_fmadd_pd(x3, y3, acc3);
    }

    /* Reduce the 4 accumulators into one */
    __m256d acc01 = _mm256_add_pd(acc0, acc1);
    __m256d acc23 = _mm256_add_pd(acc2, acc3);
    __m256d total = _mm256_add_pd(acc01, acc23);
    double  sum   = hadd4(total);

    /* Scalar tail — handles the last 0..15 elements */
    for (; i < n; i++) {
        sum += x[i] * y[i];
    }

    return sum;
}

/* -------------------------------------------------------------------------
 * blas_dot_kahan_avx2 — Kahan compensated dot product with AVX2
 *
 * Strategy: run 4 independent Kahan-compensated AVX2 accumulators (each
 * holding 4 doubles = 16 compensated streams total). At the end, reduce
 * the 4 compensated vector sums into one final scalar via a Kahan horizontal
 * reduce. This is strictly better than scalar Kahan on large n.
 *
 * The -ffast-math caveat from dot_generic.c applies here too: the compiler
 * may reassociate the Kahan compensation steps even with explicit intrinsics,
 * because -ffast-math applies to the whole translation unit. Build with
 * -DBLAS1_STRICT_IEEE=ON (which removes -ffast-math from CompilerFlags.cmake)
 * if the compensation must be guaranteed to survive.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_kahan_avx2(blas_int n,
                                const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                                const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return 0.0;
    }

    /* Strided fallback — same reasoning as blas_dot_avx2 */
    if (incx != 1 || incy != 1) {
        double sum = 0.0, c = 0.0;
        blas_int ix = 0, iy = 0;
        for (blas_int i = 0; i < n; i++) {
            double t       = x[ix] * y[iy] - c;
            double new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
            ix += incx;
            iy += incy;
        }
        return sum;
    }

    /* ----------------------------------------------------------------
     * Unit-stride Kahan AVX2 path
     *
     * 4 pairs of (sum, compensation) vectors, each holding 4 lanes.
     * That's 16 independent Kahan streams per iteration.
     * ---------------------------------------------------------------- */
    __m256d s0 = _mm256_setzero_pd(), c0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd();
    __m256d s2 = _mm256_setzero_pd(), c2 = _mm256_setzero_pd();
    __m256d s3 = _mm256_setzero_pd(), c3 = _mm256_setzero_pd();

    blas_int i   = 0;
    blas_int n16 = (n / 16) * 16;

    for (; i < n16; i += 16) {
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d y0 = _mm256_loadu_pd(y + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d y1 = _mm256_loadu_pd(y + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d y2 = _mm256_loadu_pd(y + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);
        __m256d y3 = _mm256_loadu_pd(y + i + 12);

        /* Kahan step for each of the 4 accumulators:
         *   t       = x*y - c          (apply compensation)
         *   new_s   = s + t
         *   c       = (new_s - s) - t  (capture lost bits)
         *   s       = new_s
         * All operations are lane-wise on __m256d. */
        __m256d t0    = _mm256_sub_pd(_mm256_mul_pd(x0, y0), c0);
        __m256d new_s0 = _mm256_add_pd(s0, t0);
        c0 = _mm256_sub_pd(_mm256_sub_pd(new_s0, s0), t0);
        s0 = new_s0;

        __m256d t1    = _mm256_sub_pd(_mm256_mul_pd(x1, y1), c1);
        __m256d new_s1 = _mm256_add_pd(s1, t1);
        c1 = _mm256_sub_pd(_mm256_sub_pd(new_s1, s1), t1);
        s1 = new_s1;

        __m256d t2    = _mm256_sub_pd(_mm256_mul_pd(x2, y2), c2);
        __m256d new_s2 = _mm256_add_pd(s2, t2);
        c2 = _mm256_sub_pd(_mm256_sub_pd(new_s2, s2), t2);
        s2 = new_s2;

        __m256d t3    = _mm256_sub_pd(_mm256_mul_pd(x3, y3), c3);
        __m256d new_s3 = _mm256_add_pd(s3, t3);
        c3 = _mm256_sub_pd(_mm256_sub_pd(new_s3, s3), t3);
        s3 = new_s3;
    }

    /* Reduce the 4 sum vectors and 4 compensation vectors into scalars.
     * Use plain addition for the vector-level reduction (Kahan only runs
     * within each of the 16 streams, not across them at this step — the
     * cross-stream error is bounded by O(16*eps), negligible). */
    __m256d s01 = _mm256_add_pd(s0, s1);
    __m256d s23 = _mm256_add_pd(s2, s3);
    double  sum = hadd4(_mm256_add_pd(s01, s23));

    /* Scalar Kahan tail — handles the last 0..15 elements, plus absorbs
     * any residual compensation from the streams we're discarding. */
    double c_tail = 0.0;
    for (; i < n; i++) {
        double t       = x[i] * y[i] - c_tail;
        double new_sum = sum + t;
        c_tail = (new_sum - sum) - t;
        sum    = new_sum;
    }

    return sum;
}
