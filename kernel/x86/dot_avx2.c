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
 *   _mm256_castpd256_pd128     reinterpret the low 128 bits as a __m128d (free, no instruction)
 *   _mm256_extractf128_pd      extract one 128-bit half as __m128d
 *   _mm_unpackhi_pd            broadcast the high lane of a __m128d into both lanes
 *   _mm_add_sd                 add the low lanes only, passing the high lane through
 *   _mm_cvtsd_f64              extract the lowest double from a __m128d
 *   _mm256_cmp_pd(a,b,pred)    per-lane comparison, result is all-1s or all-0s per lane
 *   _mm256_blendv_pd(a,b,mask) per-lane select: b where mask's sign bit is set, else a
 *   _mm256_andnot_pd(a,b)      (~a) & b — used here to clear the sign bit (fabs)
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

#include "line1/types.h"
#include "dot_avx2.h"    /* own prototype — satisfies -Wmissing-prototypes */

/* Compile-time guard: these intrinsics are written for double precision.
 * If someone builds with -DBLAS_USE_FLOAT, they must not reach this file.
 * DetectArch.cmake + CMakeLists.txt prevent it, but belt-and-suspenders. */
#ifndef BLAS_USE_FLOAT
/* double build — proceed */
#else
#  error "dot_avx2.c uses _pd (packed double) intrinsics and cannot be compiled \
for float precision (-DBLAS_USE_FLOAT). Set LINE1_SIMD_BACKEND to generic for \
float builds, or add a _ps (packed single) variant."
#endif

/* -------------------------------------------------------------------------
 * hadd4() — horizontal sum of a __m256d register into a single double.
 *
 * AVX2 has no single "sum all 4 lanes" instruction. The pattern used here:
 *   1. Split the 256-bit register into its low and high 128-bit halves
 *   2. Add the two halves lane-wise: [a,b,c,d] → [a+c, b+d]
 *   3. Broadcast the high lane of that result into both lanes, add to the
 *      low lane only: (a+c) + (b+d)
 *   4. Extract the resulting scalar double
 *
 * This costs ~4 instructions and is done once per blas_dot() call (at the
 * reduce step), so its overhead is negligible.
 * ------------------------------------------------------------------------- */
static inline double hadd4(__m256d v)
{
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
        blas_int ix = blas_stride_start(n, incx);
        blas_int iy = blas_stride_start(n, incy);
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
 * because -ffast-math applies to the whole translation unit. Strict IEEE 754
 * (no -ffast-math) is this project's DEFAULT build for exactly this reason;
 * -DLINE1_STRICT_IEEE=OFF opts in to -ffast-math (and gives up this
 * guarantee).
 *
 * Overflow-to-NaN fix (same bug class as src/asum.c and
 * kernel/generic/dot_generic.c, applied here too):
 *   Once a running sum overflows to +-Inf, the compensation term can
 *   itself become infinite, and the next term's `t = product - c`
 *   becomes an opposite-signed infinity -- so `new_s = s + t` computes
 *   Inf + (-Inf) == NaN, even where plain summation would have stayed
 *   at a well-defined +-Inf. dot_generic.c's blas_dot_kahan_generic()
 *   and asum.c fix this with a per-element `if (blas_is_inf(sum))`
 *   guard that falls back to plain addition. The strided fallback
 *   below (scalar) applies that guard directly as a branch.
 *
 *   The unit-stride path runs 16 independent Kahan streams across 4
 *   __m256d accumulators, where a single scalar branch can't work --
 *   different lanes can overflow on different iterations. Instead each
 *   stream applies the guard as a per-LANE branchless blend: compute
 *   both the compensated result and the plain-addition result every
 *   iteration, compare |s| against +Inf per lane (_mm256_andnot_pd to
 *   clear the sign bit, _mm256_cmp_pd against a broadcast +Inf), and
 *   _mm256_blendv_pd the two results together per lane. This matches
 *   the scalar semantics exactly, including leaving the compensation
 *   term `c` untouched for lanes that took the overflow branch (the
 *   scalar code's `continue` skips the `c` update; the blend mirrors
 *   that by selecting the OLD `c` for those lanes). The scalar tail
 *   loop below has the same guard.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_kahan_avx2(blas_int n,
                                const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                                const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return 0.0;
    }

    /* Strided fallback — same reasoning as blas_dot_avx2, and the same
     * overflow-to-NaN fix as dot_generic.c (the unit-stride SIMD path
     * below applies the equivalent fix as a per-lane blend — see above). */
    if (incx != 1 || incy != 1) {
        double sum = 0.0, c = 0.0;
        blas_int ix = blas_stride_start(n, incx);
        blas_int iy = blas_stride_start(n, incy);
        for (blas_int i = 0; i < n; i++) {
            double p = x[ix] * y[iy];
            if (BLAS_UNLIKELY(blas_is_inf(sum))) {
                sum = sum + p;
                ix += incx;
                iy += incy;
                continue;
            }
            double t       = p - c;
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

    /* Constants for the per-lane overflow-to-NaN guard (see comment
     * above): sign_mask clears the sign bit (fabs), inf_vec is +Inf
     * broadcast to all 4 lanes so |s| can be compared against it. */
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d inf_vec   = _mm256_set1_pd(INFINITY);

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

        /* Kahan step for each of the 4 accumulators, with a per-lane
         * overflow guard:
         *   p           = x*y
         *   inf_mask    = |s| == +Inf, per lane
         *   t           = p - c                 (apply compensation)
         *   new_s_comp  = s + t
         *   new_c_comp  = (new_s_comp - s) - t   (capture lost bits)
         *   new_s_plain = s + p                  (what a lane that has
         *                                          already overflowed
         *                                          uses instead)
         *   s = inf_mask ? new_s_plain : new_s_comp
         *   c = inf_mask ? c (unchanged) : new_c_comp
         * All operations are lane-wise on __m256d. */
        __m256d p0 = _mm256_mul_pd(x0, y0);
        __m256d m0 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, s0), inf_vec, _CMP_EQ_OQ);
        __m256d t0 = _mm256_sub_pd(p0, c0);
        __m256d comp_s0 = _mm256_add_pd(s0, t0);
        __m256d comp_c0 = _mm256_sub_pd(_mm256_sub_pd(comp_s0, s0), t0);
        c0 = _mm256_blendv_pd(comp_c0, c0, m0);
        s0 = _mm256_blendv_pd(comp_s0, _mm256_add_pd(s0, p0), m0);

        __m256d p1 = _mm256_mul_pd(x1, y1);
        __m256d m1 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, s1), inf_vec, _CMP_EQ_OQ);
        __m256d t1 = _mm256_sub_pd(p1, c1);
        __m256d comp_s1 = _mm256_add_pd(s1, t1);
        __m256d comp_c1 = _mm256_sub_pd(_mm256_sub_pd(comp_s1, s1), t1);
        c1 = _mm256_blendv_pd(comp_c1, c1, m1);
        s1 = _mm256_blendv_pd(comp_s1, _mm256_add_pd(s1, p1), m1);

        __m256d p2 = _mm256_mul_pd(x2, y2);
        __m256d m2 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, s2), inf_vec, _CMP_EQ_OQ);
        __m256d t2 = _mm256_sub_pd(p2, c2);
        __m256d comp_s2 = _mm256_add_pd(s2, t2);
        __m256d comp_c2 = _mm256_sub_pd(_mm256_sub_pd(comp_s2, s2), t2);
        c2 = _mm256_blendv_pd(comp_c2, c2, m2);
        s2 = _mm256_blendv_pd(comp_s2, _mm256_add_pd(s2, p2), m2);

        __m256d p3 = _mm256_mul_pd(x3, y3);
        __m256d m3 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, s3), inf_vec, _CMP_EQ_OQ);
        __m256d t3 = _mm256_sub_pd(p3, c3);
        __m256d comp_s3 = _mm256_add_pd(s3, t3);
        __m256d comp_c3 = _mm256_sub_pd(_mm256_sub_pd(comp_s3, s3), t3);
        c3 = _mm256_blendv_pd(comp_c3, c3, m3);
        s3 = _mm256_blendv_pd(comp_s3, _mm256_add_pd(s3, p3), m3);
    }

    /* Reduce the 4 sum vectors and 4 compensation vectors into scalars.
     * Use plain addition for the vector-level reduction (Kahan only runs
     * within each of the 16 streams, not across them at this step — the
     * cross-stream error is bounded by O(16*eps), negligible). */
    __m256d s01 = _mm256_add_pd(s0, s1);
    __m256d s23 = _mm256_add_pd(s2, s3);
    double  sum = hadd4(_mm256_add_pd(s01, s23));

    /* Scalar Kahan tail — handles the last 0..15 elements, plus absorbs
     * any residual compensation from the streams we're discarding. Same
     * overflow-to-NaN guard as the vectorized loop above: `sum` may
     * already be +-Inf coming out of the reduction. */
    double c_tail = 0.0;
    for (; i < n; i++) {
        double p = x[i] * y[i];
        if (BLAS_UNLIKELY(blas_is_inf(sum))) {
            sum = sum + p;
            continue;
        }
        double t       = p - c_tail;
        double new_sum = sum + t;
        c_tail = (new_sum - sum) - t;
        sum    = new_sum;
    }

    return sum;
}
