/*
 * kernel/arm/dot_neon.c — NEON intrinsics kernel for blas_dot
 *
 * Step 5.4 of the build plan.
 *
 * Compiled ONLY when USE_NEON=1 (set by cmake/DetectArch.cmake on AArch64
 * machines that pass the NEON compile test — Apple M-series, AWS Graviton,
 * Raspberry Pi 4+). NEON is mandatory on AArch64, so no extra compiler
 * flag is required (unlike AVX2 which needs -mavx2 -mfma).
 *
 * =========================================================================
 * NEON FUNDAMENTALS vs AVX2
 * =========================================================================
 *
 * AArch64 NEON uses 128-bit registers (vs AVX2's 256-bit).
 * For double precision: 2 doubles per register (vs AVX2's 4).
 *
 * float64x2_t  — a 128-bit register holding 2 doubles
 *
 * Key intrinsics:
 *   vld1q_f64(ptr)           load 2 doubles from ptr
 *   vfmaq_f64(acc, a, b)    acc += a*b  for both lanes (FMA)
 *   vaddq_f64(a, b)         add corresponding lanes
 *   vdupq_n_f64(val)        broadcast scalar to both lanes
 *   vgetq_lane_f64(v, i)    extract lane i as scalar double
 *   vaddvq_f64(v)           horizontal add (AArch64 only, not ARMv7)
 *
 * Strategy: 4 independent float64x2_t accumulators = 8 doubles per
 * iteration, hiding FMA latency. Same principle as AVX2, scaled down
 * for 128-bit registers.
 *
 * =========================================================================
 * FLOAT PRECISION GUARD
 * =========================================================================
 *
 * float64x2_t is double-precision NEON. For float builds, float32x4_t
 * would be needed (4 floats per register). We guard against the mismatch.
 */

#include <arm_neon.h>    /* NEON intrinsics */

#include "blas1/types.h"
#include "dot_neon.h"    /* own prototype — satisfies -Wmissing-prototypes */

#ifndef BLAS_USE_FLOAT
/* double build — proceed */
#else
#  error "dot_neon.c uses float64x2_t (double) NEON intrinsics and cannot be \
compiled for float precision (-DBLAS_USE_FLOAT). Add a float32x4_t variant \
or use the generic fallback for float builds."
#endif

/* -------------------------------------------------------------------------
 * hadd2() — horizontal sum of a float64x2_t into a single double.
 * Two lanes: lane[0] + lane[1].
 * vaddvq_f64 does this in one instruction on AArch64.
 * ------------------------------------------------------------------------- */
static inline double hadd2(float64x2_t v)
{
    return vaddvq_f64(v);
}

/* -------------------------------------------------------------------------
 * blas_dot_neon — standard dot product with explicit NEON
 *
 * 4 float64x2_t accumulators × 2 lanes = 8 doubles per iteration.
 * Scalar tail handles the remaining 0..7 elements.
 * Strided path falls back to scalar (gather is slow on ARM too).
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_neon(blas_int n,
                         const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                         const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return 0.0;
    }

    /* Strided fallback */
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
     * Unit-stride NEON path — 4 accumulators × 2 lanes = 8 per iter
     * ---------------------------------------------------------------- */
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    blas_int i  = 0;
    blas_int n8 = (n / 8) * 8;

    for (; i < n8; i += 8) {
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t y0 = vld1q_f64(y + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t y1 = vld1q_f64(y + i + 2);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t y2 = vld1q_f64(y + i + 4);
        float64x2_t x3 = vld1q_f64(x + i + 6);
        float64x2_t y3 = vld1q_f64(y + i + 6);

        /* vfmaq_f64(acc, a, b) = acc + a*b  (fused, one instruction) */
        acc0 = vfmaq_f64(acc0, x0, y0);
        acc1 = vfmaq_f64(acc1, x1, y1);
        acc2 = vfmaq_f64(acc2, x2, y2);
        acc3 = vfmaq_f64(acc3, x3, y3);
    }

    /* Reduce 4 vector accumulators */
    float64x2_t acc01  = vaddq_f64(acc0, acc1);
    float64x2_t acc23  = vaddq_f64(acc2, acc3);
    double      sum    = hadd2(vaddq_f64(acc01, acc23));

    /* Scalar tail */
    for (; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

/* -------------------------------------------------------------------------
 * blas_dot_kahan_neon — Kahan compensated dot product with NEON
 *
 * Same structure as blas_dot_kahan_avx2: 4 independent Kahan-compensated
 * float64x2_t accumulator pairs, running 8 streams in parallel.
 * Same -ffast-math caveat applies — build with -DBLAS1_STRICT_IEEE=ON
 * if the compensation must survive the optimiser.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_kahan_neon(blas_int n,
                               const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                               const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return 0.0;
    }

    /* Strided fallback */
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
     * Unit-stride Kahan NEON path
     * ---------------------------------------------------------------- */
    float64x2_t s0 = vdupq_n_f64(0.0), c0 = vdupq_n_f64(0.0);
    float64x2_t s1 = vdupq_n_f64(0.0), c1 = vdupq_n_f64(0.0);
    float64x2_t s2 = vdupq_n_f64(0.0), c2 = vdupq_n_f64(0.0);
    float64x2_t s3 = vdupq_n_f64(0.0), c3 = vdupq_n_f64(0.0);

    blas_int i  = 0;
    blas_int n8 = (n / 8) * 8;

    for (; i < n8; i += 8) {
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t y0 = vld1q_f64(y + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t y1 = vld1q_f64(y + i + 2);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t y2 = vld1q_f64(y + i + 4);
        float64x2_t x3 = vld1q_f64(x + i + 6);
        float64x2_t y3 = vld1q_f64(y + i + 6);

        /* Kahan step (lane-wise NEON):
         *   t     = x*y - c
         *   new_s = s + t
         *   c     = (new_s - s) - t
         *   s     = new_s                  */
        float64x2_t t0     = vsubq_f64(vmulq_f64(x0, y0), c0);
        float64x2_t new_s0 = vaddq_f64(s0, t0);
        c0 = vsubq_f64(vsubq_f64(new_s0, s0), t0);
        s0 = new_s0;

        float64x2_t t1     = vsubq_f64(vmulq_f64(x1, y1), c1);
        float64x2_t new_s1 = vaddq_f64(s1, t1);
        c1 = vsubq_f64(vsubq_f64(new_s1, s1), t1);
        s1 = new_s1;

        float64x2_t t2     = vsubq_f64(vmulq_f64(x2, y2), c2);
        float64x2_t new_s2 = vaddq_f64(s2, t2);
        c2 = vsubq_f64(vsubq_f64(new_s2, s2), t2);
        s2 = new_s2;

        float64x2_t t3     = vsubq_f64(vmulq_f64(x3, y3), c3);
        float64x2_t new_s3 = vaddq_f64(s3, t3);
        c3 = vsubq_f64(vsubq_f64(new_s3, s3), t3);
        s3 = new_s3;
    }

    /* Reduce vector sums */
    float64x2_t s01 = vaddq_f64(s0, s1);
    float64x2_t s23 = vaddq_f64(s2, s3);
    double      sum = hadd2(vaddq_f64(s01, s23));

    /* Scalar Kahan tail */
    double c_tail = 0.0;
    for (; i < n; i++) {
        double t       = x[i] * y[i] - c_tail;
        double new_sum = sum + t;
        c_tail = (new_sum - sum) - t;
        sum    = new_sum;
    }
    return sum;
}
