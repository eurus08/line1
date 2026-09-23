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
 *   vaddvq_f64(v)           horizontal add (AArch64 only, not ARMv7)
 *   vabsq_f64(v)            per-lane absolute value
 *   vceqq_f64(a, b)         per-lane equality compare -> uint64x2_t mask
 *   vbslq_f64(mask, a, b)   per-lane select: a where mask bit set, else b
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

#include "line1/types.h"
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

    /* Strided fallback — supports negative incx/incy (blas_stride_start). */
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
 * Same -ffast-math caveat applies — strict IEEE 754 (no -ffast-math) is
 * this project's DEFAULT build for exactly this reason; -DLINE1_STRICT_IEEE=OFF
 * opts in to -ffast-math (and gives up this guarantee).
 *
 * Overflow-to-NaN fix: same bug class and fix strategy as dot_avx2.c's
 * blas_dot_kahan_avx2 (see its comment for the full explanation). The
 * scalar strided fallback applies the guard as a branch; the unit-stride
 * float64x2_t path applies it as a per-lane branchless select via
 * vceqq_f64()/vbslq_f64() (NEON's equivalent of AVX2's _mm256_cmp_pd /
 * _mm256_blendv_pd), computing both the compensated and plain-addition
 * results each iteration and selecting per lane based on whether that
 * lane's running sum has already overflowed to +-Inf.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_kahan_neon(blas_int n,
                               const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                               const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return 0.0;
    }

    /* Strided fallback -- overflow-to-NaN fix applied; the unit-stride
     * SIMD path below applies the equivalent fix as a per-lane select. */
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
     * Unit-stride Kahan NEON path
     * ---------------------------------------------------------------- */
    float64x2_t s0 = vdupq_n_f64(0.0), c0 = vdupq_n_f64(0.0);
    float64x2_t s1 = vdupq_n_f64(0.0), c1 = vdupq_n_f64(0.0);
    float64x2_t s2 = vdupq_n_f64(0.0), c2 = vdupq_n_f64(0.0);
    float64x2_t s3 = vdupq_n_f64(0.0), c3 = vdupq_n_f64(0.0);

    /* +Inf broadcast to both lanes, for the per-lane overflow guard
     * below (compared against |s| via vabsq_f64). */
    const float64x2_t inf_vec = vdupq_n_f64(INFINITY);

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

        /* Kahan step (lane-wise NEON), with a per-lane overflow guard:
         *   p           = x*y
         *   inf_mask    = |s| == +Inf, per lane (vabsq_f64 + vceqq_f64)
         *   t           = p - c
         *   new_s_comp  = s + t
         *   new_c_comp  = (new_s_comp - s) - t
         *   new_s_plain = s + p        (used instead, for lanes that
         *                                have already overflowed)
         *   s = vbslq_f64(inf_mask, new_s_plain, new_s_comp)
         *   c = vbslq_f64(inf_mask, c (unchanged), new_c_comp)
         * Mirrors blas_dot_kahan_avx2's per-lane blend — see its
         * comment for the full rationale. */
        float64x2_t p0 = vmulq_f64(x0, y0);
        uint64x2_t  m0 = vceqq_f64(vabsq_f64(s0), inf_vec);
        float64x2_t t0     = vsubq_f64(p0, c0);
        float64x2_t comp_s0 = vaddq_f64(s0, t0);
        float64x2_t comp_c0 = vsubq_f64(vsubq_f64(comp_s0, s0), t0);
        c0 = vbslq_f64(m0, c0, comp_c0);
        s0 = vbslq_f64(m0, vaddq_f64(s0, p0), comp_s0);

        float64x2_t p1 = vmulq_f64(x1, y1);
        uint64x2_t  m1 = vceqq_f64(vabsq_f64(s1), inf_vec);
        float64x2_t t1     = vsubq_f64(p1, c1);
        float64x2_t comp_s1 = vaddq_f64(s1, t1);
        float64x2_t comp_c1 = vsubq_f64(vsubq_f64(comp_s1, s1), t1);
        c1 = vbslq_f64(m1, c1, comp_c1);
        s1 = vbslq_f64(m1, vaddq_f64(s1, p1), comp_s1);

        float64x2_t p2 = vmulq_f64(x2, y2);
        uint64x2_t  m2 = vceqq_f64(vabsq_f64(s2), inf_vec);
        float64x2_t t2     = vsubq_f64(p2, c2);
        float64x2_t comp_s2 = vaddq_f64(s2, t2);
        float64x2_t comp_c2 = vsubq_f64(vsubq_f64(comp_s2, s2), t2);
        c2 = vbslq_f64(m2, c2, comp_c2);
        s2 = vbslq_f64(m2, vaddq_f64(s2, p2), comp_s2);

        float64x2_t p3 = vmulq_f64(x3, y3);
        uint64x2_t  m3 = vceqq_f64(vabsq_f64(s3), inf_vec);
        float64x2_t t3     = vsubq_f64(p3, c3);
        float64x2_t comp_s3 = vaddq_f64(s3, t3);
        float64x2_t comp_c3 = vsubq_f64(vsubq_f64(comp_s3, s3), t3);
        c3 = vbslq_f64(m3, c3, comp_c3);
        s3 = vbslq_f64(m3, vaddq_f64(s3, p3), comp_s3);
    }

    /* Reduce vector sums */
    float64x2_t s01 = vaddq_f64(s0, s1);
    float64x2_t s23 = vaddq_f64(s2, s3);
    double      sum = hadd2(vaddq_f64(s01, s23));

    /* Scalar Kahan tail — same overflow-to-NaN guard as the vectorized
     * loop above: `sum` may already be +-Inf coming out of the reduction. */
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
