/*
 * kernel/arm/axpy_neon.c — NEON intrinsics kernel for blas_axpy
 *
 * Step 5.5 of the build plan (NEON part).
 *
 * Compiled ONLY when USE_NEON=1. NEON is mandatory on AArch64 so no
 * extra compiler flag is needed beyond -march=armv8-a.
 *
 * Same structure as axpy_avx2.c scaled down to 128-bit registers:
 *   - float64x2_t holds 2 doubles (vs AVX2's 4)
 *   - vdupq_n_f64(alpha) broadcasts alpha into both lanes
 *   - vfmaq_f64(y, valpha, x) computes alpha*x + y for both lanes
 *   - 4 independent streams × 2 lanes = 8 doubles per iteration
 *   - vst1q_f64 stores results back to y
 *
 * The caller guarantees alpha != 0 and n > 0 before reaching this kernel.
 */

#include <arm_neon.h>

#include "blas1/types.h"
#include "axpy_neon.h"   /* own prototype — satisfies -Wmissing-prototypes */

#ifndef BLAS_USE_FLOAT
/* double build — proceed */
#else
#  error "axpy_neon.c uses float64x2_t (double) NEON intrinsics and cannot be \
compiled for float precision (-DBLAS_USE_FLOAT)."
#endif

void blas_axpy_neon(blas_int n,
                     BLAS_REAL alpha,
                     const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                           BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    /* Strided fallback */
    if (incx != 1 || incy != 1) {
        blas_int ix = 0, iy = 0;
        for (blas_int i = 0; i < n; i++) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
        return;
    }

    /* ----------------------------------------------------------------
     * Unit-stride NEON path — 4 streams × 2 lanes = 8 doubles/iter
     * ---------------------------------------------------------------- */
    float64x2_t valpha = vdupq_n_f64(alpha);   /* [α, α] */

    blas_int i  = 0;
    blas_int n8 = (n / 8) * 8;

    for (; i < n8; i += 8) {
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t x3 = vld1q_f64(x + i + 6);

        float64x2_t y0 = vld1q_f64(y + i);
        float64x2_t y1 = vld1q_f64(y + i + 2);
        float64x2_t y2 = vld1q_f64(y + i + 4);
        float64x2_t y3 = vld1q_f64(y + i + 6);

        /* vfmaq_f64(acc, a, b) = acc + a*b — note argument order:
         * accumulator first, then the two factors. */
        y0 = vfmaq_f64(y0, valpha, x0);
        y1 = vfmaq_f64(y1, valpha, x1);
        y2 = vfmaq_f64(y2, valpha, x2);
        y3 = vfmaq_f64(y3, valpha, x3);

        vst1q_f64(y + i,     y0);
        vst1q_f64(y + i + 2, y1);
        vst1q_f64(y + i + 4, y2);
        vst1q_f64(y + i + 6, y3);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        y[i] += alpha * x[i];
    }
}
