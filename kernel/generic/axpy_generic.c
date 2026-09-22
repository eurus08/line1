/*
 * kernel/generic/axpy_generic.c — Scalar fallback kernel for blas_axpy
 *
 * Step 5.5 of the build plan (generic part).
 *
 * Called by src/axpy.c's dispatcher when USE_GENERIC=1.
 * The caller (dispatcher) has already handled:
 *   - n <= 0  guard (returns immediately)
 *   - alpha == 0 fast path (returns immediately — no NaN propagation)
 * So this kernel is only ever called with n > 0 and alpha != 0.
 *
 * The algorithm is identical to the original src/axpy.c inner loop.
 * Auto-vectorisation note: same as dot_generic.c — under -O3 -march=native
 * GCC will auto-vectorise the unit-stride loop. The AVX2 explicit kernel
 * (axpy_avx2.c) broadcasts alpha once and uses FMA intrinsics to guarantee
 * the instruction sequence on hardware that supports it.
 */

#include "line1/types.h"
#include "axpy_generic.h"   /* own prototype — satisfies -Wmissing-prototypes */

void blas_axpy_generic(blas_int n,
                        BLAS_REAL alpha,
                        const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                              BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    /* Unit-stride fast path — auto-vectorises under -O3 -march=native.
     * y[i] += alpha * x[i] has no cross-iteration dependencies, so
     * with BLAS_RESTRICT the compiler can emit packed FMA instructions. */
    if (incx == 1 && incy == 1) {
        for (blas_int i = 0; i < n; i++) {
            y[i] += alpha * x[i];
        }
        return;
    }

    /* General strided path — supports negative incx/incy, matching
     * reference BLAS's DAXPY (see blas_stride_start() in types.h). */
    blas_int ix = blas_stride_start(n, incx);
    blas_int iy = blas_stride_start(n, incy);
    for (blas_int i = 0; i < n; i++) {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
    }
}
