/*
 * src/axpy.c — Dispatcher for blas_axpy
 *
 * Phase 5 restructure. Same pattern as src/dot.c.
 *
 * Kernel selection (set by cmake/DetectArch.cmake):
 *   USE_AVX2    → kernel/x86/axpy_avx2.c
 *   USE_NEON    → kernel/arm/axpy_neon.c
 *   USE_GENERIC → kernel/generic/axpy_generic.c  (default / fallback)
 *
 * The alpha==0 and n<=0 fast paths are handled HERE in the dispatcher,
 * not in the kernels. This avoids duplicating those guards three times
 * across three kernel files.
 */

#include "blas1/axpy.h"
#include "blas1/types.h"

#if defined(USE_AVX2)
#  include "../kernel/x86/axpy_avx2.h"
#elif defined(USE_NEON)
#  include "../kernel/arm/axpy_neon.h"
#else
#  include "../kernel/generic/axpy_generic.h"
#endif

void blas_axpy(blas_int n,
               BLAS_REAL alpha,
               const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                     BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return;
    }

    /* Alpha == 0: y is unchanged. Guard here once rather than in every
     * kernel — avoids NaN/Inf propagation from x into y. */
    if (BLAS_UNLIKELY(alpha == (BLAS_REAL)0.0)) {
        return;
    }

#if defined(USE_AVX2)
    blas_axpy_avx2(n, alpha, x, incx, y, incy);
#elif defined(USE_NEON)
    blas_axpy_neon(n, alpha, x, incx, y, incy);
#else
    blas_axpy_generic(n, alpha, x, incx, y, incy);
#endif
}
