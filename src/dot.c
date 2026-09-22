/*
 * src/dot.c — Dispatcher for blas_dot and blas_dot_kahan
 *
 * Phase 5 restructure. This file selects and calls the right
 * architecture-specific kernel at compile time based on which USE_*
 * macro cmake/DetectArch.cmake defined via target_apply_arch_flags().
 *
 *   USE_AVX2    → kernel/x86/dot_avx2.c
 *   USE_NEON    → kernel/arm/dot_neon.c
 *   USE_GENERIC → kernel/generic/dot_generic.c  (default / fallback)
 *
 * The public API (blas_dot, blas_dot_kahan) is unchanged — callers
 * see no difference. Only the computation path changes at link time.
 */

#include "line1/dot.h"
#include "line1/types.h"

/* Include the private header for the selected kernel. Each header
 * declares only the kernel-internal function names (_avx2, _neon,
 * _generic). Keeping these in separate headers means each kernel file
 * is self-contained and satisfies -Wmissing-prototypes on its own. */
#if defined(USE_AVX2)
#  include "../kernel/x86/dot_avx2.h"
#elif defined(USE_NEON)
#  include "../kernel/arm/dot_neon.h"
#else
#  include "../kernel/generic/dot_generic.h"
#endif

/* ------------------------------------------------------------------
 * Public API — blas_dot
 * ------------------------------------------------------------------ */
BLAS_REAL blas_dot(blas_int n,
                   const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                   const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
#if defined(USE_AVX2)
    return blas_dot_avx2(n, x, incx, y, incy);
#elif defined(USE_NEON)
    return blas_dot_neon(n, x, incx, y, incy);
#else
    return blas_dot_generic(n, x, incx, y, incy);
#endif
}

/* ------------------------------------------------------------------
 * Public API — blas_dot_kahan
 * ------------------------------------------------------------------ */
BLAS_REAL blas_dot_kahan(blas_int n,
                          const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                          const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
#if defined(USE_AVX2)
    return blas_dot_kahan_avx2(n, x, incx, y, incy);
#elif defined(USE_NEON)
    return blas_dot_kahan_neon(n, x, incx, y, incy);
#else
    return blas_dot_kahan_generic(n, x, incx, y, incy);
#endif
}
