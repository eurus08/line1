/*
 * src/asum.c — BLAS Level 1 asum  (sum of absolute values)
 *
 * Computes the L1 norm of a vector x:
 *
 *     result = sum( |x[i]| )   for i = 0 .. n-1
 *
 * This is the simplest accumulation in the library — no scaling, no
 * two-pass algorithm needed. The absolute value prevents positive and
 * negative terms from cancelling, so overflow is the only numerical
 * hazard (and only for pathologically large vectors).
 *
 * fabs() correctness:
 *   Always use fabs() from <math.h>, never a manual  x > 0 ? x : -x.
 *   Reasons:
 *     - fabs() is correct for -0.0  (returns +0.0; negation also does,
 *       but the intent is clearer with fabs)
 *     - fabs() is correct for NaN   (returns NaN, propagates the signal)
 *     - fabs() is correct for -Inf  (returns +Inf cleanly)
 *     - On every modern compiler, fabs() compiles to a single instruction
 *       (ANDPS / FABS) — there is no runtime branch, so it is also faster
 *       than a conditional expression.
 *   In C11 with <math.h>, fabs() works for double and fabsf() for float.
 *   We use fabs() throughout; BLAS_REAL is double by default, and when
 *   built with -DBLAS_USE_FLOAT the compiler will warn if fabs/fabsf
 *   mismatch — at that point fabsf() or the type-generic fabsl() variant
 *   should be used. For now fabs() is correct for the default precision.
 *
 * Kahan summation:
 *   asum accumulates n terms. Without compensation the rounding error
 *   grows as O(n * eps). Kahan keeps it at O(eps). The terms here are
 *   all non-negative (absolute values), which makes cancellation in the
 *   Kahan step less critical than in dot — but for large n and mixed
 *   magnitudes it still matters.
 *
 * Stride support:
 *   Same convention as the rest of the library. Unit-stride fast path
 *   for auto-vectorisation; general strided path for everything else.
 */

#include "blas1/asum.h"
#include "blas1/types.h"
#include <math.h>   /* fabs() */

BLAS_REAL blas_asum(blas_int n,
                    const BLAS_REAL * BLAS_RESTRICT x, blas_int incx)
{
    /* Guard: empty vector */
    if (BLAS_UNLIKELY(n <= 0)) {
        return (BLAS_REAL)0.0;
    }

    BLAS_REAL sum = (BLAS_REAL)0.0;
    BLAS_REAL c   = (BLAS_REAL)0.0;   /* Kahan compensation */

    /* ------------------------------------------------------------------
     * Unit-stride fast path
     *
     * fabs(x[i]) has no cross-iteration dependency. With BLAS_RESTRICT
     * and unit stride the compiler vectorises this into packed abs +
     * add instructions (VADDPD + VANDPD on AVX2).
     * ------------------------------------------------------------------ */
    if (incx == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL t       = fabs(x[i]) - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
        }
        return sum;
    }

    /* General strided path */
    blas_int ix = 0;
    for (blas_int i = 0; i < n; i++) {
        BLAS_REAL t       = fabs(x[ix]) - c;
        BLAS_REAL new_sum = sum + t;
        c   = (new_sum - sum) - t;
        sum = new_sum;
        ix += incx;
    }

    return sum;
}