/*
 * src/nrm2.c — BLAS Level 1 nrm2  (Euclidean norm)
 *
 * Computes the Euclidean norm (L2 norm) of a vector x:
 *
 *     result = sqrt( sum( x[i]^2 ) )   for i = 0 .. n-1
 *
 * =========================================================================
 * WHY THE NAIVE APPROACH FAILS
 * =========================================================================
 *
 * The obvious implementation:
 *
 *     double sum = 0.0;
 *     for (int i = 0; i < n; i++) sum += x[i] * x[i];
 *     return sqrt(sum);
 *
 * fails in two ways:
 *
 *   OVERFLOW:   If x[i] = 1e200, then x[i]*x[i] = 1e400 = Inf.
 *               The norm of a perfectly finite vector becomes infinity.
 *
 *   UNDERFLOW:  If x[i] = 1e-200, then x[i]*x[i] = 1e-400 = 0.0.
 *               Small but non-zero values vanish entirely from the sum.
 *
 * Both are silent — no error is raised, you just get a wrong answer.
 *
 * =========================================================================
 * THE STABLE ALGORITHM (Blue 1978 / LAPACK dnrm2 style)
 * =========================================================================
 *
 * The key insight: if we know the largest absolute value in the vector
 * (call it 'scale'), we can factor it out:
 *
 *     ||x|| = scale * sqrt( sum( (x[i] / scale)^2 ) )
 *
 * Now every term (x[i] / scale) is in the range [-1, 1], so squaring it
 * can never overflow. And since scale is the *largest* element, at least
 * one term equals exactly 1.0 — so the sum is always >= 1.0, which means
 * underflow is also impossible.
 *
 * Two-pass structure:
 *   Pass 1: find scale = max( |x[i]| )
 *   Pass 2: accumulate sum of ( x[i] / scale )^2, with Kahan compensation
 *   Result: scale * sqrt(sum)
 *
 * Edge cases handled:
 *   - n <= 0         : return 0.0
 *   - all zeros      : scale == 0.0, return 0.0 (guarded before divide)
 *   - single element : returns |x[0]| exactly
 *
 * Kahan summation in pass 2:
 *   The scaled terms are all <= 1.0 in magnitude, so overflow is gone.
 *   But we still accumulate n terms, and rounding error grows with n.
 *   Kahan summation keeps the error at O(eps) regardless of n.
 *   Same -ffast-math caveat as dot.c applies here — build with
 *   -DBLAS1_STRICT_IEEE=ON if you need the compensation to survive.
 *
 * Stride support:
 *   Both passes respect incx. Unit-stride fast path for each pass.
 */

#include "blas1/nrm2.h"
#include "blas1/types.h"
#include <math.h>   /* fabs(), sqrt() */

BLAS_REAL blas_nrm2(blas_int n,
                    const BLAS_REAL * BLAS_RESTRICT x, blas_int incx)
{
    /* Guard: empty vector */
    if (BLAS_UNLIKELY(n <= 0)) {
        return (BLAS_REAL)0.0;
    }

    /* Single element: norm is just the absolute value */
    if (n == 1) {
        return fabs(x[0]);
    }

    /* ------------------------------------------------------------------
     * Pass 1 — find the scaling factor
     *
     * scale = max( |x[i]| ) over all i.
     * We use this to normalise every element before squaring.
     * ------------------------------------------------------------------ */
    BLAS_REAL scale = (BLAS_REAL)0.0;

    if (incx == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL ax = fabs(x[i]);
            if (ax > scale) scale = ax;
        }
    } else {
        blas_int ix = 0;
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL ax = fabs(x[ix]);
            if (ax > scale) scale = ax;
            ix += incx;
        }
    }

    /* Guard: all-zero vector */
    if (BLAS_UNLIKELY(scale == (BLAS_REAL)0.0)) {
        return (BLAS_REAL)0.0;
    }

    /* ------------------------------------------------------------------
     * Pass 2 — accumulate scaled squares with Kahan compensation
     *
     * Each term is (x[i] / scale)^2, which is in [0, 1].
     * Squaring values in [0,1] cannot overflow or underflow.
     * Kahan compensation keeps rounding error bounded at O(eps).
     * ------------------------------------------------------------------ */
    BLAS_REAL sum = (BLAS_REAL)0.0;
    BLAS_REAL c   = (BLAS_REAL)0.0;   /* Kahan compensation */

    if (incx == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL t       = (x[i] / scale) * (x[i] / scale) - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
        }
    } else {
        blas_int ix = 0;
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL t       = (x[ix] / scale) * (x[ix] / scale) - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
            ix += incx;
        }
    }

    /* Undo the scaling: ||x|| = scale * sqrt( sum of scaled squares ) */
    return scale * sqrt(sum);
}