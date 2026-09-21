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
 *   Pass 1: find scale = max( |x[i]| ) — delegated to blas_iamax()
 *   Pass 2: accumulate sum of ( x[i] / scale )^2, with Kahan compensation
 *   Result: scale * sqrt(sum)
 *
 * Reuse of blas_iamax:
 *   Pass 1 is exactly the operation iamax performs — find the element with
 *   the largest absolute value. Rather than duplicating that loop here,
 *   we call blas_iamax() directly. This keeps the two implementations in
 *   sync: any future improvement to iamax (e.g. an AVX2 kernel in Phase 5)
 *   is automatically inherited by nrm2.
 *
 *   blas_iamax returns a 1-based index k. To recover the scale value:
 *       scale = fabs( x[(k - 1) * incx] )
 *   The (k-1) converts back to 0-based; multiplying by incx gives the
 *   correct memory offset for strided arrays.
 *
 * Edge cases handled:
 *   - n <= 0         : return 0.0
 *   - all zeros      : scale == 0.0, return 0.0 (guarded before divide)
 *   - single element : blas_iamax returns 1, scale = |x[0]|, sum = 1.0,
 *                      result = scale * sqrt(1.0) = |x[0]| — exact.
 *
 * Kahan summation in pass 2:
 *   The scaled terms are all in [0, 1], so overflow is gone. But we still
 *   accumulate n terms and rounding error grows with n. Kahan keeps it at
 *   O(eps) regardless of n. Same -ffast-math caveat as dot.c applies —
 *   build with -DBLAS1_STRICT_IEEE=ON if the compensation must survive.
 *
 * Stride support:
 *   blas_iamax handles pass 1 with full stride support. Pass 2 mirrors
 *   the same unit-stride / strided split used throughout the library.
 */

#include "blas1/nrm2.h"
#include "blas1/iamax.h"
#include "blas1/types.h"
/* BLAS_FABS() / BLAS_SQRT() from types.h are precision-generic (fabs/sqrt
 * for double, fabsf/sqrtf for float) -- see BLAS1_USE_FLOAT fix notes. */

BLAS_REAL blas_nrm2(blas_int n,
                    const BLAS_REAL * BLAS_RESTRICT x, blas_int incx)
{
    /* Guard: empty vector */
    if (BLAS_UNLIKELY(n <= 0)) {
        return (BLAS_REAL)0.0;
    }

    /* ------------------------------------------------------------------
     * Pass 1 — find the scaling factor via blas_iamax
     *
     * blas_iamax returns the 1-based index of max(|x[i]|).
     * We convert it back to a memory offset to read the actual value.
     * ------------------------------------------------------------------ */
    blas_int  k     = blas_iamax(n, x, incx);
    BLAS_REAL scale = BLAS_FABS(x[(k - 1) * incx]);

    /* Guard: all-zero vector (scale == 0 means every element is zero) */
    if (BLAS_UNLIKELY(scale == (BLAS_REAL)0.0)) {
        return (BLAS_REAL)0.0;
    }

    /* ------------------------------------------------------------------
     * Pass 2 — accumulate scaled squares with Kahan compensation
     *
     * Each term is (x[i] / scale)^2, which lies in [0, 1].
     * Squaring values in [0, 1] cannot overflow or underflow.
     * Kahan compensation keeps rounding error bounded at O(eps).
     * ------------------------------------------------------------------ */
    BLAS_REAL sum = (BLAS_REAL)0.0;
    BLAS_REAL c   = (BLAS_REAL)0.0;   /* Kahan compensation */

    if (incx == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL si      = x[i] / scale;
            BLAS_REAL t       = si * si - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
        }
    } else {
        blas_int ix = 0;
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL si      = x[ix] / scale;
            BLAS_REAL t       = si * si - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
            ix += incx;
        }
    }

    /* Undo the scaling: ||x|| = scale * sqrt( sum of scaled squares ) */
    return scale * BLAS_SQRT(sum);
}