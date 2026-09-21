/*
 * src/iamax.c — BLAS Level 1 iamax  (index of maximum absolute value)
 *
 * Returns the 1-based index of the first element of x with the largest
 * absolute value:
 *
 *     result = argmax( |x[i]| )   for i = 0 .. n-1
 *              returned as (i + 1)  — 1-based, Fortran convention
 *
 * =========================================================================
 * WHY 1-BASED?
 * =========================================================================
 *
 * BLAS was originally written in Fortran, which indexes arrays from 1.
 * The standard was frozen with that convention. Every BLAS implementation
 * in the world returns 1-based indices from iamax — if we returned 0-based,
 * we would be incompatible with every piece of code that calls us.
 *
 * Caller convention: subtract 1 before using the result as a C array index.
 *
 *     blas_int k = blas_iamax(n, x, 1);
 *     BLAS_REAL largest = x[k - 1];          // correct C indexing
 *
 * =========================================================================
 * TIE-BREAKING
 * =========================================================================
 *
 * When two elements share the same absolute value, we return the index of
 * the FIRST one encountered. This matches reference BLAS (NETLIB) behaviour.
 * The comparison is strictly greater-than (>), so ties are broken by
 * position — the first maximum wins.
 *
 * =========================================================================
 * EDGE CASES
 * =========================================================================
 *
 *   n <= 0  : return 0  (BLAS standard — signals invalid input)
 *   n == 1  : return 1  (only one element, trivially the maximum)
 *
 * Note: return value 0 is the sentinel for "no valid result". Since normal
 * results are >= 1, callers can check for 0 to detect invalid input.
 *
 * =========================================================================
 * NaN BEHAVIOUR
 * =========================================================================
 *
 * If x contains NaN, fabs(NaN) produces NaN, and NaN > anything is false
 * in IEEE 754. This means NaN values are silently skipped — the function
 * returns the index of the largest finite element. This matches the
 * behaviour of most reference implementations. If the entire vector is NaN,
 * the function returns 1 (the initial index, never updated).
 *
 * Stride support:
 *   Same convention as the rest of the library. incx = 1 is the fast path.
 *   The 1-based return value is an element index, not a memory offset —
 *   the caller uses (result - 1) * incx to get the memory offset.
 */

#include "blas1/iamax.h"
#include "blas1/types.h"
/* fabs() replaced by BLAS_FABS() from types.h (precision-generic) */

blas_int blas_iamax(blas_int n,
                    const BLAS_REAL * BLAS_RESTRICT x, blas_int incx)
{
    /* Guard: empty or invalid vector — return sentinel 0 */
    if (BLAS_UNLIKELY(n <= 0)) {
        return 0;
    }

    /* Single element — trivially the maximum */
    if (n == 1) {
        return 1;
    }

    blas_int    max_idx = 0;          /* 0-based index of current maximum  */
    BLAS_REAL   max_val = (BLAS_REAL)0.0;

    if (incx == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL ax = BLAS_FABS(x[i]);
            if (ax > max_val) {
                max_val = ax;
                max_idx = i;
            }
        }
    } else {
        blas_int ix = 0;
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL ax = BLAS_FABS(x[ix]);
            if (ax > max_val) {
                max_val = ax;
                max_idx = i;
            }
            ix += incx;
        }
    }

    /* Convert 0-based internal index to 1-based BLAS return value */
    return max_idx + 1;
}