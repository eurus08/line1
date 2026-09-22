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
 * BLAS_FABS() correctness:
 *   Always use BLAS_FABS() (from types.h), never a manual x > 0 ? x : -x.
 *   Reasons:
 *     - it is correct for -0.0  (returns +0.0; negation also does,
 *       but the intent is clearer with an abs call)
 *     - it is correct for NaN   (returns NaN, propagates the signal)
 *     - it is correct for -Inf  (returns +Inf cleanly)
 *     - on every modern compiler, it compiles to a single instruction
 *       (ANDPS / FABS) — there is no runtime branch, so it is also faster
 *       than a conditional expression.
 *   BLAS_FABS() expands to fabs() for the default double build and to
 *   fabsf() when built with -DBLAS_USE_FLOAT, so it always matches
 *   BLAS_REAL's actual width -- no manual fabs/fabsf choice needed at
 *   the call site, and no accidental double-promotion of a float value.
 *
 * Kahan summation:
 *   asum accumulates n terms. Without compensation the rounding error
 *   grows as O(n * eps). Kahan keeps it at O(eps). The terms here are
 *   all non-negative (absolute values), which makes cancellation in the
 *   Kahan step less critical than in dot — but for large n and mixed
 *   magnitudes it still matters.
 *
 * Overflow handling:
 *   Kahan's compensation step can turn an ordinary overflow into a
 *   spurious NaN. Once `sum` has overflowed to +Inf, the compensation
 *   term `c` typically becomes +Inf too (c = (new_sum - sum) - t, and
 *   new_sum - sum is Inf - finite = Inf once new_sum has overflowed).
 *   On the NEXT term, `t = ax - c` becomes `finite - Inf = -Inf`, and
 *   `sum + t` becomes `Inf + (-Inf) = NaN` -- even though every term
 *   summed so far was a plain non-negative, finite magnitude and the
 *   mathematically correct answer is unambiguously +Inf (confirmed:
 *   asum({1e308, 1e308, 1e308}) returned NaN here before this fix,
 *   while plain naive summation of the same three values correctly
 *   overflows straight to +Inf with no NaN involved at all).
 *
 *   Fix: once `sum` is already +Inf (checked via blas_is_inf(), NOT
 *   isinf() -- see types.h for why), stop trusting the compensation
 *   arithmetic and just add the next term directly (`sum = sum + ax`).
 *   Since every term here is non-negative, this can only ever keep
 *   `sum` at +Inf or leave it at +Inf -- it exactly reproduces what
 *   naive (uncompensated) summation would do from that point forward.
 *   The guarantee this preserves is "Kahan never does worse than
 *   naive summation," which is what actually failed here.
 *
 * Stride support:
 *   Same convention as the rest of the library. Unit-stride fast path
 *   for auto-vectorisation; general strided path for everything else.
 */
#include "blas1/asum.h"
#include "blas1/types.h"
/* fabs() replaced by BLAS_FABS() from types.h (precision-generic) */

BLAS_REAL blas_asum(blas_int n,
                    const BLAS_REAL * BLAS_RESTRICT x, blas_int incx)
{
    /* Guard: empty vector, or invalid stride.
     *
     * Reference BLAS's DASUM: "modified 3/93 to return if incx .le. 0."
     * -- DASUM does NOT support negative increments (unlike DAXPY/DDOT,
     * which do -- see blas_stride_start() in types.h for those). This
     * guard also closes a real out-of-bounds read: without it, a
     * caller passing incx < 0 and a pointer to the true start of the
     * vector (the natural, standard-compliant way to call a stride
     * argument) would have this walk backward from offset 0 on the
     * very first step, reading memory before the start of the array.
     */
    if (BLAS_UNLIKELY(n <= 0 || incx <= 0)) {
        return (BLAS_REAL)0.0;
    }

    BLAS_REAL sum = (BLAS_REAL)0.0;
    BLAS_REAL c   = (BLAS_REAL)0.0;   /* Kahan compensation */

    /* ------------------------------------------------------------------
     * Unit-stride fast path
     *
     * BLAS_FABS(x[i]) has no cross-iteration dependency. With BLAS_RESTRICT
     * and unit stride the compiler vectorises this into packed abs +
     * add instructions (VADDPD + VANDPD on AVX2).
     * ------------------------------------------------------------------ */
    if (incx == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL ax = BLAS_FABS(x[i]);
            if (BLAS_UNLIKELY(blas_is_inf(sum))) {
                /* sum has already overflowed to +Inf -- see the file
                 * header's "Overflow handling" note. Degrade to plain
                 * addition (matches naive summation) instead of
                 * trusting a compensation term that may itself have
                 * become +Inf. */
                sum = sum + ax;
                continue;
            }
            BLAS_REAL t       = ax - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
        }
        return sum;
    }

    /* General strided path */
    blas_int ix = 0;
    for (blas_int i = 0; i < n; i++) {
        BLAS_REAL ax = BLAS_FABS(x[ix]);
        if (BLAS_UNLIKELY(blas_is_inf(sum))) {
            sum = sum + ax;
            ix += incx;
            continue;
        }
        BLAS_REAL t       = ax - c;
        BLAS_REAL new_sum = sum + t;
        c   = (new_sum - sum) - t;
        sum = new_sum;
        ix += incx;
    }

    return sum;
}
