/*
 * src/scal.c — BLAS Level 1 scal  (scale a vector)
 *
 * Computes:
 *
 *     x[i] <- alpha * x[i]   for i = 0 .. n-1
 *
 * The vector x is modified in-place. There is no second vector.
 *
 * Alpha == 0 special case:
 *   When alpha is exactly zero, the mathematically correct result is that
 *   every element of x becomes zero. However, IEEE 754 floating-point gives:
 *
 *       0.0 * NaN = NaN
 *       0.0 * Inf = NaN
 *
 *   So naively multiplying would propagate NaN into x even when alpha = 0,
 *   which is never what the caller wants. Reference BLAS specifies that
 *   alpha = 0 must zero the vector unconditionally. We handle this with a
 *   dedicated zeroing loop that never touches the existing values via multiply.
 *
 * Alpha == 1 fast path:
 *   Multiplying every element by 1.0 is a no-op mathematically. We return
 *   immediately to avoid a pointless pass over memory. Memory bandwidth is
 *   the bottleneck in BLAS 1 — unnecessary memory traffic is expensive.
 *
 * Vectorisation:
 *   The scaling loop  x[i] *= alpha  is a simple element-wise multiply with
 *   no cross-iteration dependencies. With BLAS_RESTRICT and unit stride,
 *   the compiler will auto-vectorise this to packed multiply instructions
 *   (vmulpd on AVX2, fmul on NEON).
 *
 * Stride support:
 *   incx must be > 0, matching reference BLAS's DSCAL (which returns
 *   immediately, unmodified, for incx <= 0 -- DSCAL does not support
 *   negative increments, unlike DAXPY/DDOT). Unit-stride fast path for
 *   the common case; general strided path handles incx > 1.
 */

#include "blas1/scal.h"
#include "blas1/types.h"

void blas_scal(blas_int n,
               BLAS_REAL alpha,
               BLAS_REAL * BLAS_RESTRICT x, blas_int incx)
{
    /* Guard: nothing to do, or invalid stride.
     *
     * Reference BLAS's DSCAL: "modified 3/93 to return if incx .le. 0."
     * -- DSCAL does NOT support negative increments (unlike DAXPY/DDOT,
     * which do). See src/asum.c for the full reasoning (same
     * convention, same out-of-bounds hazard this closes: without this
     * guard, incx < 0 would walk backward from offset 0 on the very
     * first step of the loops below, writing to memory before the
     * start of the array).
     */
    if (BLAS_UNLIKELY(n <= 0 || incx <= 0)) {
        return;
    }

    /*
     * Alpha == 0 special case.
     *
     * Zero every element explicitly — do not multiply. This avoids
     * NaN/Inf propagation and matches reference BLAS behaviour.
     * A plain assignment loop is also faster than a multiply loop
     * because modern CPUs can store-zero faster than load-multiply-store.
     */
    if (BLAS_UNLIKELY(alpha == (BLAS_REAL)0.0)) {
        if (incx == 1) {
            for (blas_int i = 0; i < n; i++) {
                x[i] = (BLAS_REAL)0.0;
            }
        } else {
            blas_int ix = 0;
            for (blas_int i = 0; i < n; i++) {
                x[ix] = (BLAS_REAL)0.0;
                ix += incx;
            }
        }
        return;
    }

    /*
     * Alpha == 1 fast path.
     *
     * x *= 1.0 is a no-op. Skip the memory pass entirely.
     * This saves bandwidth when scal is called redundantly inside
     * a solver loop (more common than it sounds in practice).
     */
    if (BLAS_UNLIKELY(alpha == (BLAS_REAL)1.0)) {
        return;
    }

    /*
     * General case — unit-stride fast path.
     *
     * x[i] *= alpha has no dependency between iterations.
     * BLAS_RESTRICT confirms x does not alias anything else.
     * The compiler vectorises this into packed multiply instructions.
     */
    if (incx == 1) {
        for (blas_int i = 0; i < n; i++) {
            x[i] *= alpha;
        }
        return;
    }

    /* General strided path */
    blas_int ix = 0;
    for (blas_int i = 0; i < n; i++) {
        x[ix] *= alpha;
        ix += incx;
    }
}
