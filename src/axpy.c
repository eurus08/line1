/*
 * src/axpy.c — BLAS Level 1 axpy  (alpha * x plus y)
 *
 * Computes:
 *
 *     y[i] <- alpha * x[i] + y[i]   for i = 0 .. n-1
 *
 * This operation is the backbone of iterative linear solvers. In conjugate
 * gradient alone it is called twice per iteration. Getting it fast matters.
 *
 * Vectorisation strategy:
 *   The loop body is a single fused multiply-add per element:
 *
 *       y[i] = alpha * x[i] + y[i]
 *
 *   With unit strides and BLAS_RESTRICT on both pointers, GCC/Clang will
 *   emit AVX2 vfmadd instructions automatically — no intrinsics needed here.
 *   Phase 5 will add an explicit AVX2 kernel for environments where the
 *   compiler misses this, but for Phase 2 auto-vectorisation is sufficient.
 *
 * Alpha == 0 fast path:
 *   Multiplying by zero should zero the scaled term, but floating-point
 *   rules mean 0.0 * NaN = NaN, and 0.0 * Inf = NaN. If alpha is exactly
 *   zero, we skip the multiply entirely and leave y unchanged — this matches
 *   reference BLAS behaviour and avoids NaN propagation into y.
 *
 * Stride support:
 *   Same convention as dot.c. Elements accessed as x[i*incx], y[i*incy].
 *   Unit-stride fast path keeps the loop clean for the vectoriser.
 *   The strided path is correct but will run scalar.
 *
 * Conventions:
 *   - x is const  — we only read from it.
 *   - y is mutable — we update it in-place.
 *   - n <= 0 returns immediately (no-op).
 */

#include "blas1/axpy.h"
#include "blas1/types.h"

void blas_axpy(blas_int n,
               BLAS_REAL alpha,
               const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                     BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    /* Guard: nothing to do */
    if (BLAS_UNLIKELY(n <= 0)) {
        return;
    }

    /*
     * Alpha == 0 fast path.
     *
     * The operation reduces to y <- y (identity), so we return immediately.
     * This avoids any risk of NaN/Inf contamination from x propagating into
     * y via multiplication. Reference BLAS specifies this behaviour.
     */
    if (BLAS_UNLIKELY(alpha == (BLAS_REAL)0.0)) {
        return;
    }

    /*
     * Unit-stride fast path.
     *
     * The loop is a textbook fused multiply-add reduction:
     *   y[i] += alpha * x[i]
     *
     * BLAS_RESTRICT on x and y tells the compiler they do not overlap in
     * memory. Without this promise, the compiler must assume that writing
     * to y[i] could affect a future read of x[j], which blocks vectorisation.
     * With it, all n iterations are independent and the loop vectorises.
     */
    if (incx == 1 && incy == 1) {
        for (blas_int i = 0; i < n; i++) {
            y[i] += alpha * x[i];
        }
        return;
    }

    /*
     * General strided path.
     *
     * Handles non-unit and negative strides. Negative strides are valid in
     * BLAS — the caller passes a pointer to the first element to be accessed
     * and a negative increment to walk backwards through memory.
     */
    blas_int ix = 0;
    blas_int iy = 0;

    for (blas_int i = 0; i < n; i++) {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
    }
}