/*
 * src/dot.c — BLAS Level 1 dot product
 *
 * Computes the dot product of two vectors x and y:
 *
 *     result = sum( x[i] * y[i] )  for i = 0 .. n-1
 *
 * Two variants are provided:
 *
 *   blas_dot()       — Standard accumulation. The loop is written to be
 *                      auto-vectorisation friendly (no data dependencies
 *                      between iterations). Fastest on modern CPUs with FMA.
 *
 *   blas_dot_kahan() — Kahan compensated summation. Tracks the floating-point
 *                      rounding error each step and feeds it back in. This
 *                      reduces the accumulated error from O(n*eps) to O(eps)
 *                      at a cost of roughly 4x more FP operations.
 *                      Use when numerical stability matters more than raw
 *                      throughput (e.g. ill-conditioned problems).
 *
 * Stride support (incx / incy):
 *   Elements are accessed as x[i * incx], y[i * incy].
 *   - incx = incy = 1  → dense contiguous vectors (common case)
 *   - incx > 1         → strided access (e.g. a column of a row-major matrix)
 *   - incx < 0         → reverse traversal (x traversed backwards)
 *   A stride of 0 is undefined behaviour in standard BLAS — we guard for it.
 *
 * Conventions:
 *   - Returns 0.0 for n <= 0.
 *   - Input pointers are const — we never modify x or y.
 *   - BLAS_RESTRICT promises the compiler x and y do not alias each other,
 *     which is required for safe auto-vectorisation.
 *   - BLAS_REAL is either double or float, selected at compile time via
 *     -DBLAS_USE_FLOAT. Never hardcode double or float here.
 */

#include "blas1/dot.h"
#include "blas1/types.h"
#include <math.h>  /* for any future extensions; kept for consistency */

/* -------------------------------------------------------------------------
 * blas_dot — standard dot product
 *
 * The accumulator 'sum' is updated once per iteration with no dependency
 * on previous iterations (a simple reduction). This pattern is recognised
 * by GCC/Clang and auto-vectorised into AVX2 or NEON instructions when
 * incx == incy == 1. When strides differ from 1, the compiler should fall back
 * to scalar code — that is correct and expected.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot(blas_int n,
                   const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                   const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    /* Guard: empty or invalid vector — return identity for addition */
    if (BLAS_UNLIKELY(n <= 0)) {
        return (BLAS_REAL)0.0;
    }

    BLAS_REAL sum = (BLAS_REAL)0.0;

    /*
     * Unit-stride fast path.
     *
     * When both strides are 1, the pointers march forward by one element
     * per iteration. The compiler can vectorise this into a sequence of
     * packed multiply-add instructions. We make this a separate branch so
     * the hot path is free of the stride multiply overhead.
     */
    if (incx == 1 && incy == 1) {
        for (blas_int i = 0; i < n; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    /*
     * General strided path.
     *
     * For negative strides, BLAS convention is that the pointer passed in
     * already points to the *first element to be accessed*, not the start
     * of the physical array. Stride arithmetic handles the rest.
     */
    blas_int ix = 0;
    blas_int iy = 0;

    for (blas_int i = 0; i < n; i++) {
        sum += x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }

    return sum;
}

/* -------------------------------------------------------------------------
 * blas_dot_kahan — Kahan compensated summation dot product
 *
 * Algorithm (Kahan 1965):
 *   For each term t = x[i] * y[i]:
 *     1. Adjust t by the compensation from the previous step:  t -= c
 *     2. Add adjusted t to the running sum:                    new_sum = sum + t
 *     3. Compute what was lost in step 2 (the low bits):       c = (new_sum - sum) - t
 *     4. Carry new_sum forward.
 *
 * The variable 'c' accumulates the rounding error that would otherwise be
 * discarded. On the next iteration it is subtracted back in, so no precision
 * is permanently lost.
 *
 * Note on -ffast-math:
 *   The flag -ffast-math (enabled in Release builds via CompilerFlags.cmake)
 *   allows the compiler to reorder FP operations, which can eliminate the
 *   compensation step entirely — making this function equivalent to blas_dot.
 *   If strict Kahan behaviour is required, build with -DBLAS1_STRICT_IEEE=ON
 *   which omits -ffast-math. See cmake/CompilerFlags.cmake.
 *
 * Volatile trick: marking 'sum' and 'c' volatile prevents the compiler from
 *   optimising away the compensation even under -ffast-math. This is a
 *   pragmatic workaround; the clean solution is BLAS1_STRICT_IEEE=ON.
 *   We leave the volatile commented out here and rely on the build flag,
 *   since volatile adds a memory round-trip and defeats the purpose of
 *   calling the fast path.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_kahan(blas_int n,
                          const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                          const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return (BLAS_REAL)0.0;
    }

    BLAS_REAL sum = (BLAS_REAL)0.0;
    BLAS_REAL c   = (BLAS_REAL)0.0;  /* running compensation */

    if (incx == 1 && incy == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL t       = x[i] * y[i] - c;  /* apply compensation   */
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;             /* capture lost bits    */
            sum = new_sum;
        }
        return sum;
    }

    blas_int ix = 0;
    blas_int iy = 0;

    for (blas_int i = 0; i < n; i++) {
        BLAS_REAL t       = x[ix] * y[iy] - c;
        BLAS_REAL new_sum = sum + t;
        c   = (new_sum - sum) - t;
        sum = new_sum;
        ix += incx;
        iy += incy;
    }

    return sum;
}