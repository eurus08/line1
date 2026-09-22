/*
 * kernel/generic/dot_generic.c — Scalar fallback kernel for blas_dot
 *
 * Step 5.1 of the build plan.
 *
 * This file is selected when neither USE_AVX2 nor USE_NEON is defined
 * (i.e. the build system set USE_GENERIC=1). It is functionally identical
 * to the original src/dot.c inner loop — no SIMD intrinsics, no
 * architecture-specific code. It compiles and runs correctly on any
 * C11 target.
 *
 * Why have a separate file at all, if it's the same as before?
 *
 *   src/dot.c is now a DISPATCHER — it checks #if defined(USE_AVX2) etc.
 *   and calls the right kernel. The actual computation has moved here.
 *   This separation means:
 *     - The dispatcher in src/dot.c is thin and compiler-flag-agnostic.
 *     - Each kernel file can be compiled with its own specific flags
 *       (e.g. kernel/x86/dot_avx2.c gets -mavx2 -mfma applied only to
 *       that translation unit, without polluting the rest of the build).
 *     - Switching kernels at build time requires no changes to src/dot.c.
 *
 * Auto-vectorisation note:
 *   Even though this kernel uses no explicit SIMD intrinsics, GCC will
 *   still auto-vectorise the unit-stride loop under -O3 -march=native.
 *   The difference from the AVX2 explicit kernel (step 5.2) is:
 *     - This: the compiler decides how to vectorise (may or may not use FMA,
 *       may choose vector width, may or may not unroll).
 *     - AVX2: WE decide — 4 doubles per iteration, explicit FMA, controlled
 *       unrolling, known register usage.
 *   On machines where AVX2 is available, the explicit kernel is measurably
 *   faster because we leave nothing to compiler heuristics. On this machine
 *   (generic fallback active), auto-vectorisation is the best we get here.
 *
 * Function names:
 *   blas_dot_generic()       — standard accumulation, generic scalar
 *   blas_dot_kahan_generic() — Kahan compensated, generic scalar
 *
 *   These are NOT the public API names. src/dot.c's dispatcher calls them
 *   under the #else branch and re-exports the result under the public
 *   names blas_dot() and blas_dot_kahan(). External callers always use
 *   the public names — they never see the _generic suffix.
 */

#include "line1/types.h"
#include "dot_generic.h"   /* own prototype — satisfies -Wmissing-prototypes */

/* -------------------------------------------------------------------------
 * blas_dot_generic — standard dot product, scalar fallback
 *
 * Identical to the original src/dot.c inner loops. No changes to the
 * algorithm — this is the correctness baseline that AVX2/NEON must match.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_generic(blas_int n,
                            const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                            const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return (BLAS_REAL)0.0;
    }

    BLAS_REAL sum = (BLAS_REAL)0.0;

    /* Unit-stride fast path — auto-vectorises under -O3 -march=native */
    if (incx == 1 && incy == 1) {
        for (blas_int i = 0; i < n; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    /* General strided path — supports negative incx/incy, matching
     * reference BLAS's DDOT (see blas_stride_start() in types.h). */
    blas_int ix = blas_stride_start(n, incx);
    blas_int iy = blas_stride_start(n, incy);
    for (blas_int i = 0; i < n; i++) {
        sum += x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }
    return sum;
}

/* -------------------------------------------------------------------------
 * blas_dot_kahan_generic — Kahan compensated dot product, scalar fallback
 *
 * Same algorithm as the original src/dot.c blas_dot_kahan(). The -ffast-math
 * caveat (which can eliminate the compensation) applies here identically —
 * see src/dot.c for the full discussion. Strict IEEE 754 (no -ffast-math)
 * is this project's DEFAULT build for exactly this reason; opt out with
 * -DLINE1_STRICT_IEEE=OFF only once throughput has been judged worth it.
 *
 * Overflow handling (same fix as src/asum.c, same bug class):
 *   Once `sum` overflows to +-Inf, the compensation term `c` typically
 *   becomes infinite too, and the NEXT term's `t = product - c`
 *   becomes an opposite-signed infinity -- so `sum + t` computes
 *   Inf + (-Inf) (or the mirror image), which IEEE 754 defines as
 *   NaN, even though plain uncompensated summation would have stayed
 *   at a well-defined +-Inf. Once blas_is_inf(sum) is true, this
 *   degrades to plain addition (`sum = sum + product`), matching what
 *   naive summation would do from that point forward -- "Kahan never
 *   does worse than naive summation."
 *
 *   Note this does NOT special-case the case where `sum` and the
 *   incoming term are infinite with OPPOSITE signs (e.g. sum is +Inf
 *   and a later product is -Inf): that is a genuine mathematical
 *   indeterminate form (the true dot product would depend on exactly
 *   how each side grew to infinity), and naive summation would also
 *   produce NaN there -- so NaN in that specific case is correct, not
 *   a bug, and this fix intentionally does not suppress it.
 * ------------------------------------------------------------------------- */
BLAS_REAL blas_dot_kahan_generic(blas_int n,
                                  const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                                  const BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    if (BLAS_UNLIKELY(n <= 0)) {
        return (BLAS_REAL)0.0;
    }

    BLAS_REAL sum = (BLAS_REAL)0.0;
    BLAS_REAL c   = (BLAS_REAL)0.0;

    if (incx == 1 && incy == 1) {
        for (blas_int i = 0; i < n; i++) {
            BLAS_REAL p = x[i] * y[i];
            if (BLAS_UNLIKELY(blas_is_inf(sum))) {
                sum = sum + p;
                continue;
            }
            BLAS_REAL t       = p - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
        }
        return sum;
    }

    blas_int ix = blas_stride_start(n, incx);
    blas_int iy = blas_stride_start(n, incy);
    for (blas_int i = 0; i < n; i++) {
        BLAS_REAL p = x[ix] * y[iy];
        if (BLAS_UNLIKELY(blas_is_inf(sum))) {
            sum = sum + p;
            ix += incx;
            iy += incy;
            continue;
        }
        BLAS_REAL t       = p - c;
        BLAS_REAL new_sum = sum + t;
        c   = (new_sum - sum) - t;
        sum = new_sum;
        ix += incx;
        iy += incy;
    }
    return sum;
}
