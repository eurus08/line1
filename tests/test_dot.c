/*
 * tests/test_dot.c — Test suite for blas_dot and blas_dot_kahan
 *
 * Testing philosophy for numerical code:
 *   Floating-point results are NEVER compared with ==.
 *   Instead we measure the absolute or relative error and assert it is
 *   below a tolerance (epsilon). Two tolerances are used:
 *
 *     TOL_LOOSE  1e-10   standard dot:  error grows as O(n * eps)
 *     TOL_TIGHT  1e-14   Kahan dot:     error stays at O(eps) regardless of n
 *
 * Test categories (in order):
 *   1. Basic known-answer tests  — hand-computed results
 *   2. Edge cases                — n=0, n=1, zero vector, mixed signs
 *   3. Stride != 1               — non-contiguous and strided access
 *   4. Precision / Kahan test    — compare standard vs Kahan on a
 *                                  large alternating series where rounding
 *                                  errors accumulate
 *
 * How to read the output:
 *   Each test prints PASS or FAIL with the measured error and tolerance.
 *   A non-zero exit code means at least one test failed.
 */

#include "blas1/dot.h"
#include "blas1/types.h"

#include <math.h>    /* fabs, sqrt */
#include <stdio.h>   /* printf */
#include <stdlib.h>  /* exit */
#include <string.h>  /* memset */

/* =========================================================================
 * Tolerances
 * ========================================================================= */

#define TOL_LOOSE  1e-10   /* acceptable for standard summation          */
#define TOL_TIGHT  1e-12   /* required for Kahan (and small-n standard)  */

/* =========================================================================
 * Test helpers
 * ========================================================================= */

static int g_tests  = 0;   /* total tests run  */
static int g_failed = 0;   /* total tests failed */

/*
 * check_abs — assert |result - expected| < tol
 * Prints PASS/FAIL with context. Updates global counters.
 */
static void check_abs(const char *name,
                      BLAS_REAL   result,
                      BLAS_REAL   expected,
                      BLAS_REAL   tol)
{
    g_tests++;
    BLAS_REAL err = fabs(result - expected);
    if (err < tol) {
        printf("  PASS  %-45s  err=%.3e  tol=%.3e\n", name, err, tol);
    } else {
        printf("  FAIL  %-45s  err=%.3e  tol=%.3e  got=%.15g  expected=%.15g\n",
               name, err, tol, (double)result, (double)expected);
        g_failed++;
    }
}

/*
 * check_rel — assert |result - expected| / |expected| < tol
 * Used when expected is large and absolute error would be misleading.
 */
static void check_rel(const char *name,
                      BLAS_REAL   result,
                      BLAS_REAL   expected,
                      BLAS_REAL   tol)
{
    g_tests++;
    BLAS_REAL err = fabs(result - expected) / fabs(expected);
    if (err < tol) {
        printf("  PASS  %-45s  rel_err=%.3e  tol=%.3e\n", name, err, tol);
    } else {
        printf("  FAIL  %-45s  rel_err=%.3e  tol=%.3e  got=%.15g  expected=%.15g\n",
               name, err, tol, (double)result, (double)expected);
        g_failed++;
    }
}

/* =========================================================================
 * 1. Basic known-answer tests
 *
 * These use tiny vectors whose dot products can be computed by hand.
 * They verify the fundamental accumulation is correct.
 * ========================================================================= */

static void test_basic(void)
{
    printf("\n--- 1. Basic known-answer tests ---\n");

    /* [1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32 */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL y[] = {4.0, 5.0, 6.0};
        check_abs("dot [1,2,3]·[4,5,6] == 32",
                  blas_dot(3, x, 1, y, 1), 32.0, TOL_TIGHT);
        check_abs("dot_kahan [1,2,3]·[4,5,6] == 32",
                  blas_dot_kahan(3, x, 1, y, 1), 32.0, TOL_TIGHT);
    }

    /* [1, 0, -1] . [1, 1, 1] = 1 + 0 - 1 = 0 (exact cancellation) */
    {
        BLAS_REAL x[] = {1.0, 0.0, -1.0};
        BLAS_REAL y[] = {1.0, 1.0,  1.0};
        check_abs("dot [1,0,-1]·[1,1,1] == 0",
                  blas_dot(3, x, 1, y, 1), 0.0, TOL_TIGHT);
        check_abs("dot_kahan [1,0,-1]·[1,1,1] == 0",
                  blas_dot_kahan(3, x, 1, y, 1), 0.0, TOL_TIGHT);
    }

    /* [2, 2] . [2, 2] = 4 + 4 = 8 */
    {
        BLAS_REAL x[] = {2.0, 2.0};
        BLAS_REAL y[] = {2.0, 2.0};
        check_abs("dot [2,2]·[2,2] == 8",
                  blas_dot(2, x, 1, y, 1), 8.0, TOL_TIGHT);
    }

    /* Self dot-product: [3,4].[3,4] = 9+16 = 25. sqrt would give norm=5 */
    {
        BLAS_REAL x[] = {3.0, 4.0};
        check_abs("dot [3,4]·[3,4] == 25",
                  blas_dot(2, x, 1, x, 1), 25.0, TOL_TIGHT);
    }

    /* All negative: [-1,-2].[-3,-4] = 3 + 8 = 11 */
    {
        BLAS_REAL x[] = {-1.0, -2.0};
        BLAS_REAL y[] = {-3.0, -4.0};
        check_abs("dot [-1,-2]·[-3,-4] == 11",
                  blas_dot(2, x, 1, y, 1), 11.0, TOL_TIGHT);
    }

    /* Mixed sign: [1,-1,1,-1].[1,1,1,1] = 1-1+1-1 = 0 */
    {
        BLAS_REAL x[] = { 1.0, -1.0,  1.0, -1.0};
        BLAS_REAL y[] = { 1.0,  1.0,  1.0,  1.0};
        check_abs("dot alternating signs == 0",
                  blas_dot(4, x, 1, y, 1), 0.0, TOL_TIGHT);
        check_abs("dot_kahan alternating signs == 0",
                  blas_dot_kahan(4, x, 1, y, 1), 0.0, TOL_TIGHT);
    }
}

/* =========================================================================
 * 2. Edge cases
 *
 * These probe the boundary conditions documented in dot.c:
 *   - n = 0 must return 0.0 (not crash)
 *   - n = 1 must return x[0]*y[0]
 *   - zero vector: dot should be exactly 0.0
 * ========================================================================= */

static void test_edge_cases(void)
{
    printf("\n--- 2. Edge cases ---\n");

    /* n = 0: guard clause must return 0 without touching pointers */
    {
        BLAS_REAL x[] = {9.9};
        BLAS_REAL y[] = {9.9};
        check_abs("dot n=0 returns 0",
                  blas_dot(0, x, 1, y, 1), 0.0, TOL_TIGHT);
        check_abs("dot_kahan n=0 returns 0",
                  blas_dot_kahan(0, x, 1, y, 1), 0.0, TOL_TIGHT);
    }

    /* n < 0: also treated as empty */
    {
        BLAS_REAL x[] = {9.9};
        BLAS_REAL y[] = {9.9};
        check_abs("dot n=-1 returns 0",
                  blas_dot(-1, x, 1, y, 1), 0.0, TOL_TIGHT);
    }

    /* n = 1: single element */
    {
        BLAS_REAL x[] = {7.0};
        BLAS_REAL y[] = {3.0};
        check_abs("dot n=1 returns 21",
                  blas_dot(1, x, 1, y, 1), 21.0, TOL_TIGHT);
        check_abs("dot_kahan n=1 returns 21",
                  blas_dot_kahan(1, x, 1, y, 1), 21.0, TOL_TIGHT);
    }

    /* zero vector: x=0 → dot must be 0 */
    {
        BLAS_REAL x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        BLAS_REAL y[]  = {1.0, 2.0, 3.0, 4.0, 5.0};
        check_abs("dot zero-x == 0",
                  blas_dot(5, x, 1, y, 1), 0.0, TOL_TIGHT);
        check_abs("dot_kahan zero-x == 0",
                  blas_dot_kahan(5, x, 1, y, 1), 0.0, TOL_TIGHT);
    }

    /* both zero: dot(0,0) = 0 */
    {
        BLAS_REAL x[4] = {0.0, 0.0, 0.0, 0.0};
        BLAS_REAL y[4] = {0.0, 0.0, 0.0, 0.0};
        check_abs("dot both-zero == 0",
                  blas_dot(4, x, 1, y, 1), 0.0, TOL_TIGHT);
    }
}

/* =========================================================================
 * 3. Stride tests
 *
 * Stride allows operating on non-contiguous data — e.g. a column of a
 * row-major matrix stored as a flat array.
 *
 * Layout used: arr[] = {a0, skip, a1, skip, a2, skip, a3, ...}
 *              incx=2 means take every other element starting at arr[0]
 *
 * We verify that strided dot matches the hand-computed answer on the
 * logical (non-strided) elements.
 * ========================================================================= */

static void test_stride(void)
{
    printf("\n--- 3. Stride tests ---\n");

    /*
     * incx=2, incy=2:  logical vectors are [1,2,3], [4,5,6]
     * Physical layout: x = [1, 99, 2, 99, 3], y = [4, 88, 5, 88, 6]
     * Expected dot: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
     */
    {
        BLAS_REAL x[] = {1.0, 99.0, 2.0, 99.0, 3.0};
        BLAS_REAL y[] = {4.0, 88.0, 5.0, 88.0, 6.0};
        check_abs("dot stride-2 [1,2,3]·[4,5,6] == 32",
                  blas_dot(3, x, 2, y, 2), 32.0, TOL_TIGHT);
        check_abs("dot_kahan stride-2 [1,2,3]·[4,5,6] == 32",
                  blas_dot_kahan(3, x, 2, y, 2), 32.0, TOL_TIGHT);
    }

    /*
     * incx=1, incy=3: x is contiguous, y is sparse
     * x=[1,2], y_logical=[10,20] from y=[10,_,_,20,_,_]
     * Expected: 1*10 + 2*20 = 10 + 40 = 50
     */
    {
        BLAS_REAL x[] = {1.0, 2.0};
        BLAS_REAL y[] = {10.0, 0.0, 0.0, 20.0, 0.0, 0.0};
        check_abs("dot incx=1 incy=3 result == 50",
                  blas_dot(2, x, 1, y, 3), 50.0, TOL_TIGHT);
    }

    /*
     * incx=3, incy=1: x is sparse, y is contiguous
     * x_logical=[5,10] from x=[5,_,_,10], y=[2,3]
     * Expected: 5*2 + 10*3 = 10 + 30 = 40
     */
    {
        BLAS_REAL x[] = {5.0, 0.0, 0.0, 10.0};
        BLAS_REAL y[] = {2.0, 3.0};
        check_abs("dot incx=3 incy=1 result == 40",
                  blas_dot(2, x, 3, y, 1), 40.0, TOL_TIGHT);
    }

    /*
     * Single-element with stride: n=1, incx=5, incy=5
     * Should only touch x[0] and y[0] regardless of stride.
     * Expected: 6.0 * 7.0 = 42.0
     */
    {
        BLAS_REAL x[] = {6.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        BLAS_REAL y[] = {7.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        check_abs("dot n=1 incx=5 incy=5 == 42",
                  blas_dot(1, x, 5, y, 5), 42.0, TOL_TIGHT);
    }

    /*
     * Mixed strides with n=4:
     * x has incx=2: logical = [x[0], x[2], x[4], x[6]] = [1, 3, 5, 7]
     * y has incy=1: logical = [y[0], y[1], y[2], y[3]] = [2, 4, 6, 8]
     * Expected: 1*2 + 3*4 + 5*6 + 7*8 = 2 + 12 + 30 + 56 = 100
     */
    {
        BLAS_REAL x[] = {1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0};
        BLAS_REAL y[] = {2.0, 4.0, 6.0, 8.0};
        check_abs("dot incx=2 incy=1 mixed strides == 100",
                  blas_dot(4, x, 2, y, 1), 100.0, TOL_TIGHT);
        check_abs("dot_kahan incx=2 incy=1 mixed strides == 100",
                  blas_dot_kahan(4, x, 2, y, 1), 100.0, TOL_TIGHT);
    }

    /*
     * Negative stride: BLAS convention walks backwards through memory.
     * We point at the LAST physical element and use incx = -1.
     * x physical = [1, 2, 3], traversed backwards as [3, 2, 1]
     * y physical = [10, 20, 30], traversed forwards as [10, 20, 30]
     * Expected: 3*10 + 2*20 + 1*30 = 100
     */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL y[] = {10.0, 20.0, 30.0};
        check_abs("dot negative stride x",
                  blas_dot(3, &x[2], -1, y, 1), 100.0, TOL_TIGHT);
    }
}

/* =========================================================================
 * 4. Precision / Kahan accuracy test
 *
 * We construct a vector designed to expose rounding-error accumulation in
 * the standard dot product.
 *
 * Technique — alternating large/small terms:
 *   x = y = [1e8, 1, 1e8, 1, ..., 1e8, 1]  (N_PAIRS pairs)
 *   dot(x, y) = N_PAIRS * 1e16 + N_PAIRS * 1.0
 *
 *   The exact result = N_PAIRS * (1e16 + 1)
 *
 *   With standard summation, adding 1.0 to a running sum of ~1e16 loses the
 *   1.0 entirely (it falls below the ULP of the accumulator). The final
 *   result is just N_PAIRS * 1e16, with an absolute error of N_PAIRS.
 *
 *   Kahan compensated summation tracks the dropped bits and recovers them,
 *   giving a result accurate to within O(eps * result).
 *
 * We don't assert that standard dot FAILS — with FMA enabled it may
 * actually do better than naive summation. We assert:
 *   (a) Kahan result has tight relative error (< TOL_TIGHT)
 *   (b) Standard result may have looser error (< TOL_LOOSE)
 *
 * The comparison against a long-double reference makes the "ground truth"
 * as accurate as the hardware allows.
 * ========================================================================= */

static void test_precision_kahan(void)
{
    printf("\n--- 4. Precision / Kahan accuracy test ---\n");

    /*
     * Build the alternating vector on the stack.
     * N_PAIRS pairs of (LARGE, SMALL) in both x and y.
     */
#define N_PAIRS  500
#define N_ELEMS  (2 * N_PAIRS)

    static BLAS_REAL x[N_ELEMS];
    static BLAS_REAL y[N_ELEMS];

    const BLAS_REAL LARGE = 1e8;
    const BLAS_REAL SMALL = 1.0;

    for (int i = 0; i < N_PAIRS; i++) {
        x[2*i]     = LARGE;  y[2*i]     = LARGE;
        x[2*i + 1] = SMALL;  y[2*i + 1] = SMALL;
    }

    /*
     * Reference: compute in long double (80-bit on x86) to get a
     * more accurate "ground truth" than double alone can provide.
     */
    long double ref = (long double)N_PAIRS * (LARGE * LARGE)
                    + (long double)N_PAIRS * (SMALL * SMALL);

    BLAS_REAL expected = (BLAS_REAL)ref;

    BLAS_REAL res_std   = blas_dot(N_ELEMS, x, 1, y, 1);
    BLAS_REAL res_kahan = blas_dot_kahan(N_ELEMS, x, 1, y, 1);

    /* Kahan must achieve tight accuracy */
    check_rel("dot_kahan large/small alternating — tight rel error",
              res_kahan, expected, TOL_TIGHT);

    /* Standard must be at least loosely correct (not wildly wrong) */
    check_rel("dot standard large/small alternating — loose rel error",
              res_std, expected, TOL_LOOSE);

    printf("    Reference (long double): %.15Lg\n", ref);
    printf("    Standard dot result:     %.15g\n", (double)res_std);
    printf("    Kahan   dot result:      %.15g\n", (double)res_kahan);

#undef N_PAIRS
#undef N_ELEMS
}

/* =========================================================================
 * 5. Large-n correctness
 *
 * Verify dot still gives the right answer for a vector of length 1 million.
 * This exercises the loop control logic (blas_int, no off-by-one errors).
 *
 * x = y = [1, 1, 1, ..., 1] (N elements)
 * dot(x, y) = N exactly, since each term is 1*1 = 1.
 * ========================================================================= */

static void test_large_n(void)
{
    printf("\n--- 5. Large-n correctness test ---\n");

#define N_LARGE 1000000

    static BLAS_REAL x[N_LARGE];
    static BLAS_REAL y[N_LARGE];

    for (int i = 0; i < N_LARGE; i++) {
        x[i] = 1.0;
        y[i] = 1.0;
    }

    BLAS_REAL expected = (BLAS_REAL)N_LARGE;

    /* Relative tolerance — standard dot may drift by O(n * eps) */
    check_rel("dot all-ones n=1e6 == 1e6",
              blas_dot(N_LARGE, x, 1, y, 1), expected, TOL_LOOSE);
    check_rel("dot_kahan all-ones n=1e6 == 1e6",
              blas_dot_kahan(N_LARGE, x, 1, y, 1), expected, TOL_TIGHT);

#undef N_LARGE
}

/* =========================================================================
 * main — run all test groups and report
 * ========================================================================= */

int main(void)
{
    printf("==========================================================\n");
    printf("  test_dot — blas_dot and blas_dot_kahan\n");
    printf("==========================================================\n");

    test_basic();
    test_edge_cases();
    test_stride();
    test_precision_kahan();
    test_large_n();

    printf("\n==========================================================\n");
    printf("  Results: %d / %d tests passed\n", g_tests - g_failed, g_tests);
    printf("==========================================================\n");

    return (g_failed == 0) ? 0 : 1;
}
