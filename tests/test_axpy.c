/*
 * tests/test_axpy.c — Correctness tests for blas_axpy
 *
 * Testing philosophy (see BLAS1_Build_Plan, Phase 3):
 *   Floating-point results are never compared with ==. Every check uses
 *   an epsilon tolerance — absolute tolerance for results near zero,
 *   relative tolerance for large-magnitude results.
 *
 * blas_axpy computes y[i] <- alpha * x[i] + y[i] in place, so unlike
 * test_dot.c (which compares a single scalar), most checks here compare
 * entire output vectors element-by-element against a hand-computed or
 * long-double-reference expected array.
 *
 * Coverage (per build plan step 3.2, mirroring test_dot.c's pattern):
 *   1. Known-answer tests (hand-computed)
 *   2. Edge cases: length-1, alpha == 0 (no-op / NaN-avoidance), alpha == 1,
 *      stride != 1 (incl. negative), n <= 0
 *   3. Precision test vs a long double reference
 */

#include "blas1/axpy.h"
#include "blas1/types.h"

#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Minimal test harness (same pattern as test_dot.c)                   */
/* ------------------------------------------------------------------ */

static int g_failures = 0;
static int g_checks    = 0;

#if defined(BLAS_USE_FLOAT)
    #define ABS_TOL 1e-5
    #define REL_TOL 1e-5
#else
    #define ABS_TOL 1e-9
    #define REL_TOL 1e-9
#endif

static void report(int passed, const char *name, double got, double expected)
{
    g_checks++;
    if (passed) {
        printf("  [PASS] %s\n", name);
    } else {
        g_failures++;
        printf("  [FAIL] %s  (got %.17g, expected %.17g)\n", name, got, expected);
    }
}

/* Compare an entire vector against an expected array, element by element.
 * Reports ONE pass/fail per vector (not per element) so output stays
 * readable, but the failure message points at the first mismatching index. */
static void check_vector(const char *name, const BLAS_REAL *got,
                          const BLAS_REAL *expected, blas_int n, double tol)
{
    for (blas_int i = 0; i < n; i++) {
        double diff = fabs((double)got[i] - (double)expected[i]);
        if (diff >= tol) {
            g_checks++;
            g_failures++;
            printf("  [FAIL] %s  (index %lld: got %.17g, expected %.17g)\n",
                   name, (long long)i, (double)got[i], (double)expected[i]);
            return;
        }
    }
    g_checks++;
    printf("  [PASS] %s\n", name);
}

/* ------------------------------------------------------------------ */
/*  1. Known-answer tests                                               */
/* ------------------------------------------------------------------ */

static void test_known_answers(void)
{
    printf("-- known-answer tests --\n");

    /* y <- 2*x + y
     * x = [1,2,3], y = [10,20,30]
     * expected = [2*1+10, 2*2+20, 2*3+30] = [12, 24, 36] */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL y[] = {10.0, 20.0, 30.0};
        BLAS_REAL expected[] = {12.0, 24.0, 36.0};

        blas_axpy(3, 2.0, x, 1, y, 1);
        check_vector("axpy(alpha=2, x=[1,2,3], y=[10,20,30]) == [12,24,36]",
                     y, expected, 3, ABS_TOL);
    }

    /* Negative alpha and negative values, to exercise sign handling.
     * y <- -1.5*x + y
     * x = [4, -2, 0], y = [1, 1, 1]
     * expected = [-1.5*4+1, -1.5*-2+1, -1.5*0+1] = [-5, 4, 1] */
    {
        BLAS_REAL x[] = {4.0, -2.0, 0.0};
        BLAS_REAL y[] = {1.0, 1.0, 1.0};
        BLAS_REAL expected[] = {-5.0, 4.0, 1.0};

        blas_axpy(3, -1.5, x, 1, y, 1);
        check_vector("axpy(alpha=-1.5) with negatives", y, expected, 3, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  2. Edge cases                                                       */
/* ------------------------------------------------------------------ */

static void test_edge_cases(void)
{
    printf("-- edge cases --\n");

    /* n <= 0 must be a complete no-op: y unchanged. */
    {
        BLAS_REAL x[] = {1.0, 2.0};
        BLAS_REAL y[] = {99.0, 88.0};
        BLAS_REAL expected[] = {99.0, 88.0};

        blas_axpy(0, 5.0, x, 1, y, 1);
        check_vector("axpy with n=0 leaves y unchanged", y, expected, 2, ABS_TOL);

        blas_axpy(-3, 5.0, x, 1, y, 1);
        check_vector("axpy with n<0 leaves y unchanged", y, expected, 2, ABS_TOL);
    }

    /* Length-1 vector. */
    {
        BLAS_REAL x[] = {3.0};
        BLAS_REAL y[] = {4.0};
        BLAS_REAL expected[] = {3.0 * 5.0 + 4.0}; /* 19 */

        blas_axpy(1, 5.0, x, 1, y, 1);
        check_vector("axpy length-1", y, expected, 1, ABS_TOL);
    }

    /* alpha == 1: y <- x + y, plain addition. */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL y[] = {10.0, 20.0, 30.0};
        BLAS_REAL expected[] = {11.0, 22.0, 33.0};

        blas_axpy(3, 1.0, x, 1, y, 1);
        check_vector("axpy alpha=1 is plain addition", y, expected, 3, ABS_TOL);
    }

    /*
     * alpha == 0 fast path — the most important edge case for axpy.
     *
     * Per src/axpy.c, alpha == 0 must be a literal no-op (y left exactly
     * as it was), NOT "multiply by zero and add", because 0.0 * NaN and
     * 0.0 * Inf both produce NaN under IEEE 754. We deliberately poison x
     * with NaN/Inf here to prove the fast path actually skips the multiply
     * rather than relying on multiplication happening to work out.
     */
    {
        BLAS_REAL x[] = { (BLAS_REAL)NAN, (BLAS_REAL)INFINITY, -(BLAS_REAL)INFINITY };
        BLAS_REAL y[] = {1.0, 2.0, 3.0};
        BLAS_REAL expected[] = {1.0, 2.0, 3.0};

        blas_axpy(3, 0.0, x, 1, y, 1);
        check_vector("axpy alpha=0 leaves y unchanged even with NaN/Inf in x",
                     y, expected, 3, ABS_TOL);
    }

    /* Stride != 1: write into every other slot of y, read every other
     * slot of x. Untouched slots of y must remain exactly as they were.
     *
     * x physically = [1, 99, 2, 99, 3, 99] -> logical x = [1,2,3], incx=2
     * y physically = [10, A, 20, B, 30, C] -> logical y = [10,20,30], incy=2
     * alpha = 2
     * logical y <- [2*1+10, 2*2+20, 2*3+30] = [12, 24, 36]
     * The "gap" elements A, B, C must be untouched.
     */
    {
        BLAS_REAL x[] = {1.0, 99.0, 2.0, 99.0, 3.0, 99.0};
        BLAS_REAL y[] = {10.0, -1.0, 20.0, -2.0, 30.0, -3.0};
        BLAS_REAL expected[] = {12.0, -1.0, 24.0, -2.0, 36.0, -3.0};

        blas_axpy(3, 2.0, x, 2, y, 2);
        check_vector("axpy with stride 2 updates only logical elements",
                     y, expected, 6, ABS_TOL);
    }

    /* Negative stride: pointer points at the last logical element, stride
     * walks backwards. Mirrors the negative-stride test in test_dot.c.
     *
     * x = [1,2,3], traversed in reverse as 3,2,1 (pointer starts at &x[2])
     * y = [10,20,30], traversed forwards
     * alpha = 1
     * y[0] += x traversed first -> 3 :  10 + 3 = 13
     * y[1] += 2                 :       20 + 2 = 22
     * y[2] += 1                 :       30 + 1 = 31
     */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL y[] = {10.0, 20.0, 30.0};
        BLAS_REAL expected[] = {13.0, 22.0, 31.0};

        blas_axpy(3, 1.0, &x[2], -1, y, 1);
        check_vector("axpy with negative incx", y, expected, 3, ABS_TOL);
    }

    /* Idempotency / Repeated application
     * y <- 1.0 * x + y  (Applied twice)
     * x = [1,2,3], initial y = [0,0,0]
     * expected = [2,4,6] */
    {
        BLAS_REAL x[]        = {1.0, 2.0, 3.0};
        BLAS_REAL y[]        = {0.0, 0.0, 0.0};
        BLAS_REAL expected[] = {2.0, 4.0, 6.0};

        blas_axpy(3, 1.0, x, 1, y, 1);
        blas_axpy(3, 1.0, x, 1, y, 1);
        check_vector("axpy applied twice: y == 2x", y, expected, 3, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  3. Precision test vs a long double reference                        */
/* ------------------------------------------------------------------ */

static void test_precision(void)
{
    printf("-- precision test --\n");

    enum { N = 1000 };
    static BLAS_REAL x[N], y[N], y_ref_input[N];
    const BLAS_REAL alpha = (BLAS_REAL)1.23456789;

    for (int i = 0; i < N; i++) {
        x[i] = (BLAS_REAL)(0.001 * (i + 1));
        y[i] = (BLAS_REAL)(1000.0 - 0.1 * i);
        y_ref_input[i] = y[i]; /* keep an untouched copy for the reference */
    }

    blas_axpy(N, alpha, x, 1, y, 1);

    int worst_index = -1;
    double worst_rel = 0.0;

    for (int i = 0; i < N; i++) {
        long double ref = (long double)alpha * (long double)x[i]
                         + (long double)y_ref_input[i];
        double got = (double)y[i];
        double expected = (double)ref;

        double rel_diff = (fabs(expected) < ABS_TOL)
                         ? fabs(got - expected)
                         : fabs(got - expected) / fabs(expected);

        if (rel_diff > worst_rel) {
            worst_rel = rel_diff;
            worst_index = i;
        }
    }

    report(worst_rel < REL_TOL, "axpy matches long double reference (worst-case element)",
           worst_rel, 0.0);
    if (worst_rel >= REL_TOL) {
        printf("       worst mismatch at index %d, relative error %.3e\n",
               worst_index, worst_rel);
    }
}

/* ------------------------------------------------------------------ */
/*  main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== test_axpy ===\n");

    test_known_answers();
    test_edge_cases();
    test_precision();

    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);

    if (g_failures > 0) {
        printf("RESULT: FAIL (%d failure(s))\n", g_failures);
        return 1;
    }

    printf("RESULT: PASS\n");
    return 0;
}