/*
 * tests/test_correctness.c — Cross-function correctness tests
 *
 * Testing philosophy (see BLAS1_Build_Plan, Phase 3):
 *   Floating-point results are never compared with ==. Every check uses
 *   an epsilon tolerance — absolute tolerance for results near zero,
 *   relative tolerance for large-magnitude results.
 *
 * Every other file in this test suite (test_dot.c, test_axpy.c, ...)
 * verifies ONE function in isolation against hand-computed or
 * long-double-reference answers. This file is different in kind: it
 * checks that functions COMPOSE correctly when chained together, the
 * way real callers actually use a BLAS library (e.g. an iterative
 * solver calls axpy and dot back-to-back many times per iteration).
 * A bug in how one function's output feeds the next would not
 * necessarily be caught by either function's standalone unit tests.
 *
 * Coverage (per build plan step 3.7, exactly the three cases named in
 * the build plan, plus two additional compositions in the same spirit):
 *   1. axpy then dot matches an analytical result
 *   2. nrm2 of a (constructed) unit vector returns 1.0
 *   3. scal by 0 then nrm2 returns 0.0
 *   4. asum of a non-negative vector equals dot(x,x)/nrm2(x) reasoning
 *      check — specifically: asum(x) for an all-same-sign vector should
 *      match a hand-derived identity relating it to nrm2 and the
 *      vector's structure (see function comment for the exact identity
 *      used)
 *   5. iamax's returned index, fed back through the documented
 *      (result - 1) * incx convention, must point at the true maximum
 *      element in memory — exercising the "1-based index used as a
 *      memory offset" convention end-to-end rather than just checking
 *      the returned integer in isolation
 */

#include "blas1/dot.h"
#include "blas1/axpy.h"
#include "blas1/scal.h"
#include "blas1/nrm2.h"
#include "blas1/asum.h"
#include "blas1/iamax.h"
#include "blas1/types.h"

#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Minimal test harness (same pattern as previous test files)          */
/* ------------------------------------------------------------------ */

static int g_failures = 0;
static int g_checks    = 0;

#define ABS_TOL 1e-9
#define REL_TOL 1e-9

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

static void check_abs(const char *name, double got, double expected, double tol)
{
    double diff = fabs(got - expected);
    report(diff < tol, name, got, expected);
}

static void check_rel(const char *name, double got, double expected, double tol)
{
    if (fabs(expected) < ABS_TOL) {
        check_abs(name, got, expected, ABS_TOL);
        return;
    }
    double rel_diff = fabs(got - expected) / fabs(expected);
    report(rel_diff < tol, name, got, expected);
}

/* ------------------------------------------------------------------ */
/*  1. axpy then dot matches an analytical result                       */
/* ------------------------------------------------------------------ */

/*
 * Identity under test:  dot(x, alpha*x + y) = alpha * dot(x,x) + dot(x,y)
 *
 * We compute the right-hand side analytically from the ORIGINAL x and y
 * (before axpy mutates y), then perform the actual axpy + dot call
 * sequence and check the two agree. This is exactly the "axpy then dot"
 * pattern named in the build plan, and it's also literally how a
 * Conjugate Gradient solver's inner loop is structured — alpha*x + y
 * followed immediately by a dot product against x is not a contrived
 * example, it's the real workload BLAS Level 1 exists to serve.
 */
static void test_axpy_then_dot(void)
{
    printf("-- axpy then dot matches analytical result --\n");

    enum { N = 6 };
    BLAS_REAL x[N] = {1.0, 2.0, -3.0, 4.0, 0.5, -1.5};
    BLAS_REAL y[N] = {2.0, -1.0, 0.5, 3.0, -2.0, 1.0};
    const BLAS_REAL alpha = 2.5;

    /* Analytical reference computed from the UNMUTATED x and y. */
    BLAS_REAL dot_xx = blas_dot(N, x, 1, x, 1);
    BLAS_REAL dot_xy = blas_dot(N, x, 1, y, 1);
    double expected = (double)alpha * (double)dot_xx + (double)dot_xy;

    /* The actual call sequence: axpy mutates y in place, then dot reads
     * the mutated y. This is the composition being tested. */
    blas_axpy(N, alpha, x, 1, y, 1);
    BLAS_REAL result = blas_dot(N, x, 1, y, 1);

    check_rel("dot(x, axpy(alpha,x,y)) == alpha*dot(x,x) + dot(x,y)",
              (double)result, expected, REL_TOL);
}

/* ------------------------------------------------------------------ */
/*  2. nrm2 of a constructed unit vector returns 1.0                    */
/* ------------------------------------------------------------------ */

/*
 * Build a unit vector BY CONSTRUCTION: take an arbitrary vector, measure
 * its norm with nrm2, then scal it by (1/norm). The result must have
 * nrm2 == 1.0. This chains nrm2 -> scal -> nrm2 and verifies the two
 * functions agree on what "norm" means well enough to round-trip.
 */
static void test_nrm2_of_unit_vector(void)
{
    printf("-- nrm2 of a constructed unit vector == 1.0 --\n");

    enum { N = 4 };
    BLAS_REAL x[N] = {3.0, -1.0, 4.0, -1.5};

    BLAS_REAL original_norm = blas_nrm2(N, x, 1);
    BLAS_REAL inv_norm = (BLAS_REAL)(1.0 / (double)original_norm);

    blas_scal(N, inv_norm, x, 1);
    BLAS_REAL unit_norm = blas_nrm2(N, x, 1);

    check_abs("nrm2(scal(x, 1/nrm2(x))) == 1.0", (double)unit_norm, 1.0, ABS_TOL);
}

/* ------------------------------------------------------------------ */
/*  3. scal by 0 then nrm2 returns 0.0                                   */
/* ------------------------------------------------------------------ */

static void test_scal_zero_then_nrm2(void)
{
    printf("-- scal by 0 then nrm2 == 0.0 --\n");

    enum { N = 5 };
    BLAS_REAL x[N] = {7.0, -3.0, 12.5, -100.0, 0.001};

    blas_scal(N, 0.0, x, 1);
    BLAS_REAL result = blas_nrm2(N, x, 1);

    check_abs("nrm2(scal(x, 0.0)) == 0.0", (double)result, 0.0, ABS_TOL);
}

/* ------------------------------------------------------------------ */
/*  4. asum / dot / nrm2 consistency identity                            */
/* ------------------------------------------------------------------ */

/*
 * Identity under test: for a vector x, dot(x,x) == nrm2(x)^2, and
 * separately, for a vector with ALL NON-NEGATIVE entries, asum(x) ==
 * dot(x, ones), where ones is an all-1.0 vector of the same length.
 * Both are genuine cross-checks between independently implemented
 * functions, not restatements of either one's own algorithm:
 *   - dot(x,x) vs nrm2(x)^2 exercises that nrm2's two-pass scaling
 *     algorithm and dot's direct accumulation agree on the same
 *     mathematical quantity computed two structurally different ways.
 *   - asum(x) vs dot(x,ones) exercises that asum's running sum of
 *     fabs() values agrees with an unrelated function (dot) once the
 *     sign ambiguity is removed by construction (non-negative input).
 */
static void test_asum_dot_nrm2_consistency(void)
{
    printf("-- asum / dot / nrm2 cross-consistency --\n");

    enum { N = 5 };
    BLAS_REAL x[N] = {2.0, -3.5, 7.0, -1.25, 4.5};
    BLAS_REAL ones[N] = {1.0, 1.0, 1.0, 1.0, 1.0};

    /* dot(x,x) must equal nrm2(x)^2 */
    {
        BLAS_REAL dxx = blas_dot(N, x, 1, x, 1);
        BLAS_REAL norm = blas_nrm2(N, x, 1);
        double norm_sq = (double)norm * (double)norm;
        check_rel("dot(x,x) == nrm2(x)^2", (double)dxx, norm_sq, REL_TOL);
    }

    /* For a non-negative vector, asum(x) == dot(x, ones). */
    {
        BLAS_REAL x_nonneg[N];
        for (int i = 0; i < N; i++) {
            x_nonneg[i] = (BLAS_REAL)fabs((double)x[i]);
        }
        BLAS_REAL a = blas_asum(N, x_nonneg, 1);
        BLAS_REAL d = blas_dot(N, x_nonneg, 1, ones, 1);
        check_rel("asum(|x|) == dot(|x|, ones)", (double)a, (double)d, REL_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  5. iamax's 1-based index used as a real memory offset                */
/* ------------------------------------------------------------------ */

/*
 * iamax.c documents the caller convention explicitly:
 *     blas_int k = blas_iamax(n, x, 1);
 *     BLAS_REAL largest = x[k - 1];
 * This test exercises that convention end-to-end rather than just
 * checking the returned integer in isolation (which test_iamax.c
 * already does thoroughly). Here we use the result to actually index
 * back into the array — including the strided case, where the correct
 * memory offset is (result - 1) * incx, not just (result - 1) — and
 * confirm we land on the true maximum-magnitude element both times.
 */
static void test_iamax_as_memory_offset(void)
{
    printf("-- iamax result used as a real memory offset --\n");

    /* Contiguous case: offset is simply (result - 1). */
    {
        BLAS_REAL x[] = {1.0, -8.0, 3.0, -2.0};
        blas_int k = blas_iamax(4, x, 1);
        BLAS_REAL largest = x[k - 1];
        check_abs("x[iamax(x)-1] equals the true largest-magnitude element",
                  (double)largest, -8.0, ABS_TOL);
    }

    /* Strided case: offset is (result - 1) * incx, per iamax.c's
     * documented convention. x physically = [1, 99, -8, 99, 3, 99],
     * logical x = [1, -8, 3] with incx = 2. The 1-based logical index
     * of the maximum is 2; the correct memory offset is (2-1)*2 = 2,
     * which must land on the value -8.0, not on the gap element 99.0. */
    {
        BLAS_REAL x[] = {1.0, 99.0, -8.0, 99.0, 3.0, 99.0};
        blas_int incx = 2;
        blas_int k = blas_iamax(3, x, incx);
        BLAS_REAL largest = x[(k - 1) * incx];
        check_abs("x[(iamax-1)*incx] equals the true largest-magnitude element (strided)",
                  (double)largest, -8.0, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/* 6. axpy invariant: y += 0*x leaves y unchanged                    */
/* ------------------------------------------------------------------ */
static void test_axpy_zero_invariant(void)
{
    printf("-- axpy(0,x,y) leaves y unchanged --\n");

    enum { N = 3 };
    BLAS_REAL x[N] = {9.0, 9.0, 9.0};
    BLAS_REAL y[N] = {1.0, 2.0, 3.0};

    /* First mutate y to a known state via axpy */
    blas_axpy(N, 1.0, x, 1, y, 1);   /* y becomes [10.0, 11.0, 12.0] */

    BLAS_REAL nrm_before = blas_nrm2(N, y, 1);

    /* Now add zero — y must not change */
    blas_axpy(N, 0.0, x, 1, y, 1);
    BLAS_REAL nrm_after = blas_nrm2(N, y, 1);

    check_abs("nrm2(y) completely unchanged after axpy(0,...)", 
              (double)nrm_after, (double)nrm_before, ABS_TOL);
}

/* ------------------------------------------------------------------ */
/*  main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== test_correctness (cross-function tests) ===\n");

    test_axpy_then_dot();
    test_nrm2_of_unit_vector();
    test_scal_zero_then_nrm2();
    test_asum_dot_nrm2_consistency();
    test_iamax_as_memory_offset();

    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);

    if (g_failures > 0) {
        printf("RESULT: FAIL (%d failure(s))\n", g_failures);
        return 1;
    }

    printf("RESULT: PASS\n");
    return 0;
}
