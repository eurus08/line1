/*
 * tests/test_nrm2.c — Correctness tests for blas_nrm2
 *
 * Testing philosophy (see BLAS1_Build_Plan, Phase 3):
 *   Floating-point results are never compared with ==. Every check uses
 *   an epsilon tolerance — absolute tolerance for results near zero,
 *   relative tolerance for large-magnitude results.
 *
 * blas_nrm2 returns a single scalar: sqrt(sum(x[i]^2)), computed via a
 * numerically stable two-pass scaling algorithm (see src/nrm2.c) that
 * delegates pass 1 to blas_iamax(). This file links against both
 * nrm2.c and iamax.c for that reason.
 *
 * Coverage (per build plan step 3.4):
 *   1. Known-answer tests (hand-computed, e.g. 3-4-5 triangle)
 *   2. Edge cases: length-1, zero vector, stride != 1 (incl. negative), n <= 0
 *   3. Overflow test: large values (~1e200) that would overflow a naive
 *      sum-of-squares but must NOT overflow under the scaled algorithm
 *   4. Underflow test: small values (~1e-200) that would vanish under a
 *      naive sum-of-squares but must be recovered correctly
 *   5. Precision test vs a long double reference (well-conditioned case)
 *
 * IMPORTANT — why overflow/underflow checks use an ANALYTICAL reference,
 * not a long-double recomputation of sum-of-squares:
 *   For x[i] = 1e200, squaring it is 1e400, which overflows even long
 *   double (x86-64 extended precision tops out around 1e4932 in range,
 *   so 1e400 alone doesn't overflow long double, but using a naive sum
 *   of squares would defeat the entire purpose of this test — we want
 *   to confirm nrm2 matches a hand-derived analytical answer, not a
 *   "less naive but still naive" approach). For vectors built entirely
 *   from copies of one value v, ||x|| = |v| * sqrt(n) exactly — that
 *   closed form is the reference here, independent of squaring.
 *
 * IMPORTANT — why isnan()/isinf() are avoided for verification:
 *   As established in test_scal.c, -ffast-math implies
 *   -ffinite-math-only, under which GCC may fold isnan()/isinf() to a
 *   constant false. All checks below use ordinary finite-value
 *   comparisons (fabs, relative error) instead, which remain valid
 *   under any flag regime used by this project.
 */

#include "blas1/nrm2.h"
#include "blas1/types.h"

#include <stdio.h>
#include <math.h>
#include <float.h>

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
/*  1. Known-answer tests                                               */
/* ------------------------------------------------------------------ */

static void test_known_answers(void)
{
    printf("-- known-answer tests --\n");

    /* [5, 12] → 13 */
    {
        BLAS_REAL x[] = {5.0, 12.0};
        check_abs("nrm2([5,12]) == 13", blas_nrm2(2, x, 1), 13.0, ABS_TOL);
    }

    /* [8, 15] → 17 */
    {
        BLAS_REAL x[] = {8.0, 15.0};
        check_abs("nrm2([8,15]) == 17", blas_nrm2(2, x, 1), 17.0, ABS_TOL);
    }

    /* Mixed signs: [3, -4] → 5 */
    {
        BLAS_REAL x[] = {3.0, -4.0};
        check_abs("nrm2([3,-4]) with mixed signs == 5", blas_nrm2(2, x, 1), 5.0, ABS_TOL);
    }

    /* Classic 3-4-5 right triangle: ||[3,4]|| = 5 */
    {
        BLAS_REAL x[] = {3.0, 4.0};
        BLAS_REAL result = blas_nrm2(2, x, 1);
        check_abs("nrm2([3,4]) == 5", result, 5.0, ABS_TOL);
    }

    /* Unit vector along one axis: ||[0,0,1,0]|| = 1 */
    {
        BLAS_REAL x[] = {0.0, 0.0, 1.0, 0.0};
        BLAS_REAL result = blas_nrm2(4, x, 1);
        check_abs("nrm2 of a unit basis vector == 1", result, 1.0, ABS_TOL);
    }

    /* Negative values: sign must not matter, only magnitude.
     * ||[-3, 4, -12]|| ... actually use a clean Pythagorean quadruple:
     * 1^2 + 2^2 + 2^2 = 1 + 4 + 4 = 9 -> sqrt(9) = 3 */
    {
        BLAS_REAL x[] = {-1.0, 2.0, -2.0};
        BLAS_REAL result = blas_nrm2(3, x, 1);
        check_abs("nrm2([-1,2,-2]) == 3", result, 3.0, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  2. Edge cases                                                       */
/* ------------------------------------------------------------------ */

static void test_edge_cases(void)
{
    printf("-- edge cases --\n");

    /* n <= 0 must return 0.0 — per nrm2.c's documented guard. */
    {
        BLAS_REAL x[] = {5.0, 6.0};
        BLAS_REAL result = blas_nrm2(0, x, 1);
        check_abs("nrm2 with n=0 returns 0.0", result, 0.0, ABS_TOL);

        result = blas_nrm2(-4, x, 1);
        check_abs("nrm2 with n<0 returns 0.0", result, 0.0, ABS_TOL);
    }

    /* Length-1 vector: norm of a single element is just its absolute value. */
    {
        BLAS_REAL x[] = {-7.0};
        BLAS_REAL result = blas_nrm2(1, x, 1);
        check_abs("nrm2 length-1 == |x[0]|", result, 7.0, ABS_TOL);
    }

    /* Zero vector: scale == 0, guarded explicitly before the divide in
     * nrm2.c (would otherwise be a division by zero in pass 2). */
    {
        BLAS_REAL x[] = {0.0, 0.0, 0.0, 0.0};
        BLAS_REAL result = blas_nrm2(4, x, 1);
        check_abs("nrm2 of zero vector == 0.0", result, 0.0, ABS_TOL);
    }

    /* Stride != 1: only the logical (strided) elements contribute.
     * x physically = [3, 99, 4, 99] -> logical x = [3,4], incx=2
     * ||[3,4]|| = 5, the gap value 99 must NOT contribute. */
    {
        BLAS_REAL x[] = {3.0, 99.0, 4.0, 99.0};
        BLAS_REAL result = blas_nrm2(2, x, 2);
        check_abs("nrm2 with stride 2 ignores gap elements", result, 5.0, ABS_TOL);
    }

    /* Negative stride: pointer points at the last logical element,
     * stride walks backwards. The norm is direction-independent, so
     * traversal order must not change the result.
     * x = [3,4], traversed in reverse (pointer starts at &x[1])
     * ||[3,4]|| = 5 regardless of traversal order. */
    {
        BLAS_REAL x[] = {3.0, 4.0};
        BLAS_REAL result = blas_nrm2(2, &x[1], -1);
        check_abs("nrm2 with negative incx == 5", result, 5.0, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  3. Overflow test — large values                                     */
/* ------------------------------------------------------------------ */

/*
 * x[i] = 1e200 for all i. A naive sum-of-squares computes (1e200)^2 =
 * 1e400, which overflows to +Inf in double precision (max ~1.8e308).
 * The scaled algorithm in nrm2.c must avoid this entirely.
 *
 * Analytical reference: for a vector of n identical elements v,
 *   ||x|| = sqrt(n * v^2) = |v| * sqrt(n)   exactly.
 */
static void test_overflow(void)
{
    printf("-- overflow test (large values, ~1e200) --\n");

    enum { N = 5 };
    BLAS_REAL x[N];
    const BLAS_REAL v = (BLAS_REAL)1e200;
    for (int i = 0; i < N; i++) {
        x[i] = v;
    }

    BLAS_REAL result = blas_nrm2(N, x, 1);
    double expected = fabs((double)v) * sqrt((double)N);

    /* Sanity check: the result should not be an extreme/overflowed value.
     * Note this check is NOT the primary safety net for this test — under
     * -ffast-math, a naive sum-of-squares implementation does not
     * reliably produce a literal +Inf (reassociation can leave the
     * intermediate sum at a large-but-finite value like DBL_MAX before
     * sqrt() is applied, which itself then returns a finite-but-wrong
     * number). The relative-error check below is what actually catches a
     * broken implementation in all cases; this is a secondary sanity
     * check valid mainly for strict-IEEE builds.
     *
     * We avoid isfinite()/isinf() for this check: under -ffast-math,
     * -ffinite-math-only lets GCC assume no Inf/NaN ever occurs, and
     * isfinite(Inf) has been directly confirmed to fold to true (and
     * isinf(Inf) to false) under these flags — the same class of issue
     * found with isnan() in test_scal.c. A plain ordering comparison
     * against DBL_MAX uses ordinary IEEE comparison semantics rather
     * than a classification function. */
    report(fabs((double)result) <= DBL_MAX, "nrm2 overflow test produces a finite result",
           (double)result, expected);

    check_rel("nrm2([1e200]*5) matches analytical sqrt(5)*1e200",
              (double)result, expected, REL_TOL);
}

/* ------------------------------------------------------------------ */
/*  4. Underflow test — small values                                    */
/* ------------------------------------------------------------------ */

/*
 * x[i] = 1e-200 for all i. A naive sum-of-squares computes (1e-200)^2 =
 * 1e-400, which underflows to exactly 0.0 in double precision (min
 * normal ~2.2e-308, min subnormal ~4.9e-324 — 1e-400 is unrepresentable
 * either way). The scaled algorithm must recover the correct non-zero
 * answer by factoring out the scale before squaring.
 *
 * Analytical reference: same closed form as the overflow test.
 */
static void test_underflow(void)
{
    printf("-- underflow test (small values, ~1e-200) --\n");

    enum { N = 7 };
    BLAS_REAL x[N];
    const BLAS_REAL v = (BLAS_REAL)1e-200;
    for (int i = 0; i < N; i++) {
        x[i] = v;
    }

    BLAS_REAL result = blas_nrm2(N, x, 1);
    double expected = fabs((double)v) * sqrt((double)N);

    /* The headline property: the result must be a real, non-zero
     * number. A naive implementation would silently return 0.0 here,
     * losing the entire vector. */
    report((double)result > 0.0, "nrm2 underflow test produces a non-zero result",
           (double)result, expected);

    check_rel("nrm2([1e-200]*7) matches analytical sqrt(7)*1e-200",
              (double)result, expected, REL_TOL);
}

/* ------------------------------------------------------------------ */
/*  5. Precision test vs a long double reference (well-conditioned)     */
/* ------------------------------------------------------------------ */

static void test_precision(void)
{
    printf("-- precision test: well-conditioned vector --\n");

    enum { N = 1000 };
    static BLAS_REAL x[N];

    for (int i = 0; i < N; i++) {
        x[i] = (BLAS_REAL)(1.0 + 0.01 * i);
    }

    /* Reference computed entirely in long double — safe here since all
     * values are modest in magnitude (no overflow risk in the squaring
     * step), unlike the overflow/underflow cases above. */
    long double sum_sq = 0.0L;
    for (int i = 0; i < N; i++) {
        long double xi = (long double)x[i];
        sum_sq += xi * xi;
    }
    long double ref = sqrtl(sum_sq);

    BLAS_REAL result = blas_nrm2(N, x, 1);
    check_rel("nrm2 matches long double reference (well-conditioned)",
              (double)result, (double)ref, REL_TOL);
}

/* ------------------------------------------------------------------ */
/*  main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== test_nrm2 ===\n");

    test_known_answers();
    test_edge_cases();
    test_overflow();
    test_underflow();
    test_precision();

    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);

    if (g_failures > 0) {
        printf("RESULT: FAIL (%d failure(s))\n", g_failures);
        return 1;
    }

    printf("RESULT: PASS\n");
    return 0;
}
