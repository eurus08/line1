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
 *   3. Overflow test: large values that would overflow a naive
 *      sum-of-squares but must NOT overflow under the scaled algorithm
 *   4. Underflow test: small values that would vanish under a
 *      naive sum-of-squares but must be recovered correctly
 *   5. Precision test vs a long double reference (well-conditioned case)
 *
 * IMPORTANT — why overflow/underflow checks use an ANALYTICAL reference,
 * not a long-double recomputation of sum-of-squares:
 *   For x[i] = 1e200 (the double-precision magnitude used below), squaring
 *   it is 1e400, which overflows even long double (x86-64 extended
 *   precision tops out around 1e4932 in range, so 1e400 alone doesn't
 *   overflow long double, but using a naive sum of squares would defeat
 *   the entire purpose of this test — we want to confirm nrm2 matches a
 *   hand-derived analytical answer, not a "less naive but still naive"
 *   approach). For vectors built entirely from copies of one value v,
 *   ||x|| = |v| * sqrt(n) exactly — that closed form is the reference
 *   here, independent of squaring.
 *
 * IMPORTANT — why the overflow/underflow magnitudes and the tolerances
 * are precision-dependent (see OVERFLOW_V / UNDERFLOW_V / ABS_TOL /
 * REL_TOL below):
 *   1e200 cast to a 32-bit float overflows to +Inf immediately, at the
 *   test's own input construction, before blas_nrm2 ever runs — every
 *   check in this file would then be exercising Inf-arithmetic garbage,
 *   not the scaled algorithm under test. Symmetrically, 1e-200 cast to
 *   float underflows to exactly 0.0f on input. Both constants are
 *   rescaled for BLAS_USE_FLOAT to values that are themselves
 *   representable in float but still overflow/underflow when naively
 *   squared -- preserving the actual point of the test. The tolerances
 *   are similarly loosened for float: float has roughly 7 significant
 *   decimal digits (FLT_EPSILON ~1.19e-7) against double's ~16
 *   (DBL_EPSILON ~2.22e-16), so a tolerance tight enough to be
 *   meaningful for double will fail every float build regardless of
 *   correctness.
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

#if defined(BLAS_USE_FLOAT)
    #define ABS_TOL 1e-5
    #define REL_TOL 1e-5
#else
    #define ABS_TOL 1e-9
    #define REL_TOL 1e-9
#endif

/* Overflow/underflow test magnitudes -- see file header comment for why
 * these must be precision-dependent rather than a single shared 1e200 /
 * 1e-200 constant. */
#if defined(BLAS_USE_FLOAT)
    /* 1e30 is safely representable in float (FLT_MAX ~3.4e38), but its
     * square (1e60) overflows a naive sum-of-squares. */
    #define OVERFLOW_V  1e30
    /* 1e-30 is safely representable in float (FLT_MIN normal ~1.18e-38),
     * but its square (1e-60) underflows even float's subnormal range
     * (min subnormal ~1.4e-45). */
    #define UNDERFLOW_V 1e-30
#else
    #define OVERFLOW_V  1e200
    #define UNDERFLOW_V 1e-200
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

    /* Negative (or zero) stride: matches reference BLAS's DNRM2
     * convention exactly -- "IF (N.LT.1 .OR. INCX.LT.1) ... NORM = ZERO".
     * DNRM2 does NOT support negative strides (unlike DAXPY/DDOT).
     * This used to walk backward from the given pointer instead, which
     * read out of bounds for any caller passing the true start of the
     * array (the standard way to call it) with a negative stride. */
    {
        BLAS_REAL x[] = {3.0, 4.0};
        check_abs("nrm2 with negative incx returns 0.0",
                  blas_nrm2(2, x, -1), 0.0, ABS_TOL);
        check_abs("nrm2 with incx == 0 returns 0.0",
                  blas_nrm2(2, x, 0), 0.0, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  3. Overflow test — large values                                     */
/* ------------------------------------------------------------------ */

/*
 * x[i] = ~1e200 (double) / ~1e30 (float) for all i -- see OVERFLOW_V
 * above. A naive sum-of-squares computes v^2, which overflows to +Inf
 * in the working precision (double max ~1.8e308, float max ~3.4e38).
 * The scaled algorithm in nrm2.c must avoid this entirely.
 *
 * Analytical reference: for a vector of n identical elements v,
 *   ||x|| = sqrt(n * v^2) = |v| * sqrt(n)   exactly.
 */
static void test_overflow(void)
{
    printf("-- overflow test (large values, ~%g) --\n", (double)OVERFLOW_V);

    enum { N = 5 };
    BLAS_REAL x[N];
    const BLAS_REAL v = (BLAS_REAL)OVERFLOW_V;
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

    check_rel("nrm2(overflow-scale vector) matches analytical |v|*sqrt(5)",
              (double)result, expected, REL_TOL);
}

/* ------------------------------------------------------------------ */
/*  4. Underflow test — small values                                    */
/* ------------------------------------------------------------------ */

/*
 * x[i] = ~1e-200 (double) / ~1e-30 (float) for all i -- see UNDERFLOW_V
 * above. A naive sum-of-squares computes v^2, which underflows to
 * exactly 0.0 in the working precision (double: min normal ~2.2e-308,
 * min subnormal ~4.9e-324; float: min normal ~1.18e-38, min subnormal
 * ~1.4e-45 -- v^2 is unrepresentable either way, in both precisions).
 * The scaled algorithm must recover the correct non-zero answer by
 * factoring out the scale before squaring.
 *
 * Analytical reference: same closed form as the overflow test.
 */
static void test_underflow(void)
{
    printf("-- underflow test (small values, ~%g) --\n", (double)UNDERFLOW_V);

    enum { N = 7 };
    BLAS_REAL x[N];
    const BLAS_REAL v = (BLAS_REAL)UNDERFLOW_V;
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

    check_rel("nrm2(underflow-scale vector) matches analytical |v|*sqrt(7)",
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
/*  6. Infinity handling                                                */
/* ------------------------------------------------------------------ */

/*
 * Regression test for a real bug: blas_nrm2({Inf, 1, 2}) used to return
 * NaN instead of Inf.
 *
 * Root cause: the two-pass scaled algorithm picks scale = max(|x[i]|)
 * in pass 1, then computes sum((x[i]/scale)^2) in pass 2. If any
 * element is +-Inf, it dominates the max-search, so scale itself
 * becomes +Inf -- and for that same element, x[i]/scale is Inf/Inf,
 * which IEEE 754 defines as NaN. That NaN then poisons the entire
 * Kahan-compensated sum, corrupting a result whose true Euclidean norm
 * is unambiguously +Inf. nrm2.c now special-cases this (see its
 * blas_is_inf() guard) and returns scale directly before pass 2 runs.
 *
 * Note this file's own isinf()/isnan() calls below are fine: test
 * binaries are NOT built with this project's -ffast-math flags (see
 * tests/CMakeLists.txt), so none of the -ffinite-math-only caveats
 * documented in nrm2.c / types.h apply to this file itself -- only to
 * the library code under test.
 */
static void test_infinity(void)
{
    printf("-- infinity test (regression: nrm2({Inf,...}) used to return NaN) --\n");

    const double inf_val  =  1.0 / 0.0;
    const double ninf_val = -1.0 / 0.0;
    const double nan_val  =  0.0 / 0.0;

    /* A single +Inf element */
    {
        BLAS_REAL x[3] = { (BLAS_REAL)inf_val, (BLAS_REAL)1.0, (BLAS_REAL)2.0 };
        BLAS_REAL result = blas_nrm2(3, x, 1);
        report(isinf((double)result) && (double)result > 0.0,
               "nrm2({+Inf, 1, 2}) is +Inf, not NaN",
               (double)result, inf_val);
    }

    /* A single -Inf element -- BLAS_FABS(-Inf) is +Inf, same as above */
    {
        BLAS_REAL x[3] = { (BLAS_REAL)ninf_val, (BLAS_REAL)1.0, (BLAS_REAL)2.0 };
        BLAS_REAL result = blas_nrm2(3, x, 1);
        report(isinf((double)result) && (double)result > 0.0,
               "nrm2({-Inf, 1, 2}) is +Inf, not NaN",
               (double)result, inf_val);
    }

    /* Multiple Inf elements, mixed signs */
    {
        BLAS_REAL x[3] = { (BLAS_REAL)inf_val, (BLAS_REAL)inf_val, (BLAS_REAL)ninf_val };
        BLAS_REAL result = blas_nrm2(3, x, 1);
        report(isinf((double)result) && (double)result > 0.0,
               "nrm2({+Inf, +Inf, -Inf}) is +Inf",
               (double)result, inf_val);
    }

    /* Strided: the Inf is not at index 0 and not contiguous */
    {
        BLAS_REAL x[5] = { (BLAS_REAL)inf_val, (BLAS_REAL)99.0,
                           (BLAS_REAL)1.0,     (BLAS_REAL)99.0,
                           (BLAS_REAL)2.0 };
        BLAS_REAL result = blas_nrm2(3, x, 2);
        report(isinf((double)result) && (double)result > 0.0,
               "nrm2 with stride != 1 still returns +Inf for an Inf element",
               (double)result, inf_val);
    }

    /*
     * Inf together with NaN: per the C99/IEEE 754 special-case rule
     * for hypot() (hypot(+-inf, y) == +inf even if y is NaN),
     * infinity takes precedence over NaN. blas_nrm2's Inf guard fires
     * before pass 2 ever sees the NaN element, so this is expected
     * (and consistent) behavior, not an oversight.
     */
    {
        BLAS_REAL x[3] = { (BLAS_REAL)inf_val, (BLAS_REAL)nan_val, (BLAS_REAL)2.0 };
        BLAS_REAL result = blas_nrm2(3, x, 1);
        report(isinf((double)result) && (double)result > 0.0,
               "nrm2({+Inf, NaN, 2}) is +Inf (hypot's Inf-over-NaN precedent)",
               (double)result, inf_val);
    }

    /* NaN with NO Inf present: this should still be NaN -- confirms
     * the Inf guard doesn't overreach into misclassifying plain NaN
     * propagation as something to special-case. */
    {
        BLAS_REAL x[3] = { (BLAS_REAL)nan_val, (BLAS_REAL)1.0, (BLAS_REAL)2.0 };
        BLAS_REAL result = blas_nrm2(3, x, 1);
        report(isnan((double)result),
               "nrm2({NaN, 1, 2}) (no Inf present) is still NaN",
               (double)result, nan_val);
    }
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
    test_infinity();

    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);

    if (g_failures > 0) {
        printf("RESULT: FAIL (%d failure(s))\n", g_failures);
        return 1;
    }

    printf("RESULT: PASS\n");
    return 0;
}
