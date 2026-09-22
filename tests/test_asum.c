/*
 * tests/test_asum.c — Correctness tests for blas_asum
 *
 * Testing philosophy (see LINE1_Build_Plan, Phase 3):
 *   Floating-point results are never compared with ==. Every check uses
 *   an epsilon tolerance — absolute tolerance for results near zero,
 *   relative tolerance for large-magnitude results.
 *
 * blas_asum returns a single scalar: sum(|x[i]|), via Kahan-compensated
 * summation (see src/asum.c). It is the simplest accumulation in the
 * library — no scaling pass, no fast-path branches — so the tests focus
 * on fabs() correctness (sign handling, -0.0) and summation precision
 * rather than special-case edge behavior.
 *
 * Coverage (per build plan step 3.5, mirroring test_dot.c's pattern):
 *   1. Known-answer tests (hand-computed, mixed signs)
 *   2. Edge cases: length-1, zero vector, -0.0 handling, stride != 1
 *      (incl. negative), n <= 0
 *   3. Precision test vs a long double reference, including a case with
 *      many small terms designed to expose naive-summation rounding error
 */

#include "line1/asum.h"
#include "line1/types.h"

#include <stdio.h>
#include <math.h>

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

    /* All-positive: asum reduces to a plain sum.
     * |1| + |2| + |3| = 6 */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL result = blas_asum(3, x, 1);
        check_abs("asum([1,2,3]) == 6", result, 6.0, ABS_TOL);
    }

    /* Mixed signs: this is the entire point of asum vs a plain sum.
     * |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10
     * (a plain, non-absolute sum would give -2 instead) */
    {
        BLAS_REAL x[] = {1.0, -2.0, 3.0, -4.0};
        BLAS_REAL result = blas_asum(4, x, 1);
        check_abs("asum([1,-2,3,-4]) == 10", result, 10.0, ABS_TOL);
    }

    /* All-negative: every term flips sign under fabs().
     * |-1| + |-2| + |-3| = 6 */
    {
        BLAS_REAL x[] = {-1.0, -2.0, -3.0};
        BLAS_REAL result = blas_asum(3, x, 1);
        check_abs("asum([-1,-2,-3]) == 6", result, 6.0, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  2. Edge cases                                                       */
/* ------------------------------------------------------------------ */

static void test_edge_cases(void)
{
    printf("-- edge cases --\n");

    /* n <= 0 must return 0.0 — per asum.c's documented guard. */
    {
        BLAS_REAL x[] = {5.0, -6.0};
        BLAS_REAL result = blas_asum(0, x, 1);
        check_abs("asum with n=0 returns 0.0", result, 0.0, ABS_TOL);

        result = blas_asum(-4, x, 1);
        check_abs("asum with n<0 returns 0.0", result, 0.0, ABS_TOL);
    }

    /* Length-1 vector: asum of a single element is just its magnitude. */
    {
        BLAS_REAL x[] = {-9.5};
        BLAS_REAL result = blas_asum(1, x, 1);
        check_abs("asum length-1 == |x[0]|", result, 9.5, ABS_TOL);
    }

    /* Zero vector. */
    {
        BLAS_REAL x[] = {0.0, 0.0, 0.0};
        BLAS_REAL result = blas_asum(3, x, 1);
        check_abs("asum of zero vector == 0.0", result, 0.0, ABS_TOL);
    }

    /*
     * Negative zero: fabs(-0.0) must behave as +0.0 in the sum. This is
     * specifically called out in asum.c's header comment as a reason to
     * prefer fabs() over a hand-written "x > 0 ? x : -x" ternary, which
     * would NOT reliably normalise -0.0 the same way. We mix -0.0 with
     * an ordinary positive value so a sign bookkeeping bug would show
     * up as a wrong total, not just an invisible -0.0 result.
     */
    {
        BLAS_REAL x[] = {-0.0, 5.0, -0.0};
        BLAS_REAL result = blas_asum(3, x, 1);
        check_abs("asum treats -0.0 as 0.0 in the running sum", result, 5.0, ABS_TOL);
    }

    /* Stride != 1: only the logical (strided) elements contribute.
     * x physically = [1, 99, -2, 99, 3, 99] -> logical x = [1,-2,3], incx=2
     * asum = |1| + |-2| + |3| = 6, the gap value 99 must NOT contribute. */
    {
        BLAS_REAL x[] = {1.0, 99.0, -2.0, 99.0, 3.0, 99.0};
        BLAS_REAL result = blas_asum(3, x, 2);
        check_abs("asum with stride 2 ignores gap elements", result, 6.0, ABS_TOL);
    }

    /* Negative (or zero) stride: matches reference BLAS's DASUM
     * convention exactly -- "modified 3/93 to return if incx .le. 0."
     * DASUM does NOT support negative strides (unlike DAXPY/DDOT).
     * This used to walk backward from the given pointer instead, which
     * read out of bounds for any caller passing the true start of the
     * array (the standard way to call it) with a negative stride. */
    {
        BLAS_REAL x[] = {1.0, -2.0, 3.0};
        check_abs("asum with negative incx returns 0.0",
                  blas_asum(3, x, -1), 0.0, ABS_TOL);
        check_abs("asum with incx == 0 returns 0.0",
                  blas_asum(3, x, 0), 0.0, ABS_TOL);
    }

    /* Zeros interspersed: |0| + |2| + |0| + |4| = 6 */
    {
        BLAS_REAL x[] = {0.0, 2.0, 0.0, 4.0};
        BLAS_REAL result = blas_asum(4, x, 1);
        check_abs("asum interspersed zeros == 6", result, 6.0, ABS_TOL);
    }

    /* Fractional values: |0.5| + |0.25| + |0.125| = 0.875 */
    {
        BLAS_REAL x[] = {0.5, 0.25, 0.125};
        BLAS_REAL result = blas_asum(3, x, 1);
        check_abs("asum fractional entries == 0.875", result, 0.875, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  3. Precision tests vs a long double reference                       */
/* ------------------------------------------------------------------ */

/*
 * Reference implementation using long double for extra precision.
 *
 * CAVEAT for anyone extending this pattern: `long double` is only
 * genuinely wider than `double` on x86_64 (80-bit extended
 * precision there). On AArch64/ARM64 (Apple Silicon, Linux ARM64),
 * the platform ABI defines `long double` as IDENTICAL to `double` --
 * there is no extended-precision type at all on that platform. A
 * `long double` running-sum reference is safe for well-conditioned
 * inputs (moderate-magnitude terms, no sub-ULP cancellation), which
 * is all this function is used for below. It is NOT safe as a
 * reference for an adversarial test specifically designed to probe
 * sub-ULP accumulation effects -- see test_precision_many_small_terms()
 * below, which used to call this function and failed on ARM64 CI for
 * exactly this reason. That test now computes its reference
 * analytically instead, with no dependence on `long double` at all.
 */
static long double reference_asum_ld(blas_int n, const BLAS_REAL *x)
{
    long double sum = 0.0L;
    for (blas_int i = 0; i < n; i++) {
        sum += fabsl((long double)x[i]);
    }
    return sum;
}

static void test_precision_well_conditioned(void)
{
    printf("-- precision test: well-conditioned vector --\n");

    enum { N = 1000 };
    static BLAS_REAL x[N];

    for (int i = 0; i < N; i++) {
        /* Alternate sign so the test genuinely exercises fabs(), not
         * just a plain sum that happens to equal its own absolute sum. */
        BLAS_REAL sign = (i % 2 == 0) ? (BLAS_REAL)1.0 : (BLAS_REAL)-1.0;
        x[i] = sign * (BLAS_REAL)(1.0 + 0.001 * i);
    }

    long double ref    = reference_asum_ld(N, x);
    BLAS_REAL   result = blas_asum(N, x, 1);

    check_rel("blas_asum matches long double reference", (double)result, (double)ref, REL_TOL);
}

/*
 * Precision test designed to stress Kahan summation specifically.
 *
 * The classic adversarial pattern for compensated summation is NOT "one
 * huge term plus many tiny terms" — a direct probe showed that pattern
 * doesn't actually produce a large enough RELATIVE error to distinguish
 * Kahan from naive summation (the absolute loss is real, but tiny
 * relative to a 1e16-scale running sum). The pattern that actually
 * separates the two is a running sum that grows gradually as many
 * terms near machine epsilon are added one at a time: each addition
 * individually loses precision to rounding, and those losses compound.
 *
 * The term size must scale with the working precision's epsilon, not
 * be a single constant shared between double and float: DBL_EPSILON
 * (~2.22e-16) and FLT_EPSILON (~1.19e-7) differ by nine orders of
 * magnitude, so a term chosen to sit "just below epsilon" for double
 * is nine orders of magnitude too small to register at all in float --
 * confirmed empirically: forcing the double-scale 1e-16 term through
 * this project's actual blas_asum() under BLAS_USE_FLOAT produced
 * *zero* measurable contribution from Kahan's compensation (result
 * landed on exactly 1.0, identical to naive summation), because the
 * term is that far below what a float-precision compensation variable
 * can even represent as a meaningful residual against a sum near 1.0.
 *
 * Double-precision case (1e-16 term, N-1 = 1e6 additions), measured
 * directly against this project's asum.c:
 *   - naive summation:                  relative error ~9.996e-11
 *   - Kahan, strict IEEE (-O2, no -ffast-math): relative error ~3.7e-14
 *   - Kahan, this project's Release flags
 *     (-O3 -march=native -ffast-math):  relative error ~2.5e-11
 *
 * Float-precision case (1e-7 term, N-1 = 1e6 additions), measured the
 * same way against this project's actual blas_asum() built with
 * -DBLAS_USE_FLOAT:
 *   - naive summation, Debug (-O0):            relative error ~1.746e-2
 *   - Kahan,           Debug (-O0):             relative error ~2.167e-8
 *   - naive AND Kahan, Release (-ffast-math):   relative error ~2.094e-3
 *     (identical -- see below)
 *
 * As documented in asum.c, -ffast-math is known to neutralise some of
 * Kahan's compensation by allowing the compiler to reassociate the
 * floating-point operations the algorithm depends on. For double, some
 * benefit survives even under -ffast-math (Kahan ~2.5e-11 vs naive
 * ~9.996e-11, roughly 4x better). For float the effect is total: under
 * -ffast-math, Kahan's result is bit-for-bit identical to naive's here
 * -- the compiler's reassociation fully eliminates the compensation's
 * effect at this precision. The tolerance below is chosen accordingly:
 * strictly between the -ffast-math result (shared by both
 * implementations) and the Debug-build naive-summation error, so the
 * test still catches a regression to plain summation under Debug
 * builds, passes under Release for the real (if fully degraded under
 * that flag regime) Kahan implementation, and does not pretend float
 * -ffast-math builds get any of Kahan's real benefit -- they don't.
 * This mirrors the double case's documented caveat: the guarantee is
 * that the actual implementation passes under every flag regime this
 * project uses, not that Kahan's benefit is visible under all of them.
 */
static void test_precision_many_small_terms(void)
{
#if defined(BLAS_USE_FLOAT)
    #define STRESS_TERM     1e-7   /* just below FLT_EPSILON (~1.19e-7) */
    #define STRESS_REL_TOL  5e-3   /* between the -ffast-math result
                                       (~2.1e-3) and a Debug-build naive
                                       regression (~1.7e-2) */
#else
    #define STRESS_TERM     1e-16  /* just below DBL_EPSILON (~2.22e-16) */
    #define STRESS_REL_TOL  5e-11  /* between the -ffast-math Kahan
                                       result (~2.5e-11) and the naive
                                       result (~9.996e-11) */
#endif

    printf("-- precision test: many epsilon-scale terms (Kahan stress case) --\n");

    enum { N = 1000001 };
    static BLAS_REAL x[N];

    x[0] = (BLAS_REAL)1.0;
    for (int i = 1; i < N; i++) {
        x[i] = (BLAS_REAL)STRESS_TERM;
    }

    /*
     * Reference: computed analytically in ONE step, not via a
     * long-double accumulation loop.
     *
     * This test used to compute its reference the same way
     * reference_asum_ld() does -- a running sum in `long double`,
     * trusted to be more precise than the `double` result under
     * test. That assumption is platform-dependent in a way that
     * broke this specific test (caught via a real CI failure on
     * macOS/Apple Silicon): on x86_64, `long double` is 80-bit
     * extended precision, genuinely wider than `double`. On AArch64
     * (ARM64) -- both Apple Silicon and Linux ARM64 -- the platform
     * ABI defines `long double` as IDENTICAL to `double`, with no
     * extended-precision type at all. On ARM64, a `long double`
     * running-sum reference therefore suffers the EXACT SAME
     * sub-ULP rounding-away problem this test exists to detect in
     * the code under test -- corrupting the "ground truth" itself,
     * not just the thing being measured against it. Concretely: the
     * reference computed 1.0 as its answer instead of the true
     * 1.0000000001, because on ARM64 each individual addition of
     * 1e-16 to a running sum near 1.0 is below that sum's rounding
     * threshold and vanishes -- a million times in a row.
     *
     * The fix: this test's input is fully known and deterministic
     * (one 1.0 term, (N-1) identical STRESS_TERM terms), so the exact
     * answer can be computed in a single multiply and a single add
     * -- each individually correctly rounded under IEEE 754, with no
     * repeated sub-ULP accumulation for rounding to silently erase.
     * This is accurate to double precision's own limits on any
     * platform, with no dependence on `long double` at all.
     */
    double exact_expected = 1.0 + (double)(N - 1) * STRESS_TERM;

    BLAS_REAL result = blas_asum(N, x, 1);

    check_rel("blas_asum resolves epsilon-scale terms better than naive summation",
              (double)result, exact_expected, STRESS_REL_TOL);

#undef STRESS_TERM
#undef STRESS_REL_TOL
}

/* ------------------------------------------------------------------ */
/*  Overflow handling                                                    */
/* ------------------------------------------------------------------ */

/*
 * Regression test for a real bug: asum({1e308, 1e308, 1e308}) used to
 * return NaN instead of +Inf.
 *
 * Root cause: once the Kahan running sum overflows to +Inf, the
 * compensation term can itself become +Inf, and the next term's
 * `t = ax - c` becomes -Inf -- so `sum + t` computes Inf + (-Inf),
 * which IEEE 754 defines as NaN, even though every term summed is a
 * non-negative, finite magnitude and plain (uncompensated) summation
 * of the same values correctly overflows straight to +Inf with no NaN
 * involved. See src/asum.c's file header comment for the full
 * mechanism and fix (falling back to plain addition once the running
 * sum is already infinite).
 */
static void test_overflow(void)
{
    printf("-- overflow test (regression: asum({1e308,...}) used to return NaN) --\n");

    {
        BLAS_REAL x[3] = { (BLAS_REAL)1e308, (BLAS_REAL)1e308, (BLAS_REAL)1e308 };
        BLAS_REAL result = blas_asum(3, x, 1);
        report(isinf((double)result) && (double)result > 0.0,
               "asum({1e308, 1e308, 1e308}) is +Inf, not NaN",
               (double)result, 1.0/0.0);
    }

    /* Overflow partway through a longer vector, with finite terms
     * both before and after the point where sum first overflows --
     * exercises the "keep adding subsequent finite terms once sum is
     * already Inf" path, not just the single overflowing addition. */
    {
        BLAS_REAL x[5] = { (BLAS_REAL)1.0, (BLAS_REAL)1e308,
                           (BLAS_REAL)1e308, (BLAS_REAL)1e308,
                           (BLAS_REAL)2.0 };
        BLAS_REAL result = blas_asum(5, x, 1);
        report(isinf((double)result) && (double)result > 0.0,
               "asum with overflow partway through, finite terms after, is +Inf",
               (double)result, 1.0/0.0);
    }
}

/* ------------------------------------------------------------------ */
/*  main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== test_asum ===\n");

    test_known_answers();
    test_edge_cases();
    test_precision_well_conditioned();
    test_precision_many_small_terms();
    test_overflow();

    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);

    if (g_failures > 0) {
        printf("RESULT: FAIL (%d failure(s))\n", g_failures);
        return 1;
    }

    printf("RESULT: PASS\n");
    return 0;
}
