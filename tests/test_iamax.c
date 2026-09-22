/*
 * tests/test_iamax.c — Correctness tests for blas_iamax
 *
 * Testing philosophy (see BLAS1_Build_Plan, Phase 3):
 *   Floating-point results are never compared with ==. Every check uses
 *   an epsilon tolerance — absolute tolerance for results near zero,
 *   relative tolerance for large-magnitude results. iamax itself
 *   returns an INTEGER index, so most checks here are exact integer
 *   comparisons rather than tolerance-based — but the underlying
 *   magnitude comparisons that DRIVE the index choice are still
 *   floating point, which is exactly what makes tie-breaking and NaN
 *   behavior worth testing carefully.
 *
 * iamax.c documents several precise contracts (see its header comment)
 * that this file tests directly, one assertion per documented behavior:
 *   1. 1-based indexing (Fortran/BLAS convention)
 *   2. n <= 0 returns the sentinel 0
 *   3. n == 1 returns 1 unconditionally (even for x[0] == 0.0)
 *   4. Tie-breaking: strictly first-occurrence wins (comparison is >)
 *   5. NaN values are silently skipped (NaN > anything is false in
 *      IEEE 754), and an all-NaN vector returns 1 (never updated)
 *   6. max_val is initialised to 0.0, not -Infinity — an all-zero
 *      vector returns 1, not "no maximum found"
 *   7. Stride != 1 (including negative stride)
 */

#include "blas1/iamax.h"
#include "blas1/types.h"

#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Minimal test harness (same pattern as previous test files)          */
/* ------------------------------------------------------------------ */

static int g_failures = 0;
static int g_checks    = 0;

/* iamax returns an exact integer — compare with == directly, no
 * epsilon needed. This is different from every other test file in
 * this suite, and deliberately so: the return type is blas_int, not
 * BLAS_REAL, and integer equality is exact by definition. */
static void check_eq(const char *name, blas_int got, blas_int expected)
{
    g_checks++;
    if (got == expected) {
        printf("  [PASS] %s\n", name);
    } else {
        g_failures++;
        printf("  [FAIL] %s  (got %lld, expected %lld)\n",
               name, (long long)got, (long long)expected);
    }
}

/* ------------------------------------------------------------------ */
/*  1. Known-answer tests                                               */
/* ------------------------------------------------------------------ */

static void test_known_answers(void)
{
    printf("-- known-answer tests --\n");

    /* Clear single maximum in the middle. 1-based: index of 9.0 is 3. */
    {
        BLAS_REAL x[] = {1.0, 5.0, 9.0, 2.0};
        blas_int result = blas_iamax(4, x, 1);
        check_eq("iamax([1,5,9,2]) == 3 (1-based)", result, 3);
    }

    /* Maximum magnitude via a negative value — iamax compares |x[i]|,
     * not x[i], so the most negative value can still be the maximum. */
    {
        BLAS_REAL x[] = {1.0, -8.0, 3.0};
        blas_int result = blas_iamax(3, x, 1);
        check_eq("iamax([1,-8,3]) == 2 (largest |.| via negative)", result, 2);
    }

    /* Maximum at the first position. */
    {
        BLAS_REAL x[] = {-100.0, 1.0, 2.0};
        blas_int result = blas_iamax(3, x, 1);
        check_eq("iamax with maximum at index 1", result, 1);
    }

    /* Maximum at the last position. */
    {
        BLAS_REAL x[] = {1.0, 2.0, -3.0, 4.0, -50.0};
        blas_int result = blas_iamax(5, x, 1);
        check_eq("iamax with maximum at last index", result, 5);
    }
}

/* ------------------------------------------------------------------ */
/*  2. Tie-breaking                                                      */
/* ------------------------------------------------------------------ */

static void test_tie_breaking(void)
{
    printf("-- tie-breaking --\n");

    /* Two elements share the same absolute value (5.0 and -5.0).
     * Per iamax.c: comparison is strictly >, so the FIRST occurrence
     * wins — the second equal value must NOT overwrite it. */
    {
        BLAS_REAL x[] = {1.0, 5.0, -5.0, 2.0};
        blas_int result = blas_iamax(4, x, 1);
        check_eq("iamax tie [1,5,-5,2] returns FIRST occurrence (index 2)", result, 2);
    }

    /* Three-way tie, same absolute value, all positive. */
    {
        BLAS_REAL x[] = {3.0, 7.0, 7.0, 7.0, 1.0};
        blas_int result = blas_iamax(5, x, 1);
        check_eq("iamax three-way tie returns first occurrence (index 2)", result, 2);
    }

    /* Tie at the very first two elements. */
    {
        BLAS_REAL x[] = {9.0, -9.0, 1.0};
        blas_int result = blas_iamax(3, x, 1);
        check_eq("iamax tie at indices 1,2 returns index 1", result, 1);
    }

}

/* ------------------------------------------------------------------ */
/*  3. NaN behaviour                                                     */
/* ------------------------------------------------------------------ */

static void test_nan_behavior(void)
{
    printf("-- NaN behaviour --\n");

    /* Per iamax.c: fabs(NaN) is NaN, and "NaN > anything" is false in
     * IEEE 754, so NaN entries are silently skipped. The function
     * should return the index of the largest FINITE element. */
    {
        BLAS_REAL x[] = { (BLAS_REAL)NAN, 3.0, (BLAS_REAL)NAN, 1.0 };
        blas_int result = blas_iamax(4, x, 1);
        check_eq("iamax skips NaN, returns largest finite element (index 2)", result, 2);
    }

    /* NaN at the position that would otherwise be the maximum: the NaN
     * must be skipped even though it appears where the "biggest number"
     * visually sits in the source. */
    {
        BLAS_REAL x[] = {1.0, 2.0, (BLAS_REAL)NAN, 2.0};
        blas_int result = blas_iamax(4, x, 1);
        /* Two finite elements tie at value 2.0 (indices 2 and 4);
         * first occurrence wins -> index 2. */
        check_eq("iamax with NaN where the max would be, ties go to first finite (index 2)",
                  result, 2);
    }

    /* All-NaN vector: per iamax.c, max_idx is never updated from its
     * initial value of 0, so the function returns 1 (1-based). This is
     * the most easily-missed edge case in the whole file — a test that
     * only checked "NaN is skipped" without this case would not catch
     * a regression where an all-NaN vector crashes or returns 0/garbage. */
    {
        BLAS_REAL x[] = { (BLAS_REAL)NAN, (BLAS_REAL)NAN, (BLAS_REAL)NAN };
        blas_int result = blas_iamax(3, x, 1);
        check_eq("iamax of all-NaN vector returns 1 (never updated)", result, 1);
    }
}

/* ------------------------------------------------------------------ */
/*  4. Edge cases                                                        */
/* ------------------------------------------------------------------ */

static void test_edge_cases(void)
{
    printf("-- edge cases --\n");

    /* n <= 0 must return the sentinel 0. */
    {
        BLAS_REAL x[] = {1.0, 2.0};
        blas_int result = blas_iamax(0, x, 1);
        check_eq("iamax with n=0 returns sentinel 0", result, 0);

        result = blas_iamax(-5, x, 1);
        check_eq("iamax with n<0 returns sentinel 0", result, 0);
    }

    /* n == 1 always returns 1 — even when x[0] is 0.0. This specifically
     * exercises the "n == 1" early-return branch in iamax.c, which
     * bypasses the max_val comparison loop entirely. A naive
     * reimplementation without that early return would still need
     * max_val initialised correctly to get this right; testing it
     * directly catches a regression in either version of the logic. */
    {
        BLAS_REAL x[] = {0.0};
        blas_int result = blas_iamax(1, x, 1);
        check_eq("iamax length-1 with x[0]==0.0 still returns 1", result, 1);
    }
    {
        BLAS_REAL x[] = {-42.0};
        blas_int result = blas_iamax(1, x, 1);
        check_eq("iamax length-1 with nonzero value returns 1", result, 1);
    }

    /* Regression: incx <= 0 must be checked BEFORE the n == 1 special
     * case, not after -- matching reference BLAS's IDAMAX, which
     * checks incx unconditionally first. A naive reordering (check
     * n == 1 before incx) would silently accept an invalid incx
     * whenever n happens to be 1, returning 1 instead of the sentinel
     * 0. */
    {
        BLAS_REAL x[] = {-42.0};
        check_eq("iamax with n==1 and incx<=0 still returns sentinel 0 (not 1)",
                  blas_iamax(1, x, -1), 0);
        check_eq("iamax with n==1 and incx==0 still returns sentinel 0 (not 1)",
                  blas_iamax(1, x, 0), 0);
    }

    /*
     * All-zero vector (n > 1): per iamax.c, max_val starts at 0.0 and
     * the comparison is strictly >, so 0.0 > 0.0 is always false and
     * max_idx never updates. The function must return 1 (the initial
     * index), not a sentinel and not garbage. This is worth testing
     * explicitly and separately from the n==1 case above, since this
     * path DOES go through the comparison loop (n > 1), unlike the
     * n==1 early return.
     */
    {
        BLAS_REAL x[] = {0.0, 0.0, 0.0, 0.0};
        blas_int result = blas_iamax(4, x, 1);
        check_eq("iamax of all-zero vector (n>1) returns 1", result, 1);
    }

    /* Stride != 1: only the logical (strided) elements participate.
     * x physically = [1, 99, 8, 99, 2, 99] -> logical x = [1,8,2], incx=2
     * Logical maximum is 8 at logical index 2 (1-based) -> result == 2.
     * The gap value 99 must NOT be considered, even though it is the
     * largest value physically present in memory. */
    {
        BLAS_REAL x[] = {1.0, 10.0, 8.0, 99.0, 2.0, 99.0};
        blas_int result = blas_iamax(3, x, 2);
        check_eq("iamax with stride 2 ignores gap elements", result, 2);
    }

    /* Negative (or zero) stride: matches reference BLAS's IDAMAX
     * convention exactly -- "modified 3/93 to return if incx .le. 0."
     * IDAMAX does NOT support negative strides (unlike DAXPY/DDOT).
     * This used to walk backward from the given pointer instead, which
     * read out of bounds for any caller passing the true start of the
     * array (the standard way to call it) with a negative stride. */
    {
        BLAS_REAL x[] = {1.0, 8.0, 2.0};
        check_eq("iamax with negative incx returns sentinel 0",
                  blas_iamax(3, x, -1), 0);
        check_eq("iamax with incx == 0 returns sentinel 0",
                  blas_iamax(3, x, 0), 0);
    }
}

/* ------------------------------------------------------------------ */
/*  main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== test_iamax ===\n");

    test_known_answers();
    test_tie_breaking();
    test_nan_behavior();
    test_edge_cases();

    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);

    if (g_failures > 0) {
        printf("RESULT: FAIL (%d failure(s))\n", g_failures);
        return 1;
    }

    printf("RESULT: PASS\n");
    return 0;
}
