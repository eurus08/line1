/*
 * tests/test_scal.c — Correctness tests for blas_scal
 *
 * Testing philosophy (see BLAS1_Build_Plan, Phase 3):
 *   Floating-point results are never compared with ==. Every check uses
 *   an epsilon tolerance — absolute tolerance for results near zero,
 *   relative tolerance for large-magnitude results.
 *
 * blas_scal computes x[i] <- alpha * x[i] in place — a single vector,
 * no second operand. As with test_axpy.c, checks compare the entire
 * output vector element-by-element against a hand-computed expected
 * array.
 *
 * Coverage (per build plan step 3.3, mirroring test_axpy.c's pattern):
 *   1. Known-answer tests (hand-computed)
 *   2. Edge cases: length-1, alpha == 0 (explicit zeroing / NaN-avoidance),
 *      alpha == 1 (no-op fast path), stride != 1 (incl. negative), n <= 0
 *   3. Precision test vs a long double reference
 *
 * IMPORTANT DISTINCTION vs axpy's alpha==0 fast path:
 *   axpy's alpha==0 means "leave y completely unchanged."
 *   scal's  alpha==0 means "zero x completely" — a different, ACTIVE
 *   result, not a no-op. Both must avoid letting NaN/Inf in the input
 *   propagate through the multiply, but the expected output differs:
 *   scal's expected output is all-zero, not "unchanged."
 */

#include "blas1/scal.h"
#include "blas1/types.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Minimal test harness (same pattern as test_dot.c / test_axpy.c)     */
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

    /* x <- 3 * x
     * x = [1, 2, 3] -> expected = [3, 6, 9] */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL expected[] = {3.0, 6.0, 9.0};

        blas_scal(3, 3.0, x, 1);
        check_vector("scal(alpha=3, x=[1,2,3]) == [3,6,9]", x, expected, 3, ABS_TOL);
    }

    /* Negative alpha and negative values, to exercise sign handling.
     * x = [4, -2, 0], alpha = -2.5 -> expected = [-10, 5, -0] */
    {
        BLAS_REAL x[] = {4.0, -2.0, 0.0};
        BLAS_REAL expected[] = {-10.0, 5.0, 0.0};

        blas_scal(3, -2.5, x, 1);
        check_vector("scal(alpha=-2.5) with negatives", x, expected, 3, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  2. Edge cases                                                       */
/* ------------------------------------------------------------------ */

static void test_edge_cases(void)
{
    printf("-- edge cases --\n");

    /* n <= 0 must be a complete no-op: x unchanged. */
    {
        BLAS_REAL x[] = {99.0, 88.0};
        BLAS_REAL expected[] = {99.0, 88.0};

        blas_scal(0, 5.0, x, 1);
        check_vector("scal with n=0 leaves x unchanged", x, expected, 2, ABS_TOL);

        blas_scal(-3, 5.0, x, 1);
        check_vector("scal with n<0 leaves x unchanged", x, expected, 2, ABS_TOL);
    }

    /* Length-1 vector. */
    {
        BLAS_REAL x[] = {7.0};
        BLAS_REAL expected[] = {42.0};

        blas_scal(1, 6.0, x, 1);
        check_vector("scal length-1", x, expected, 1, ABS_TOL);
    }

    /*
     * alpha == 1 fast path: must be a true no-op, not "multiply by 1
     * and happen to get the same answer." We poison x with NaN/Inf and
     * verify the underlying bytes are completely untouched.
     *
     * We deliberately use memcmp() rather than isnan()/isinf() here:
     * under -ffast-math, GCC assumes -ffinite-math-only, which means
     * isnan() and isinf() can be folded to a constant 0 at compile time
     * (NaN/Inf are assumed not to occur), making them unreliable as a
     * verification tool in this codebase's Release flag regime. A raw
     * byte comparison has no floating-point semantics to optimise away,
     * so it correctly proves the fast path skipped the memory pass
     * under any compiler flags.
     */
    {
        BLAS_REAL x[] = { (BLAS_REAL)NAN, (BLAS_REAL)INFINITY, 3.0 };
        BLAS_REAL before[3];
        memcpy(before, x, sizeof(x));

        blas_scal(3, 1.0, x, 1);

        int unchanged = (memcmp(x, before, sizeof(x)) == 0);
        report(unchanged, "scal alpha=1 leaves x untouched (incl. NaN/Inf slots)", 0.0, 0.0);
    }

    /*
     * alpha == 0 special case — the most important edge case for scal.
     *
     * Per src/scal.c, alpha == 0 must EXPLICITLY ZERO x via an assignment
     * loop, never via multiplication, because 0.0 * NaN = NaN and
     * 0.0 * Inf = NaN under IEEE 754. We poison x with NaN/Inf and expect
     * a clean all-zero result — proving the zeroing loop, not a multiply,
     * actually ran.
     */
    {
        BLAS_REAL x[] = { (BLAS_REAL)NAN, (BLAS_REAL)INFINITY, -(BLAS_REAL)INFINITY, 5.0 };
        BLAS_REAL expected[] = {0.0, 0.0, 0.0, 0.0};

        blas_scal(4, 0.0, x, 1);
        check_vector("scal alpha=0 zeroes x even with NaN/Inf present",
                     x, expected, 4, ABS_TOL);
    }

    /* Stride != 1: scale every other slot, leave the gaps untouched.
     *
     * x physically = [1, A, 2, B, 3, C] -> logical x = [1,2,3], incx=2
     * alpha = 10
     * logical x <- [10, 20, 30]
     * Gap elements A, B, C must be untouched.
     */
    {
        BLAS_REAL x[] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
        BLAS_REAL expected[] = {10.0, -1.0, 20.0, -2.0, 30.0, -3.0};

        blas_scal(3, 10.0, x, 2);
        check_vector("scal with stride 2 updates only logical elements",
                     x, expected, 6, ABS_TOL);
    }

    /* Stride != 1 combined with alpha == 0: zeroing loop with stride gaps.
     * x physically = [5.0, -1.0, -1.0, 10.0, -1.0, -1.0] -> logical x = [5.0, 10.0], incx=3
     * expected physical = [0.0, -1.0, -1.0, 0.0, -1.0, -1.0]
     * Gaps at indices 1, 2, 4, and 5 must remain untouched.
     */
    {
        BLAS_REAL x[] = {5.0, -1.0, -1.0, 10.0, -1.0, -1.0};
        BLAS_REAL expected[] = {0.0, -1.0, -1.0, 0.0, -1.0, -1.0};

        blas_scal(2, 0.0, x, 3);
        check_vector("scal alpha=0 with stride 3 zeroes only logical elements", x, expected, 6, ABS_TOL);
    }

    /* Negative (or zero) stride: matches reference BLAS's DSCAL
     * convention exactly -- DSCAL is documented to return immediately,
     * completely unmodified, for incx <= 0 (unlike DAXPY/DDOT, which
     * DO support negative strides). This used to walk backward from
     * the given pointer instead, which read/wrote out of bounds for
     * any caller passing the true start of the array (the standard
     * way to call it) with a negative stride.
     */
    {
        BLAS_REAL x[] = {1.0, 2.0, 3.0};
        BLAS_REAL expected[] = {1.0, 2.0, 3.0};   /* unchanged */

        blas_scal(3, 2.0, x, -1);
        check_vector("scal with negative incx is a no-op (x unchanged)",
                     x, expected, 3, ABS_TOL);

        blas_scal(3, 2.0, x, 0);
        check_vector("scal with incx == 0 is a no-op (x unchanged)",
                     x, expected, 3, ABS_TOL);
    }
}

/* ------------------------------------------------------------------ */
/*  3. Precision test vs a long double reference                        */
/* ------------------------------------------------------------------ */

static void test_precision(void)
{
    printf("-- precision test --\n");

    enum { N = 1000 };
    static BLAS_REAL x[N], x_ref_input[N];
    const BLAS_REAL alpha = (BLAS_REAL)0.987654321;

    for (int i = 0; i < N; i++) {
        x[i] = (BLAS_REAL)(0.001 * (i + 1) - 0.5);
        x_ref_input[i] = x[i]; /* keep an untouched copy for the reference */
    }

    blas_scal(N, alpha, x, 1);

    int worst_index = -1;
    double worst_rel = 0.0;

    for (int i = 0; i < N; i++) {
        long double ref = (long double)alpha * (long double)x_ref_input[i];
        double got = (double)x[i];
        double expected = (double)ref;

        double rel_diff = (fabs(expected) < ABS_TOL)
                         ? fabs(got - expected)
                         : fabs(got - expected) / fabs(expected);

        if (rel_diff > worst_rel) {
            worst_rel = rel_diff;
            worst_index = i;
        }
    }

    report(worst_rel < REL_TOL, "scal matches long double reference (worst-case element)",
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
    printf("=== test_scal ===\n");

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