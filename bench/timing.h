/*
 * bench/timing.h — Portable high-resolution timing utility
 *
 * Step 4.3 of the build plan. Written first (out of numeric order)
 * because bench_dot.c (step 4.1) depends on it.
 *
 * Why clock_gettime(CLOCK_MONOTONIC) and not clock() or time()?
 *   - clock() measures CPU time, not wall-clock time. For a memory-bound
 *     BLAS 1 routine we want wall-clock time, since stalls waiting on
 *     memory are exactly what we're trying to measure.
 *   - time() only has 1-second resolution — useless for timing a single
 *     dot() call, which takes nanoseconds to milliseconds.
 *   - CLOCK_MONOTONIC never jumps backwards (unlike CLOCK_REALTIME, which
 *     can be adjusted by NTP). It's the right clock for measuring elapsed
 *     intervals, as opposed to wall-clock date/time.
 *
 * This is header-only (BLAS_INLINE functions) so both bench_dot.c and
 * bench_axpy.c can #include it directly with no separate .c file or
 * extra link step.
 */

#ifndef LINE1_BENCH_TIMING_H
#define LINE1_BENCH_TIMING_H

/*
 * Feature-test macro note:
 *   This project builds with -std=c11 (strict ISO C, not -std=gnu11 -- see
 *   the root CMakeLists.txt: CMAKE_C_EXTENSIONS OFF). Under strict C11,
 *   glibc hides POSIX-only declarations like clock_gettime() and
 *   CLOCK_MONOTONIC unless a POSIX feature-test macro is defined first,
 *   BEFORE any system header is included. _POSIX_C_SOURCE 199309L is the
 *   value that specifically unlocks clock_gettime (the value comes from
 *   POSIX.1b, the realtime-extensions spec that introduced it). Without
 *   this, gcc -std=c11 fails with "CLOCK_MONOTONIC undeclared".
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <time.h>
#include <stdlib.h>   /* qsort */
#include "line1/types.h"  /* BLAS_INLINE */

/* ---------------------------------------------------------------------
 * bench_now() — current monotonic time, in seconds, as a double.
 *
 * struct timespec gives seconds + nanoseconds separately (to avoid
 * precision loss). We combine them into one double for convenience;
 * a double has ~15-17 significant decimal digits, more than enough
 * range+precision for measuring intervals of nanoseconds up to hours
 * in the same benchmark run.
 * ------------------------------------------------------------------- */
BLAS_INLINE double bench_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------------
 * bench_cmp_double() — comparator for qsort, used by bench_median().
 * ------------------------------------------------------------------- */
static int bench_cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

/* ---------------------------------------------------------------------
 * bench_median() — median of an array of timing samples.
 *
 * The build plan specifically calls for median, not mean: a single OS
 * scheduling interrupt, page fault, or thermal throttle event can make
 * one trial out of 100 take 10-100x longer than the rest. The mean
 * would be dragged upward by that one outlier; the median ignores it
 * entirely as long as it's not literally the middle sample.
 *
 * NOTE: this sorts the array in place. Callers should treat the
 * `samples` array as consumed after calling this.
 * ------------------------------------------------------------------- */
BLAS_INLINE double bench_median(double *samples, int count)
{
    qsort(samples, (size_t)count, sizeof(double), bench_cmp_double);

    if (count % 2 == 1) {
        return samples[count / 2];
    }
    /* Even count: average the two middle elements. */
    return 0.5 * (samples[count / 2 - 1] + samples[count / 2]);
}

#endif /* LINE1_BENCH_TIMING_H */