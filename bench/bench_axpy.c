/*
 * Feature-test macro: must be defined before ANY system header is
 * included anywhere in this translation unit. See bench/timing.h for
 * the full explanation of why -std=c11 needs this for clock_gettime().
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blas1/blas1.h"
#include "timing.h"

/*
 * bench/bench_axpy.c — Benchmark for blas_axpy
 *
 * Step 4.2 of the build plan. Same structure as bench_dot.c (step 4.1),
 * reusing bench/timing.h as-is. See bench_dot.c for the detailed
 * methodology writeup (warm vs. cold cache, why median not mean, etc.)
 * — this file only documents what's actually DIFFERENT about axpy.
 *
 * What's different from bench_dot:
 *
 *   1. Memory traffic per element is 3x sizeof(BLAS_REAL), not 2x.
 *      axpy computes  y[i] <- alpha * x[i] + y[i]
 *      That's a READ of x[i], a READ of y[i], AND a WRITE of y[i] back --
 *      three memory operations per element versus dot's two (dot only
 *      reads, never writes). FLOPs are still 2 per element (one multiply,
 *      one add), so axpy's arithmetic intensity is even lower than dot's:
 *        dot:  2 FLOPs / 16 bytes = 0.125 FLOPs/byte
 *        axpy: 2 FLOPs / 24 bytes = 0.083 FLOPs/byte
 *      axpy is, if anything, an even cleaner illustration of "BLAS 1 is
 *      memory-bandwidth limited" than dot is.
 *
 *   2. blas_axpy returns void -- there's no return value to "sink" into
 *      a volatile variable the way bench_dot.c did with blas_dot's
 *      result. Instead, after each call we take a volatile read of one
 *      element of y and fold it into a running volatile accumulator.
 *      This gives the compiler a visible, unavoidable use of y's new
 *      contents, so it cannot prove the axpy call had no effect and
 *      optimise it away.
 *
 *   3. axpy MUTATES y in place. Calling it 100 times in a row on the
 *      same y buffer (the warm-cache trial loop) means y keeps changing
 *      value every iteration -- unlike dot, which never modifies its
 *      inputs. We use a small alpha (1e-3) so this drift stays numerically
 *      tame across 100 repeated calls (confirmed empirically: starting
 *      from values in [-1,1) and accumulating 100 times with alpha=1e-3
 *      moves y by only a few tenths -- nowhere near overflow). We do NOT
 *      reset y between trials within a single warm/cold run, by design:
 *      resetting would require re-touching memory every trial, which
 *      would itself perturb the very cache behaviour we're trying to
 *      measure. The absolute value of y is irrelevant here; only the
 *      TIME each call takes is being measured.
 */

#define BENCH_TRIALS 100

/* Same eviction buffer size rationale as bench_dot.c: bigger than any
 * cache level we expect on the target machine. */
#define EVICT_BYTES   (64UL * 1024 * 1024)
#define EVICT_DOUBLES (EVICT_BYTES / sizeof(BLAS_REAL))

/* A small, fixed alpha keeps y's drift over repeated warm-cache calls
 * numerically gentle (see file header comment, point 3). */
#define BENCH_ALPHA ((BLAS_REAL)1e-3)

/* ---------------------------------------------------------------------
 * evict_cache() -- identical technique to bench_dot.c's version: a real
 * read-modify-write sweep over a large buffer, forcing x and y out of
 * cache before a cold-cache trial.
 * ------------------------------------------------------------------- */
static BLAS_REAL evict_cache(BLAS_REAL *evict_buf, blas_int evict_n)
{
    BLAS_REAL acc = (BLAS_REAL)0.0;
    for (blas_int i = 0; i < evict_n; i++) {
        evict_buf[i] += (BLAS_REAL)1.0;
        acc += evict_buf[i];
    }
    return acc;
}

/* ---------------------------------------------------------------------
 * fill_vector() -- same deterministic xorshift fill as bench_dot.c.
 * ------------------------------------------------------------------- */
static void fill_vector(BLAS_REAL *v, blas_int n, unsigned int seed)
{
    unsigned int state = seed;
    for (blas_int i = 0; i < n; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        BLAS_REAL frac = (BLAS_REAL)(state % 1000000u) / (BLAS_REAL)1000000.0;
        v[i] = (BLAS_REAL)2.0 * frac - (BLAS_REAL)1.0;
    }
}

/* ---------------------------------------------------------------------
 * run_warm() -- median time for BENCH_TRIALS back-to-back axpy calls on
 * data that is already resident. y is mutated by every call; we read
 * one element back afterward purely to keep the call from being
 * optimised away (see file header, point 2).
 * ------------------------------------------------------------------- */
static double run_warm(const BLAS_REAL *x, BLAS_REAL *y, blas_int n)
{
    double samples[BENCH_TRIALS];
    volatile BLAS_REAL sink = (BLAS_REAL)0.0;

    for (int t = 0; t < BENCH_TRIALS; t++) {
        double t0 = bench_now();
        blas_axpy(n, BENCH_ALPHA, x, 1, y, 1);
        double t1 = bench_now();
        sink = y[0];
        samples[t] = t1 - t0;
    }

    (void)sink;
    return bench_median(samples, BENCH_TRIALS);
}

/* ---------------------------------------------------------------------
 * run_cold() -- median time for BENCH_TRIALS axpy calls, each preceded
 * by a deliberate cache eviction so every call pays the full memory cost
 * of reading x, reading y, and writing y back.
 * ------------------------------------------------------------------- */
static double run_cold(const BLAS_REAL *x, BLAS_REAL *y, blas_int n,
                        BLAS_REAL *evict_buf, blas_int evict_n)
{
    double samples[BENCH_TRIALS];
    volatile BLAS_REAL sink = (BLAS_REAL)0.0;

    for (int t = 0; t < BENCH_TRIALS; t++) {
        sink = evict_cache(evict_buf, evict_n); /* not timed -- eviction only */

        double t0 = bench_now();
        blas_axpy(n, BENCH_ALPHA, x, 1, y, 1);
        double t1 = bench_now();
        sink = y[0];
        samples[t] = t1 - t0;
    }

    (void)sink;
    return bench_median(samples, BENCH_TRIALS);
}

/* ---------------------------------------------------------------------
 * report_row() -- compute and print GFlop/s and GB/s for one size.
 *
 * blas_axpy: 2 FLOPs/element (1 mul + 1 add), 3 elements of traffic per
 * iteration (read x, read y, write y).
 * ------------------------------------------------------------------- */
static void report_row(blas_int n, double warm_t, double cold_t)
{
    double flops = 2.0 * (double)n;
    double bytes = 3.0 * (double)n * (double)sizeof(BLAS_REAL);

    double warm_gflops = (warm_t > 0.0) ? (flops / warm_t) / 1e9 : 0.0;
    double warm_gbs    = (warm_t > 0.0) ? (bytes / warm_t) / 1e9 : 0.0;
    double cold_gflops = (cold_t > 0.0) ? (flops / cold_t) / 1e9 : 0.0;
    double cold_gbs    = (cold_t > 0.0) ? (bytes / cold_t) / 1e9 : 0.0;

    printf("%12lld %14.3f %12.4f %12.4f %14.3f %12.4f %12.4f\n",
           (long long)n,
           warm_t * 1e6, warm_gflops, warm_gbs,
           cold_t * 1e6, cold_gflops, cold_gbs);
}

int main(void)
{
    /* Same sweep range and intermediate points as bench_dot.c, for a
     * directly comparable table. */
    const blas_int sizes[] = {
        1000,
        4000,
        16000,
        64000,
        256000,
        1000000,
        4000000,
        16000000,
        64000000,
        100000000
    };
    const int num_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    blas_int max_n = sizes[num_sizes - 1];

    printf("=================================================================="
           "===========\n");
    printf(" blas_axpy benchmark\n");
    printf(" Precision: %s   Trials per size: %d (median reported)\n",
           (sizeof(BLAS_REAL) == sizeof(double)) ? "double" : "float",
           BENCH_TRIALS);
    printf(" SIMD backend: %s\n",
#if defined(USE_AVX2)
           "avx2"
#elif defined(USE_NEON)
           "neon"
#else
           "generic (scalar)"
#endif
    );
    printf("=================================================================="
           "===========\n");

    BLAS_REAL *x = (BLAS_REAL *)malloc((size_t)max_n * sizeof(BLAS_REAL));
    BLAS_REAL *y = (BLAS_REAL *)malloc((size_t)max_n * sizeof(BLAS_REAL));
    BLAS_REAL *evict_buf =
        (BLAS_REAL *)malloc((size_t)EVICT_DOUBLES * sizeof(BLAS_REAL));

    if (!x || !y || !evict_buf) {
        fprintf(stderr, "bench_axpy: allocation failure for n=%lld\n",
                (long long)max_n);
        free(x);
        free(y);
        free(evict_buf);
        return 1;
    }

    fill_vector(x, max_n, 12345u);
    fill_vector(y, max_n, 67890u);
    memset(evict_buf, 0, (size_t)EVICT_DOUBLES * sizeof(BLAS_REAL));

    printf("%12s %14s %12s %12s %14s %12s %12s\n",
           "n", "warm us", "warm GF/s", "warm GB/s",
           "cold us", "cold GF/s", "cold GB/s");
    printf("------------------------------------------------------------------"
           "-----------\n");

    for (int s = 0; s < num_sizes; s++) {
        blas_int n = sizes[s];

        double warm_t = run_warm(x, y, n);
        double cold_t = run_cold(x, y, n, evict_buf, (blas_int)EVICT_DOUBLES);

        report_row(n, warm_t, cold_t);
    }

    printf("=================================================================="
           "===========\n");
    printf("Notes:\n");
    printf("  - 'warm' = data resident from a previous touch; repeated calls\n");
    printf("    on the same buffers (best case, cache-friendly).\n");
    printf("  - 'cold' = cache deliberately flushed before every call (worst\n");
    printf("    case, first-touch).\n");
    printf("  - blas_axpy is %zu FLOPs and %zu bytes per element (read x,\n",
           (size_t)2, (size_t)(3 * sizeof(BLAS_REAL)));
    printf("    read y, write y) -- an even lower arithmetic intensity than\n");
    printf("    dot, since axpy also pays a write cost dot never does.\n");
    printf("  - Compare these GB/s figures directly against bench_dot's --\n");
    printf("    they should land in a similar range, since both are bound\n");
    printf("    by the same DRAM bandwidth ceiling, not by FLOPs.\n");

    free(x);
    free(y);
    free(evict_buf);
    return 0;
}
