/*
 * bench/bench_dot.c — Benchmark for blas_dot
 *
 * Step 4.1 of the build plan.
 *
 * Goal: establish a BASELINE throughput number for blas_dot() before any
 * SIMD work happens in Phase 5. Nothing here makes the code faster — this
 * file only measures. "Correct -> Measured -> Optimised", in that order.
 *
 * What's measured, per vector size:
 *   - Warm-cache median time   (data already resident, repeated calls)
 *   - Cold-cache median time   (each trial forced to re-fetch from RAM)
 *   - GFlop/s   (billions of floating-point ops per second)
 *   - GB/s      (effective memory bandwidth)
 *
 * Why GFlop/s AND GB/s, and why this matters:
 *   blas_dot does 2 FLOPs per element (1 multiply + 1 add) but reads
 *   2 * sizeof(BLAS_REAL) bytes per element (x[i] and y[i]), and writes
 *   nothing back. That's an arithmetic intensity of (2 FLOPs) / (16 bytes)
 *   = 0.125 FLOPs/byte for double precision -- far below what's needed to
 *   keep modern CPU ALUs busy. This is *why* BLAS 1 routines are almost
 *   always memory-bandwidth limited, not compute limited (see build plan,
 *   Phase 4 introduction). The GB/s number is the one that actually
 *   explains the GFlop/s number here, not the other way around.
 *
 * Warm vs. cold cache methodology:
 *   WARM: allocate x, y once. "Touch" them (write once) so the OS has
 *         already handled page faults and the data is resident in some
 *         level of cache. Then call blas_dot() 100x back-to-back and take
 *         the median. Repeated calls on the same buffers mean later calls
 *         may find earlier-touched data still warm in L1/L2/L3 -- this
 *         measures best-case, cache-friendly performance.
 *
 *   COLD: before EVERY timed trial, deliberately evict x and y from cache
 *         by streaming through a large "eviction buffer" (sized larger
 *         than the biggest cache level we assume, currently 64 MB) that
 *         touches every cache line. This forces the very next blas_dot()
 *         call to fetch x and y from main memory rather than getting a
 *         free ride from a previous trial's cached data. This measures
 *         worst-case, first-touch performance -- closer to what happens
 *         in a real iterative solver where each vector is large and used
 *         once per iteration.
 *
 * Sweep range: 1K to 100M elements (per the build plan), stepping through
 * roughly one order of magnitude at a time, plus a couple of intermediate
 * points so the table also shows the L1 -> L2 -> L3 -> DRAM transitions
 * clearly rather than jumping straight from "fits in cache" to "doesn't".
 */

/*
 * Feature-test macro: must be defined before ANY system header is
 * included anywhere in this translation unit (stdio.h included below
 * already pulls in glibc's feature-test machinery, so timing.h defining
 * this on its own is too late). See bench/timing.h for the full
 * explanation of why -std=c11 needs this for clock_gettime().
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blas1/blas1.h"
#include "timing.h"

#define BENCH_TRIALS 100

/* Eviction buffer size: bigger than any cache level we expect to see on
 * the target machine (typical L3 is a few MB to a few tens of MB).
 * 64 MB of double = 8,388,608 elements -- chosen generously. */
#define EVICT_BYTES   (64UL * 1024 * 1024)
#define EVICT_DOUBLES (EVICT_BYTES / sizeof(BLAS_REAL))

/* ---------------------------------------------------------------------
 * evict_cache() -- read+write a large buffer to flush x/y out of cache.
 *
 * We do a real read-modify-write (not just a read) because some compilers
 * or prefetchers can short-circuit a pure read of unused data. The sum is
 * deliberately consumed (returned) so the compiler cannot prove the loop
 * is dead code and optimise it away entirely.
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
 * fill_vector() -- deterministic pseudo-random fill, no external deps.
 *
 * Values are kept in a modest range ([-1, 1)) so dot products of large
 * vectors don't approach overflow -- we're measuring speed here, Phase 3
 * already covers numerical edge cases.
 * ------------------------------------------------------------------- */
static void fill_vector(BLAS_REAL *v, blas_int n, unsigned int seed)
{
    unsigned int state = seed;
    for (blas_int i = 0; i < n; i++) {
        /* Simple xorshift -- fast, deterministic, good enough for
         * benchmark data (we are not testing RNG quality here). */
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        BLAS_REAL frac = (BLAS_REAL)(state % 1000000u) / (BLAS_REAL)1000000.0;
        v[i] = (BLAS_REAL)2.0 * frac - (BLAS_REAL)1.0;
    }
}

/* ---------------------------------------------------------------------
 * run_warm() -- median time for BENCH_TRIALS back-to-back calls on data
 * that is already resident (touched once, before timing starts).
 * ------------------------------------------------------------------- */
static double run_warm(const BLAS_REAL *x, const BLAS_REAL *y, blas_int n)
{
    double samples[BENCH_TRIALS];
    volatile BLAS_REAL sink = (BLAS_REAL)0.0; /* prevent dead-code elimination */

    for (int t = 0; t < BENCH_TRIALS; t++) {
        double t0 = bench_now();
        BLAS_REAL r = blas_dot(n, x, 1, y, 1);
        double t1 = bench_now();
        sink = r;
        samples[t] = t1 - t0;
    }

    (void)sink;
    return bench_median(samples, BENCH_TRIALS);
}

/* ---------------------------------------------------------------------
 * run_cold() -- median time for BENCH_TRIALS calls, each preceded by a
 * deliberate cache eviction so every call pays the full memory-fetch
 * cost for x and y.
 * ------------------------------------------------------------------- */
static double run_cold(const BLAS_REAL *x, const BLAS_REAL *y, blas_int n,
                        BLAS_REAL *evict_buf, blas_int evict_n)
{
    double samples[BENCH_TRIALS];
    volatile BLAS_REAL sink = (BLAS_REAL)0.0;

    for (int t = 0; t < BENCH_TRIALS; t++) {
        sink = evict_cache(evict_buf, evict_n); /* not timed -- eviction only */

        double t0 = bench_now();
        BLAS_REAL r = blas_dot(n, x, 1, y, 1);
        double t1 = bench_now();
        sink = r;
        samples[t] = t1 - t0;
    }

    (void)sink;
    return bench_median(samples, BENCH_TRIALS);
}

/* ---------------------------------------------------------------------
 * report_row() -- compute and print GFlop/s and GB/s for one size.
 *
 * blas_dot: 2 FLOPs/element (1 mul + 1 add), reads 2 elements/iteration,
 * writes none.
 * ------------------------------------------------------------------- */
static void report_row(blas_int n, double warm_t, double cold_t)
{
    double flops      = 2.0 * (double)n;
    double bytes      = 2.0 * (double)n * (double)sizeof(BLAS_REAL);

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
    /* Sweep sizes: 1K to 100M, roughly log-spaced with a few extra
     * intermediate points to make the cache-level transitions visible. */
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
    printf(" blas_dot benchmark\n");
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

    /* Allocate once at the largest size; reuse the prefix for smaller n.
     * This avoids num_sizes separate malloc/free cycles and lets warm-cache
     * trials at small n behave consistently (same memory layout each time). */
    BLAS_REAL *x = (BLAS_REAL *)malloc((size_t)max_n * sizeof(BLAS_REAL));
    BLAS_REAL *y = (BLAS_REAL *)malloc((size_t)max_n * sizeof(BLAS_REAL));
    BLAS_REAL *evict_buf =
        (BLAS_REAL *)malloc((size_t)EVICT_DOUBLES * sizeof(BLAS_REAL));

    if (!x || !y || !evict_buf) {
        fprintf(stderr, "bench_dot: allocation failure for n=%lld\n",
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

        /* Warm cache: touch (already done via fill_vector above for the
         * full prefix), then time repeated calls. */
        double warm_t = run_warm(x, y, n);

        /* Cold cache: evict before every trial. */
        double cold_t = run_cold(x, y, n, evict_buf, (blas_int)EVICT_DOUBLES);

        report_row(n, warm_t, cold_t);
    }

    printf("=================================================================="
           "===========\n");
    printf("Notes:\n");
    printf("  - 'warm' = data resident from a previous touch; repeated calls\n");
    printf("    on the same buffer (best case, cache-friendly).\n");
    printf("  - 'cold' = cache deliberately flushed before every call (worst\n");
    printf("    case, first-touch). Real applications usually sit between\n");
    printf("    these two numbers.\n");
    printf("  - blas_dot is %zu FLOPs and %zu bytes per element -- compare\n",
           (size_t)2, (size_t)(2 * sizeof(BLAS_REAL)));
    printf("    the GB/s columns against your machine's known DRAM bandwidth\n");
    printf("    to see how close to the memory-bandwidth roofline this is.\n");

    free(x);
    free(y);
    free(evict_buf);
    return 0;
}