/*
 * bench/mpi/bench_mpi_dot.c — MPI scaling benchmark for blas_mpi_dot
 *
 * Step 6.8 of the build plan.
 *
 * Usage:  mpirun -np <ranks> ./bench_mpi_dot <n_local>
 *
 * This program measures exactly ONE data point per invocation: given
 * a per-rank local vector length (n_local, identical on every rank
 * for simplicity -- real scaling studies almost always use equal
 * chunks), it runs BENCH_TRIALS_MPI timed calls to blas_mpi_dot() and
 * reports the median time, GFlop/s, and GB/s for the resulting global
 * vector of length n_local * ranks.
 *
 * WHY ONE DATA POINT PER RUN, NOT A SWEEP LIKE bench_dot.c
 * -----------------------------------------------------------------
 * bench_dot.c (Phase 4) can sweep many vector sizes inside one
 * process because process count never changes mid-run. Rank count
 * is fixed for the LIFETIME of an MPI job -- you cannot add or
 * remove ranks from inside a running program. So a "sweep across
 * rank counts" fundamentally has to be multiple separate `mpirun`
 * invocations, not multiple loop iterations inside one. This program
 * is deliberately kept simple (one size in, one row out); the actual
 * scaling STUDY -- running this at -np 1, 2, 4, 8, ... and
 * assembling the results into a table -- is what run_scaling.sh
 * does, by driving this binary repeatedly. That split (simple
 * single-point tool + a script that drives it across configurations)
 * is the standard way HPC scaling studies are structured.
 *
 * STRONG vs. WEAK SCALING -- decided entirely by the CALLER, not by
 * any flag in this program
 * -----------------------------------------------------------------
 * This program only ever sees ONE number, n_local, and has no idea
 * whether it's being used for a strong-scaling or weak-scaling study.
 * That's intentional:
 *   STRONG scaling: fix a GLOBAL size N, and as rank count P grows,
 *                   call this program with n_local = N / P (shrinking
 *                   per-rank work). Time should DECREASE as P grows.
 *   WEAK scaling:   fix n_local itself, and call this program with
 *                   the SAME n_local at every rank count P (so global
 *                   size N = n_local * P grows with P). Time should
 *                   stay roughly CONSTANT as P grows -- any increase
 *                   reveals communication overhead, since local work
 *                   per rank never changes.
 * See run_scaling.sh for both studies driven automatically.
 *
 * MEASUREMENT METHODOLOGY
 * -----------------------------------------------------------------
 * blas_mpi_dot is a COLLECTIVE operation -- every rank must call it,
 * and the operation doesn't complete anywhere until every rank has
 * arrived at the MPI_Allreduce inside it. That means the wall-clock
 * time that matters is the SLOWEST rank's time, not rank 0's time in
 * isolation (a fast rank finishing early still has to wait for a
 * straggler). So each rank:
 *   1. MPI_Barrier()s to line up the starting gun as closely as MPI
 *      allows (not perfect, but close -- true sub-microsecond global
 *      synchronisation isn't achievable at all on a real cluster)
 *   2. Times its OWN BENCH_TRIALS_MPI calls, takes the LOCAL median
 *   3. MPI_Reduce(..., MPI_MAX, root=0, ...) finds the slowest rank's
 *      local median
 * Only rank 0 then computes and prints GFlop/s / GB/s, using that
 * max time and the GLOBAL vector size (n_local * ranks) -- since the
 * "work done" for throughput purposes is the whole distributed
 * computation, not just one rank's slice of it.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "line1/mpi/mpi_dot.h"
#include "line1/types.h"
#include "timing.h"

#define BENCH_TRIALS_MPI 30   /* fewer than bench_dot.c's 100 -- each
                                 trial here includes a real MPI
                                 collective, which is far more
                                 expensive per-call than a serial
                                 function, so fewer trials still give
                                 a stable median in reasonable time. */

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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "usage: %s <n_local>\n", argv[0]);
            fprintf(stderr, "  n_local: number of elements per rank "
                            "(identical on every rank)\n");
        }
        MPI_Finalize();
        return 1;
    }

    blas_int n_local = (blas_int)atoll(argv[1]);
    if (n_local <= 0) {
        if (rank == 0) {
            fprintf(stderr, "n_local must be positive, got %lld\n",
                    (long long)n_local);
        }
        MPI_Finalize();
        return 1;
    }

    BLAS_REAL *x = (BLAS_REAL *)malloc((size_t)n_local * sizeof(BLAS_REAL));
    BLAS_REAL *y = (BLAS_REAL *)malloc((size_t)n_local * sizeof(BLAS_REAL));
    if (!x || !y) {
        fprintf(stderr, "rank %d: allocation failure for n_local=%lld\n",
                rank, (long long)n_local);
        free(x);
        free(y);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    /* Different seed per rank so each rank's slice is genuinely
     * different data, not a suspiciously identical copy. */
    fill_vector(x, n_local, 12345u + (unsigned int)rank);
    fill_vector(y, n_local, 67890u + (unsigned int)rank);

    double samples[BENCH_TRIALS_MPI];
    volatile BLAS_REAL sink = (BLAS_REAL)0.0;

    for (int t = 0; t < BENCH_TRIALS_MPI; t++) {
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = bench_now();
        BLAS_REAL r = blas_mpi_dot(n_local, x, 1, y, 1, MPI_COMM_WORLD);
        double t1 = bench_now();

        sink = r;
        samples[t] = t1 - t0;
    }
    (void)sink;

    double local_median = bench_median(samples, BENCH_TRIALS_MPI);

    double global_max_time;
    MPI_Reduce(&local_median, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        blas_int n_global = n_local * (blas_int)size;

        /* Same per-element cost model as bench_dot.c: 2 FLOPs/element
         * (1 mul + 1 add), 2 * sizeof(BLAS_REAL) bytes/element read,
         * applied to the GLOBAL vector since that's the work the
         * whole distributed computation performs together. */
        double flops = 2.0 * (double)n_global;
        double bytes = 2.0 * (double)n_global * (double)sizeof(BLAS_REAL);

        double gflops = (global_max_time > 0.0) ? (flops / global_max_time) / 1e9 : 0.0;
        double gbs    = (global_max_time > 0.0) ? (bytes / global_max_time) / 1e9 : 0.0;

        /* One machine-readable row: ranks, n_local, n_global, time_us,
         * GFlop/s, GB/s -- easy for run_scaling.sh to parse with awk,
         * and also perfectly readable run standalone. */
        printf("%8d %14lld %14lld %14.3f %12.4f %12.4f\n",
               size, (long long)n_local, (long long)n_global,
               global_max_time * 1e6, gflops, gbs);
    }

    free(x);
    free(y);
    MPI_Finalize();
    return 0;
}
