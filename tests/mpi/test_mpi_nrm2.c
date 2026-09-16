/*
 * tests/mpi/test_mpi_nrm2.c — MPI test suite for blas_mpi_nrm2
 *
 * Run with: mpirun -np <N> ./test_mpi_nrm2   (any N from 1 upward)
 *
 * See test_mpi_dot.c for general MPI test conventions (rank-0-only
 * printing, per-rank independent pass/fail, partition() helper).
 *
 * This file specifically tests the reason blas_mpi_nrm2 uses two
 * MPI_Allreduce calls instead of one (see mpi_nrm2.h) -- the
 * overflow/underflow scenarios that a naive "sum of squares, then
 * Allreduce, then sqrt" approach would get wrong.
 */

#include "blas1/mpi/mpi_nrm2.h"
#include "blas1/types.h"

#include <mpi.h>
#include <math.h>
#include <stdio.h>

#define TOL_LOOSE 1e-9

static int g_tests  = 0;
static int g_failed = 0;
static int g_rank    = 0;

static void check_rel(const char *name, BLAS_REAL result, BLAS_REAL expected, BLAS_REAL tol)
{
    g_tests++;
    BLAS_REAL scale = fmax(fabs(result), fabs(expected));
    BLAS_REAL err = (scale == 0.0) ? fabs(result - expected) : fabs(result - expected) / scale;
    int finite = isfinite((double)result);

    if (finite && err < tol) {
        if (g_rank == 0) {
            printf("  PASS  %-45s  rel_err=%.3e  tol=%.3e\n", name, err, tol);
        }
    } else {
        printf("  FAIL  %-45s  rel_err=%.3e  tol=%.3e  got=%.15g  expected=%.15g  finite=%d  (rank %d)\n",
               name, err, tol, (double)result, (double)expected, finite, g_rank);
        g_failed++;
    }
}

static void partition(blas_int n_global, int rank, int size,
                      blas_int *n_local, blas_int *offset)
{
    blas_int base = n_global / size;
    blas_int rem  = n_global % size;

    if (rank < rem) {
        *n_local = base + 1;
        *offset  = (blas_int)rank * (base + 1);
    } else {
        *n_local = base;
        *offset  = rem * (base + 1) + (blas_int)(rank - rem) * base;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_rank = rank;

    if (rank == 0) {
        printf("==========================================================\n");
        printf("  test_mpi_nrm2 — blas_mpi_nrm2  (running with %d rank%s)\n",
               size, size == 1 ? "" : "s");
        printf("==========================================================\n");
    }

    /* ---------------------------------------------------------------
     * Test 1 — known answer, uneven partition (N=23, prime).
     * Global x[i] = i+1. nrm2 = sqrt(sum_{i=1}^{N} i^2)
     *              = sqrt(N(N+1)(2N+1)/6)
     * --------------------------------------------------------------- */
    {
        const blas_int N = 23;
        blas_int n_local, offset;
        partition(N, rank, size, &n_local, &offset);

        BLAS_REAL x[23];
        for (blas_int i = 0; i < n_local; i++) {
            x[i] = (BLAS_REAL)(offset + i + 1);
        }

        BLAS_REAL result = blas_mpi_nrm2(n_local, x, 1, MPI_COMM_WORLD);

        double Nd = (double)N;
        double expected = sqrt(Nd * (Nd + 1.0) * (2.0 * Nd + 1.0) / 6.0);

        check_rel("nrm2 known-answer, uneven partition, N=23", result, (BLAS_REAL)expected, TOL_LOOSE);
    }

    /* ---------------------------------------------------------------
     * Test 2 — overflow-prone: one rank alone holds a 1e200 value,
     * everyone else holds zeros. A naive "sum of squares" approach
     * would compute (1e200)^2 = 1e400, which overflows double's
     * range and becomes Inf. The scaled two-collective algorithm
     * must still return exactly 1e200.
     * --------------------------------------------------------------- */
    {
        blas_int n_local = 2;
        BLAS_REAL x[2];
        if (rank == 0) {
            x[0] = 1.0e200; x[1] = 0.0;
        } else {
            x[0] = 0.0; x[1] = 0.0;
        }

        BLAS_REAL result = blas_mpi_nrm2(n_local, x, 1, MPI_COMM_WORLD);
        check_rel("nrm2 overflow-prone value 1e200", result, (BLAS_REAL)1.0e200, TOL_LOOSE);
    }

    /* ---------------------------------------------------------------
     * Test 3 — underflow-prone: a dominant 3-4-5 triangle on rank 0
     * plus tiny 1e-250 values scattered on other ranks. A naive
     * approach handles this one fine on its own (no overflow risk
     * from small values), but it's worth confirming the scaled
     * algorithm doesn't introduce a NEW problem -- e.g. the tiny
     * values getting divided by a large global_scale and underflowing
     * to exactly 0 is EXPECTED and fine, since their true
     * contribution to the sum is genuinely negligible at double
     * precision; what matters is the dominant contribution survives.
     * --------------------------------------------------------------- */
    {
        blas_int n_local = 2;
        BLAS_REAL x[2];
        if (rank == 0) {
            x[0] = 3.0; x[1] = 4.0;
        } else {
            x[0] = 1.0e-250; x[1] = 1.0e-250;
        }

        BLAS_REAL result = blas_mpi_nrm2(n_local, x, 1, MPI_COMM_WORLD);
        check_rel("nrm2 underflow-prone values alongside dominant 3-4-5", result, (BLAS_REAL)5.0, 1e-6);
    }

    /* ---------------------------------------------------------------
     * Test 4 — all-zero global vector must return exactly 0.0, not
     * NaN from a 0/0 division in the scaling step.
     * --------------------------------------------------------------- */
    {
        blas_int n_local = 3;
        BLAS_REAL x[3] = {0.0, 0.0, 0.0};

        BLAS_REAL result = blas_mpi_nrm2(n_local, x, 1, MPI_COMM_WORLD);
        check_rel("nrm2 all-zero global vector == 0", result, (BLAS_REAL)0.0, TOL_LOOSE);
    }

    if (rank == 0) {
        printf("\n==========================================================\n");
        printf("  Results: %d / %d tests passed\n", g_tests - g_failed, g_tests);
        printf("==========================================================\n");
    }

    MPI_Finalize();
    return (g_failed == 0) ? 0 : 1;
}
