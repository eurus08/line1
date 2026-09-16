/*
 * tests/mpi/test_mpi_axpy.c — MPI test suite for blas_mpi_axpy
 *
 * Run with: mpirun -np <N> ./test_mpi_axpy   (any N from 1 upward)
 *
 * See test_mpi_dot.c for the general MPI test conventions used here
 * (rank-0-only printing, per-rank independent pass/fail, partition()
 * helper). This file additionally exercises the property that makes
 * axpy different from the other three MPI functions: it needs no
 * communication at all, so per-rank local slice sizes don't even
 * need to agree with any partition scheme -- each rank can use
 * whatever n_local it likes independently.
 */

#include "blas1/mpi/mpi_axpy.h"
#include "blas1/axpy.h"
#include "blas1/types.h"

#include <mpi.h>
#include <math.h>
#include <stdio.h>

#define TOL_TIGHT 1e-12

static int g_tests  = 0;
static int g_failed = 0;
static int g_rank    = 0;

static void check_abs(const char *name, BLAS_REAL result, BLAS_REAL expected, BLAS_REAL tol)
{
    g_tests++;
    BLAS_REAL err = fabs(result - expected);
    if (err < tol) {
        if (g_rank == 0) {
            printf("  PASS  %-45s  err=%.3e  tol=%.3e\n", name, err, tol);
        }
    } else {
        printf("  FAIL  %-45s  err=%.3e  tol=%.3e  got=%.15g  expected=%.15g  (rank %d)\n",
               name, err, tol, (double)result, (double)expected, g_rank);
        g_failed++;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_rank = rank;
    (void)size; /* not needed for axpy's own logic, only for the banner */

    if (rank == 0) {
        printf("==========================================================\n");
        printf("  test_mpi_axpy — blas_mpi_axpy  (running with %d rank%s)\n",
               size, size == 1 ? "" : "s");
        printf("==========================================================\n");
    }

    /* ---------------------------------------------------------------
     * Test 1 — each rank uses a DIFFERENT local size (5 + rank),
     * proving no cross-rank agreement on n_local is required. Every
     * rank checks its own slice independently against a hand-
     * computed expected value.
     * --------------------------------------------------------------- */
    {
        blas_int n_local = 5 + (blas_int)rank;
        BLAS_REAL alpha = 2.0;
        BLAS_REAL x[16], y[16];

        for (blas_int i = 0; i < n_local; i++) {
            x[i] = (BLAS_REAL)(i + 1);
            y[i] = (BLAS_REAL)(rank + 10);
        }

        blas_mpi_axpy(n_local, alpha, x, 1, y, 1, MPI_COMM_WORLD);

        BLAS_REAL max_err = 0.0;
        for (blas_int i = 0; i < n_local; i++) {
            BLAS_REAL expected = alpha * (BLAS_REAL)(i + 1) + (BLAS_REAL)(rank + 10);
            BLAS_REAL err = fabs(y[i] - expected);
            if (err > max_err) max_err = err;
        }

        check_abs("axpy differing per-rank n_local, max error", max_err, 0.0, TOL_TIGHT);
    }

    /* ---------------------------------------------------------------
     * Test 2 — cross-check against serial blas_axpy on the exact
     * same local data (this comparison is trivially exact for axpy
     * since there's no cross-rank interaction to get wrong, but it's
     * still worth confirming the wrapper doesn't introduce a bug of
     * its own, e.g. an off-by-one in how it forwards arguments).
     * --------------------------------------------------------------- */
    {
        blas_int n_local = 7;
        BLAS_REAL alpha = -1.5;
        BLAS_REAL x[7], y_mpi[7], y_serial[7];

        for (blas_int i = 0; i < n_local; i++) {
            x[i]        = (BLAS_REAL)(i % 4) - 1.5;
            y_mpi[i]    = (BLAS_REAL)(i * 2);
            y_serial[i] = y_mpi[i];
        }

        blas_mpi_axpy(n_local, alpha, x, 1, y_mpi, 1, MPI_COMM_WORLD);
        blas_axpy(n_local, alpha, x, 1, y_serial, 1);

        BLAS_REAL max_err = 0.0;
        for (blas_int i = 0; i < n_local; i++) {
            BLAS_REAL err = fabs(y_mpi[i] - y_serial[i]);
            if (err > max_err) max_err = err;
        }

        check_abs("axpy MPI wrapper matches serial blas_axpy exactly", max_err, 0.0, TOL_TIGHT);
    }

    /* ---------------------------------------------------------------
     * Test 3 — alpha = 0 edge case (guarded specially in serial
     * scal.c to avoid NaN propagation; axpy has its own alpha==0
     * guard documented in axpy.c -- confirm it survives the MPI
     * wrapper unchanged).
     * --------------------------------------------------------------- */
    {
        blas_int n_local = 4;
        BLAS_REAL alpha = 0.0;
        BLAS_REAL x[4] = {1.0, 2.0, 3.0, 4.0};
        BLAS_REAL y[4] = {10.0, 20.0, 30.0, 40.0};

        blas_mpi_axpy(n_local, alpha, x, 1, y, 1, MPI_COMM_WORLD);

        BLAS_REAL max_err = 0.0;
        BLAS_REAL expected[4] = {10.0, 20.0, 30.0, 40.0};
        for (blas_int i = 0; i < n_local; i++) {
            BLAS_REAL err = fabs(y[i] - expected[i]);
            if (err > max_err) max_err = err;
        }

        check_abs("axpy alpha=0 leaves y unchanged", max_err, 0.0, TOL_TIGHT);
    }

    if (rank == 0) {
        printf("\n==========================================================\n");
        printf("  Results: %d / %d tests passed\n", g_tests - g_failed, g_tests);
        printf("==========================================================\n");
    }

    MPI_Finalize();
    return (g_failed == 0) ? 0 : 1;
}
