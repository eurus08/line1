/*
 * tests/mpi/test_mpi_dot.c — MPI test suite for blas_mpi_dot
 *
 * Run with: mpirun -np <N> ./test_mpi_dot   (any N from 1 upward)
 *
 * Testing philosophy: same as the serial tests (tolerance-based
 * comparison, never ==), with one addition specific to MPI: only
 * rank 0 prints, since every rank computes the identical collective
 * result and duplicate output from every rank would just be noise.
 * Each rank still independently checks its own result and returns
 * non-zero on failure -- if any rank's check fails, mpirun's own
 * exit code goes non-zero and CTest reports the test as failed.
 *
 * Partitioning: N = 23 is used deliberately because it is prime --
 * it cannot divide evenly across any rank count except 1 and 23, so
 * every other -np value forces the UNEVEN partition path (some
 * ranks get one more element than others). This exercises the
 * partition() helper below the same way real production code would
 * be exercised, rather than only ever testing the easy evenly-
 * divisible case.
 */

#include "line1/mpi/mpi_dot.h"
#include "line1/dot.h"
#include "line1/types.h"

#include <mpi.h>
#include <math.h>
#include <stdio.h>

#define TOL_TIGHT 1e-12

static int g_tests  = 0;
static int g_failed = 0;
static int g_rank    = 0;   /* set in main, used by check_abs to gate printing */

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

/*
 * partition — split a global vector of length n_global across
 * `size` ranks as evenly as possible. Ranks 0..(n_global % size - 1)
 * get one extra element; every rank gets a contiguous, non-
 * overlapping slice, and the slices cover the whole vector exactly
 * once. This is the same scheme any caller of the MPI LINE1 layer
 * would need to implement -- these test files double as a worked
 * example of it.
 */
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
        printf("  test_mpi_dot — blas_mpi_dot  (running with %d rank%s)\n",
               size, size == 1 ? "" : "s");
        printf("==========================================================\n");
    }

    /* ---------------------------------------------------------------
     * Test 1 — known answer, uneven partition
     *
     * Global x[i] = i+1 (1-based values), global y[i] = 1.0 for all i.
     * dot(x, y) = sum_{i=1}^{N} i = N(N+1)/2
     * --------------------------------------------------------------- */
    {
        const blas_int N = 23;
        blas_int n_local, offset;
        partition(N, rank, size, &n_local, &offset);

        BLAS_REAL x[23], y[23];
        for (blas_int i = 0; i < n_local; i++) {
            x[i] = (BLAS_REAL)(offset + i + 1);
            y[i] = (BLAS_REAL)1.0;
        }

        BLAS_REAL result = blas_mpi_dot(n_local, x, 1, y, 1, MPI_COMM_WORLD);
        BLAS_REAL expected = (BLAS_REAL)N * ((BLAS_REAL)N + 1.0) / 2.0;

        check_abs("dot known-answer, uneven partition, N=23", result, expected, TOL_TIGHT);
    }

    /* ---------------------------------------------------------------
     * Test 2 — cross-check against the serial blas_dot on a
     * reconstructed full copy of the same vector. Confirms the MPI
     * result matches what the serial library itself would compute
     * on the whole vector in one process -- the real point of "does
     * the parallel version agree with serial".
     * --------------------------------------------------------------- */
    {
        const blas_int N = 23;
        blas_int n_local, offset;
        partition(N, rank, size, &n_local, &offset);

        BLAS_REAL x_local[23], y_local[23];
        for (blas_int i = 0; i < n_local; i++) {
            /* Deliberately different pattern from Test 1, mixed signs. */
            x_local[i] = (BLAS_REAL)((offset + i) % 5) - 2.0;
            y_local[i] = (BLAS_REAL)((offset + i) % 3) + 1.0;
        }

        BLAS_REAL mpi_result = blas_mpi_dot(n_local, x_local, 1, y_local, 1, MPI_COMM_WORLD);

        /* Every rank reconstructs the full global vector independently
         * and computes the serial answer locally -- N=23 is tiny, so
         * this redundant computation is cheap and needs no MPI_Gather. */
        BLAS_REAL x_full[23], y_full[23];
        for (blas_int i = 0; i < N; i++) {
            x_full[i] = (BLAS_REAL)(i % 5) - 2.0;
            y_full[i] = (BLAS_REAL)(i % 3) + 1.0;
        }
        BLAS_REAL serial_result = blas_dot(N, x_full, 1, y_full, 1);

        check_abs("dot MPI result matches serial on full vector", mpi_result, serial_result, TOL_TIGHT);
    }

    /* ---------------------------------------------------------------
     * Test 3 — n_local = 0 on rank 0 (only meaningful for size > 1;
     * for size == 1 this degenerates to an empty global vector,
     * which is also a valid case to check).
     * --------------------------------------------------------------- */
    {
        blas_int n_local = (rank == 0) ? 0 : 3;
        BLAS_REAL x[3] = {2.0, 2.0, 2.0};
        BLAS_REAL y[3] = {3.0, 3.0, 3.0};

        BLAS_REAL result = blas_mpi_dot(n_local, x, 1, y, 1, MPI_COMM_WORLD);
        BLAS_REAL expected = (BLAS_REAL)(size - 1) * 3.0 * (2.0 * 3.0);

        check_abs("dot rank-0 empty slice", result, expected, TOL_TIGHT);
    }

    if (rank == 0) {
        printf("\n==========================================================\n");
        printf("  Results: %d / %d tests passed\n", g_tests - g_failed, g_tests);
        printf("==========================================================\n");
    }

    MPI_Finalize();
    return (g_failed == 0) ? 0 : 1;
}
