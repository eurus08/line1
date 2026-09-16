/*
 * tests/mpi/test_mpi_iamax.c — MPI test suite for blas_mpi_iamax
 *
 * Run with: mpirun -np <N> ./test_mpi_iamax   (any N from 1 upward)
 *
 * See test_mpi_dot.c for general MPI test conventions (rank-0-only
 * printing, per-rank independent pass/fail, partition() helper).
 *
 * These tests target the two things that make blas_mpi_iamax the
 * trickiest function in the MPI layer (see mpi_iamax.h): getting the
 * TRUE 64-bit global index right (not truncated through MAXLOC's
 * 32-bit location field), and correct tie-breaking / empty-slice
 * handling across ranks.
 */

#include "blas1/mpi/mpi_iamax.h"
#include "blas1/types.h"

#include <mpi.h>
#include <math.h>
#include <stdio.h>

static int g_tests  = 0;
static int g_failed = 0;
static int g_rank    = 0;

static void check_result(const char *name,
                         blas_mpi_iamax_result_t got,
                         BLAS_REAL expected_value,
                         blas_int expected_index)
{
    g_tests++;
    int value_ok = (fabs((double)got.value - (double)expected_value) < 1e-9);
    int index_ok = (got.index == expected_index);

    if (value_ok && index_ok) {
        if (g_rank == 0) {
            printf("  PASS  %-45s  value=%.6g  index=%lld\n",
                   name, (double)got.value, (long long)got.index);
        }
    } else {
        printf("  FAIL  %-45s  got value=%.6g index=%lld  expected value=%.6g index=%lld  (rank %d)\n",
               name, (double)got.value, (long long)got.index,
               (double)expected_value, (long long)expected_index, g_rank);
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
        printf("  test_mpi_iamax — blas_mpi_iamax  (running with %d rank%s)\n",
               size, size == 1 ? "" : "s");
        printf("==========================================================\n");
    }

    /* ---------------------------------------------------------------
     * Test 1 — known answer, uneven partition. A negative spike
     * (planted at global index 3) confirms comparison is by absolute
     * value, not signed value.
     * --------------------------------------------------------------- */
    {
        const blas_int N = 23;
        blas_int n_local, offset;
        partition(N, rank, size, &n_local, &offset);

        BLAS_REAL x[23];
        for (blas_int i = 0; i < n_local; i++) {
            x[i] = (BLAS_REAL)(offset + i + 1);
        }
        /* Global index 3 (1-based) means 0-based global position 2. */
        for (blas_int i = 0; i < n_local; i++) {
            if (offset + i == 2) {
                x[i] = -1000.0;
            }
        }

        blas_mpi_iamax_result_t r = blas_mpi_iamax(n_local, x, 1, offset, MPI_COMM_WORLD);
        check_result("iamax known-answer, negative spike, uneven partition", r, 1000.0, 3);
    }

    /* ---------------------------------------------------------------
     * Test 2 — cross-rank tie (only meaningful for size >= 2): rank 0
     * and rank 1 both hold the exact same max abs value. Expect the
     * LOWER rank (earlier in the global vector) to win, matching
     * blas_iamax's own first-occurrence tie-break.
     * --------------------------------------------------------------- */
    if (size >= 2) {
        blas_int n_local = 3;
        BLAS_REAL x[3] = {1.0, 1.0, 1.0};
        if (rank == 0 || rank == 1) {
            x[1] = 50.0;
        }
        blas_int offset = (blas_int)rank * n_local;

        blas_mpi_iamax_result_t r = blas_mpi_iamax(n_local, x, 1, offset, MPI_COMM_WORLD);
        /* Rank 0's copy is at global index 2 (1-based). */
        check_result("iamax cross-rank tie, lower rank wins", r, 50.0, 2);
    }

    /* ---------------------------------------------------------------
     * Test 3 — rank 0 has an empty local slice (n_local = 0); must
     * not win the reduction and must not crash. For size == 1 this
     * degenerates to a fully empty global vector.
     * --------------------------------------------------------------- */
    {
        blas_int n_local = (rank == 0) ? 0 : 3;
        BLAS_REAL x[3] = {2.0, 7.0, 3.0};
        blas_int offset = (rank == 0) ? 0 : 3 * (blas_int)(rank - 1);

        blas_mpi_iamax_result_t r = blas_mpi_iamax(n_local, x, 1, offset, MPI_COMM_WORLD);

        if (size == 1) {
            check_result("iamax rank-0 empty, size=1 -> whole vector empty", r, 0.0, 0);
        } else {
            check_result("iamax rank-0 empty, size>1 -> max found on rank 1", r, 7.0, 2);
        }
    }

    /* ---------------------------------------------------------------
     * Test 4 — every rank empty: fully degenerate all-empty global
     * vector must return the sentinel (value=0, index=0), matching
     * serial blas_iamax's own n<=0 convention.
     * --------------------------------------------------------------- */
    {
        blas_int n_local = 0;
        BLAS_REAL dummy = 0.0;

        blas_mpi_iamax_result_t r = blas_mpi_iamax(n_local, &dummy, 1, 0, MPI_COMM_WORLD);
        check_result("iamax all ranks empty -> sentinel (0, 0)", r, 0.0, 0);
    }

    if (rank == 0) {
        printf("\n==========================================================\n");
        printf("  Results: %d / %d tests passed\n", g_tests - g_failed, g_tests);
        printf("==========================================================\n");
    }

    MPI_Finalize();
    return (g_failed == 0) ? 0 : 1;
}
