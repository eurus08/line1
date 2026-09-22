/*
 * mpi_iamax.c — MPI-parallel index of maximum absolute value (step 6.6)
 *
 * See mpi_iamax.h for the full algorithm and why it needs both an
 * MPI_MAXLOC reduction AND a follow-up MPI_Bcast. Implementation
 * notes specific to this file:
 *
 * - The -1.0 sentinel for an empty local slice works because
 *   fabs(x[i]) is always >= 0.0 for any real x[i] -- so -1.0 can
 *   never accidentally win a MAXLOC against a rank that actually
 *   holds elements, including an all-zero slice (0.0 still beats
 *   -1.0). Only if EVERY rank is empty does global_pair.value stay
 *   at -1.0, which is the signal used below to return the "empty
 *   global vector" result without ever computing a division or an
 *   out-of-bounds array access.
 *
 * - blas_mpi_maxloc_pair_t's memory layout (from mpi_types.h) must
 *   match what MPI_DOUBLE_INT / MPI_FLOAT_INT expect on this
 *   platform. This is the standard, portable MAXLOC pattern used in
 *   MPI programs generally -- not something specific to this file.
 */

#include "line1/mpi/mpi_iamax.h"
#include "line1/mpi/mpi_types.h"
#include "line1/iamax.h"
#include "line1/types.h"
/* fabs() replaced by BLAS_FABS() from types.h (precision-generic) */

blas_mpi_iamax_result_t blas_mpi_iamax(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    blas_int            global_offset,
    MPI_Comm            comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    /* ------------------------------------------------------------------
     * Step 1 — local max abs value + local 1-based index
     * ------------------------------------------------------------------ */
    blas_int  local_index = blas_iamax(n_local, x, incx);
    BLAS_REAL local_value;

    if (local_index > 0) {
        local_value = BLAS_FABS(x[(local_index - 1) * incx]);
    } else {
        /* Empty local slice: guaranteed to lose any real MAXLOC
         * comparison (see file header comment). */
        local_value = (BLAS_REAL)-1.0;
    }

    /* ------------------------------------------------------------------
     * Step 2 — MAXLOC across ranks, location = RANK NUMBER, not an
     * element index. See mpi_iamax.h for why. [FIRST SYNC POINT]
     * ------------------------------------------------------------------ */
    blas_mpi_maxloc_pair_t local_pair;
    blas_mpi_maxloc_pair_t global_pair;

    local_pair.value = local_value;
    local_pair.loc   = rank;

    MPI_Allreduce(
        &local_pair,
        &global_pair,
        1,
        BLAS_MPI_MAXLOC_TYPE,
        MPI_MAXLOC,
        comm
    );

    blas_mpi_iamax_result_t result;

    /* Every rank was empty -- the global vector has no elements. */
    if (global_pair.value == (BLAS_REAL)-1.0) {
        result.value = (BLAS_REAL)0.0;
        result.index = 0;
        return result;
    }

    result.value = global_pair.value;

    /* ------------------------------------------------------------------
     * Step 3 — only the winning rank knows the true global index;
     * every other rank's local_index is irrelevant here.
     * Step 4 — broadcast it to everyone. [SECOND SYNC POINT]
     * ------------------------------------------------------------------ */
    int      winning_rank = global_pair.loc;
    blas_int global_index = 0;

    if (rank == winning_rank) {
        /* local_index is 1-based; global_offset is the 0-based start
         * of this rank's slice -- adding them directly gives the
         * correct 1-based GLOBAL index. */
        global_index = global_offset + local_index;
    }

    MPI_Bcast(&global_index, 1, BLAS_MPI_INT, winning_rank, comm);

    result.index = global_index;
    return result;
}
