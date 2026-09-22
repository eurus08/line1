/*
 * mpi_dot.c — MPI-parallel dot product (step 6.3)
 *
 * See mpi_dot.h for the full contract. Implementation notes:
 *
 * - blas_dot() (not blas_dot_kahan()) is used for the local reduction.
 *   This is a deliberate choice, not an oversight: Kahan compensation
 *   only protects against rounding error WITHIN one rank's partial
 *   sum. The cross-rank combination step (MPI_Allreduce with MPI_SUM)
 *   re-introduces exactly the same kind of floating-point reordering
 *   error Kahan is designed to fight, and MPI_Allreduce has no
 *   compensated-summation variant. So a Kahan-summed local result
 *   still loses precision at the reduction step — the accuracy gain
 *   from local Kahan summation is real but partial, and not worth the
 *   extra complexity in this first MPI pass. Revisit if a future
 *   phase adds a custom compensated MPI reduction.
 *
 * - MPI_Allreduce combines with MPI_SUM, which is the direct parallel
 *   analogue of the sequential summation blas_dot() does internally
 *   — we are just changing WHO does the adding, not HOW.
 */

#include "line1/mpi/mpi_dot.h"
#include "line1/mpi/mpi_types.h"
#include "line1/dot.h"

BLAS_REAL blas_mpi_dot(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    const BLAS_REAL   * BLAS_RESTRICT y,
    blas_int            incy,
    MPI_Comm            comm)
{
    /* Step 1: local dot product, this rank's slice only. */
    BLAS_REAL local_result = blas_dot(n_local, x, incx, y, incy);

    /* Step 2: sum every rank's local_result into global_result,
     * delivered to every rank in comm. This is the one and only
     * synchronisation point in the whole function. */
    BLAS_REAL global_result;
    MPI_Allreduce(
        &local_result,
        &global_result,
        1,
        BLAS_MPI_REAL,
        MPI_SUM,
        comm
    );

    return global_result;
}
