/*
 * mpi_axpy.c — MPI-parallel axpy (step 6.4)
 *
 * See mpi_axpy.h for the full contract. This is the simplest
 * function in the whole MPI layer: it calls the serial blas_axpy()
 * on the local slice and returns. No MPI_* call is made anywhere in
 * this file.
 *
 * comm is intentionally unused -- BLAS_UNUSED(comm) documents that
 * this is deliberate (matches -Wextra's -Wunused-parameter without
 * silently dropping the parameter from the signature, which would
 * break the uniform call pattern described in mpi_axpy.h).
 */

#include "blas1/mpi/mpi_axpy.h"
#include "blas1/axpy.h"

void blas_mpi_axpy(
    blas_int            n_local,
    BLAS_REAL           alpha,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    BLAS_REAL         * BLAS_RESTRICT y,
    blas_int            incy,
    MPI_Comm            comm)
{
    BLAS_UNUSED(comm);

    blas_axpy(n_local, alpha, x, incx, y, incy);
}
