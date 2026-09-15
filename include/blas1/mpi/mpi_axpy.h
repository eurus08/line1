/*
 * mpi_axpy.h — MPI-parallel axpy (y = alpha*x + y)
 *
 * Unlike dot, this operation needs NO communication at all. axpy
 * computes y[i] += alpha * x[i] one index at a time -- rank 0 never
 * needs to know anything about rank 1's slice of the vectors, and
 * vice versa. This is what "embarrassingly parallel" means: the work
 * splits across ranks with zero synchronisation required.
 *
 * comm is still part of the signature, but purely for API
 * consistency with the rest of the MPI layer (blas_mpi_dot,
 * blas_mpi_nrm2, blas_mpi_iamax all take a comm because they
 * genuinely need one). A caller that treats "the MPI BLAS1 layer" as
 * one uniform interface doesn't have to special-case axpy just
 * because it happens not to communicate. The implementation ignores
 * the parameter entirely -- see mpi_axpy.c.
 *
 * Parameters:
 *   n_local - number of elements in THIS RANK's local slice
 *   alpha   - scalar multiplier (same value on every rank -- the
 *             caller's responsibility to ensure that; this function
 *             does not broadcast it)
 *   x       - this rank's local slice of the input vector (read-only)
 *   incx    - stride for x
 *   y       - this rank's local slice of the input/output vector,
 *             modified in place
 *   incy    - stride for y
 *   comm    - accepted for signature uniformity, not used
 *
 * Not a collective operation: unlike blas_mpi_dot, ranks do not need
 * to call this in lockstep, and it will never deadlock.
 */

#ifndef BLAS1_MPI_AXPY_H
#define BLAS1_MPI_AXPY_H

#include <mpi.h>
#include "blas1/types.h"

void blas_mpi_axpy(
    blas_int            n_local,
    BLAS_REAL           alpha,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    BLAS_REAL         * BLAS_RESTRICT y,
    blas_int            incy,
    MPI_Comm            comm
);

#endif /* BLAS1_MPI_AXPY_H */
