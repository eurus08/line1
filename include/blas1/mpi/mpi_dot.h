/*
 * mpi_dot.h — MPI-parallel dot product
 *
 * Each rank holds a contiguous LOCAL slice of the two vectors — this
 * function does not know or care about the global vector layout, and
 * it does not do any scattering. That is the caller's responsibility
 * (typically: divide N by the rank count once, up front, outside any
 * BLAS1 call).
 *
 * What this function does:
 *   1. Compute the LOCAL dot product of this rank's slice, using the
 *      existing serial blas_dot() — no reimplementation, no
 *      duplicated math.
 *   2. MPI_Allreduce(..., MPI_SUM, ...) to sum every rank's local
 *      result into a single global result, delivered to ALL ranks
 *      (that is what "Allreduce" means, vs. "Reduce" which delivers
 *      only to one root rank).
 *
 * Parameters:
 *   n_local - number of elements in THIS RANK's local slice
 *             (NOT the global vector length — see note above)
 *   x       - this rank's local slice of the first vector (read-only)
 *   incx    - stride for x
 *   y       - this rank's local slice of the second vector (read-only)
 *   incy    - stride for y
 *   comm    - the MPI communicator to reduce across (usually
 *             MPI_COMM_WORLD, but passed explicitly so this function
 *             works correctly inside a sub-communicator too)
 *
 * Returns: the GLOBAL dot product, identical on every rank in comm.
 *
 * Collective operation: every rank in `comm` MUST call this function,
 * or the program deadlocks waiting on MPI_Allreduce. This is true of
 * every function in the MPI layer that performs a collective.
 */

#ifndef BLAS1_MPI_DOT_H
#define BLAS1_MPI_DOT_H

#include <mpi.h>
#include "blas1/types.h"

BLAS_REAL blas_mpi_dot(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    const BLAS_REAL   * BLAS_RESTRICT y,
    blas_int            incy,
    MPI_Comm            comm
);

#endif /* BLAS1_MPI_DOT_H */
