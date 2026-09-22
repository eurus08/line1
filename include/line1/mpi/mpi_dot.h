/**
 * @file mpi_dot.h
 * @ingroup mpi
 * @brief MPI-parallel dot product.
 */

#ifndef LINE1_MPI_DOT_H
#define LINE1_MPI_DOT_H

#include <mpi.h>
#include "line1/types.h"

/**
 * @ingroup mpi
 * @brief Computes the dot product of two vectors distributed across
 *        an MPI communicator.
 *
 * Each rank holds a contiguous local slice of the two vectors -- this
 * function does not know or care about the global vector layout, and
 * does no scattering. Partitioning the global vector (typically:
 * dividing @c N by the rank count once, up front) is the caller's
 * responsibility.
 *
 * Algorithm:
 *   -# Compute the local dot product of this rank's slice, using the
 *      existing serial blas_dot() -- no reimplementation.
 *   -# @c MPI_Allreduce(..., @c MPI_SUM, ...) sums every rank's local
 *      result into a single global result, delivered to *every* rank
 *      (as opposed to @c MPI_Reduce, which delivers only to one root).
 *
 * @warning **Collective operation.** Every rank in @p comm must call
 *          this function, or the program deadlocks waiting on
 *          @c MPI_Allreduce.
 *
 * @param n_local Number of elements in *this rank's* local slice
 *                (not the global vector length).
 * @param x       This rank's local slice of the first vector (read-only).
 * @param incx    Stride for @p x.
 * @param y       This rank's local slice of the second vector (read-only).
 * @param incy    Stride for @p y.
 * @param comm    The MPI communicator to reduce across (usually
 *                @c MPI_COMM_WORLD, but passed explicitly so this
 *                function also works correctly inside a
 *                sub-communicator).
 * @return The global dot product, identical on every rank in @p comm.
 */
BLAS_REAL blas_mpi_dot(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    const BLAS_REAL   * BLAS_RESTRICT y,
    blas_int            incy,
    MPI_Comm            comm
);

#endif /* LINE1_MPI_DOT_H */
