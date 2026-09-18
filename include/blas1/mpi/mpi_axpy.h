/**
 * @file mpi_axpy.h
 * @ingroup mpi
 * @brief MPI-parallel axpy (y = alpha*x + y).
 */

#ifndef BLAS1_MPI_AXPY_H
#define BLAS1_MPI_AXPY_H

#include <mpi.h>
#include "blas1/types.h"

/**
 * @ingroup mpi
 * @brief Computes @f$ y \leftarrow \alpha x + y @f$ on each rank's
 *        local slice.
 *
 * Unlike blas_mpi_dot(), this operation needs **no communication at
 * all**. @c axpy computes @f$ y_i \mathrel{+}= \alpha x_i @f$ one
 * index at a time -- rank 0 never needs to know anything about rank
 * 1's slice of the vectors, and vice versa. This is what
 * "embarrassingly parallel" means: the work splits across ranks with
 * zero synchronisation required.
 *
 * @note @p comm is part of the signature purely for API consistency
 *       with the rest of the MPI layer (blas_mpi_dot(),
 *       blas_mpi_nrm2(), and blas_mpi_iamax() all take a @c comm
 *       because they genuinely need one). A caller that treats "the
 *       MPI BLAS1 layer" as one uniform interface doesn't have to
 *       special-case @c axpy just because it happens not to
 *       communicate. The implementation ignores this parameter
 *       entirely.
 *
 * @note **Not a collective operation** -- unlike blas_mpi_dot(),
 *       ranks do not need to call this in lockstep, and it will never
 *       deadlock.
 *
 * @param n_local Number of elements in *this rank's* local slice.
 * @param alpha   Scalar multiplier (must be the same value on every
 *                rank -- the caller's responsibility to ensure this;
 *                this function does not broadcast it).
 * @param x       This rank's local slice of the input vector (read-only).
 * @param incx    Stride for @p x.
 * @param y       This rank's local slice of the input/output vector,
 *                modified in place.
 * @param incy    Stride for @p y.
 * @param comm    Accepted for signature uniformity with the rest of
 *                the MPI layer; not used.
 */
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
