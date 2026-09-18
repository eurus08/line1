/**
 * @file mpi_nrm2.h
 * @ingroup mpi
 * @brief MPI-parallel Euclidean (L2) norm.
 */

#ifndef BLAS1_MPI_NRM2_H
#define BLAS1_MPI_NRM2_H

#include <mpi.h>
#include "blas1/types.h"

/**
 * @ingroup mpi
 * @brief Computes the Euclidean norm of a vector distributed across
 *        an MPI communicator.
 *
 * @f[ \mathrm{result} = \sqrt{\sum_{\text{all ranks}} \sum_i x_i^2} @f]
 *
 * @par Why this needs two MPI collectives, not one
 * The naive approach -- local sum of squares, @c MPI_Allreduce, then
 * @c sqrt -- produces the right answer for ordinary vectors, but
 * silently reintroduces exactly the bug the serial blas_nrm2() was
 * written to avoid: an element around @f$ 10^{200} @f$ overflows to
 * @c Inf when squared, before MPI is ever involved. This function
 * instead performs the distributed version of blas_nrm2()'s scaled
 * two-pass algorithm:
 *   -# Local @c scale = max(|x_i|) on this rank's slice (reuses
 *      blas_iamax(), exactly as blas_nrm2() does).
 *   -# @c MPI_Allreduce(..., @c MPI_MAX, ...) -> global scale.
 *      **First synchronisation point.**
 *   -# Local sum of @f$ (x_i / \text{global\_scale})^2 @f$,
 *      Kahan-compensated. Every rank now divides by the *same*
 *      global scale, so every term across every rank is safely in
 *      @f$ [0, 1] @f$ -- squaring it can neither overflow nor
 *      underflow.
 *   -# @c MPI_Allreduce(..., @c MPI_SUM, ...) -> global sum of
 *      scaled squares. **Second synchronisation point.**
 *   -# Return @c global_scale @f$ \times \sqrt{\text{global\_sum}} @f$.
 *
 * @warning **Collective operation, twice over.** Every rank in
 *          @p comm must call this function, and in the same relative
 *          order as every other rank (both @c MPI_Allreduce calls are
 *          synchronisation points).
 *
 * @param n_local Number of elements in *this rank's* local slice.
 * @param x       This rank's local slice (read-only).
 * @param incx    Stride for @p x.
 * @param comm    The communicator to reduce across.
 * @return The global Euclidean norm, identical on every rank.
 */
BLAS_REAL blas_mpi_nrm2(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    MPI_Comm            comm
);

#endif /* BLAS1_MPI_NRM2_H */
