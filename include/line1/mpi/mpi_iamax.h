/**
 * @file mpi_iamax.h
 * @ingroup mpi
 * @brief MPI-parallel index of the maximum absolute value.
 */

#ifndef LINE1_MPI_IAMAX_H
#define LINE1_MPI_IAMAX_H

#include <mpi.h>
#include "line1/types.h"

/**
 * @struct blas_mpi_iamax_result_t
 * @ingroup mpi
 * @brief Result of blas_mpi_iamax(): the global maximum absolute
 *        value and its global 1-based index.
 */
typedef struct {
    BLAS_REAL value; /**< max(|x_global[i]|) across the whole distributed vector. */
    blas_int  index; /**< 1-based *global* index of that element (BLAS
                           convention, spanning every rank's slice --
                           not just one rank's local index), or 0 if
                           the global vector is empty. */
} blas_mpi_iamax_result_t;

/**
 * @ingroup mpi
 * @brief Finds the global index of the largest-magnitude element in a
 *        vector distributed across an MPI communicator.
 *
 * @par Why this needs two MPI collectives, not one
 * The natural tool is @c MPI_MAXLOC: it reduces (value, location)
 * pairs and returns the pair with the maximum value, breaking ties by
 * the *smallest* location -- matching blas_iamax()'s own
 * "first occurrence wins" tie-break, as long as ranks hold
 * contiguous, ordered slices of the global vector.
 *
 * The catch: every predefined MAXLOC-compatible MPI pair type
 * (@c MPI_DOUBLE_INT, @c MPI_FLOAT_INT, ...) uses a plain 32-bit
 * @c int for the location field. ::blas_int, however, is deliberately
 * @c int64_t (see line1/types.h) specifically so this library can
 * address vectors past 2 billion elements. Packing a 64-bit global
 * index into MAXLOC's 32-bit location field would silently truncate
 * and return a *wrong* index for any global vector longer than
 * roughly 2.1 billion elements -- quietly defeating the entire reason
 * @c blas_int is 64-bit.
 *
 * The fix used here: use MAXLOC's location field to hold the *rank
 * number* instead of an element index (rank counts always fit
 * comfortably in 32 bits), then recover the true 64-bit global index
 * with one @c MPI_Bcast from the winning rank:
 *   -# Local @c blas_iamax() on this rank's slice, giving a local
 *      index and value (or a sentinel @f$ -1 @f$ if this rank's slice
 *      is empty -- guaranteed to lose to any real @f$ |x_i| \ge 0 @f$).
 *   -# @c MPI_Allreduce(..., @c MPI_MAXLOC, ...) on
 *      (local_value, this_rank) pairs -> (global_value, winning_rank).
 *      **First synchronisation point.**
 *   -# Only the winning rank computes
 *      @c global_index @c = @c global_offset @c + @c local_index.
 *   -# @c MPI_Bcast(&global_index, ..., root=winning_rank, ...) hands
 *      the true @c int64 global index to every rank. **Second
 *      synchronisation point.**
 *
 * @warning **Collective operation, twice over.** Every rank in
 *          @p comm must call this function, in the same relative
 *          order as every other rank.
 *
 * @param n_local       Number of elements in *this rank's* local slice.
 * @param x             This rank's local slice (read-only).
 * @param incx          Stride for @p x.
 * @param global_offset 0-based starting position of this rank's slice
 *                       within the conceptual global vector (e.g. if
 *                       ranks hold contiguous equal chunks of size
 *                       @c k, rank @c r's offset is @c r*k). The
 *                       caller's responsibility to compute correctly
 *                       -- this function cannot check it.
 * @param comm          The communicator to reduce and broadcast across.
 * @return A ::blas_mpi_iamax_result_t with the global max absolute
 *         value and its 1-based global index, identical on every
 *         rank. @c index is 0 if the global vector is empty (every
 *         rank had @f$ n\_local \le 0 @f$), matching blas_iamax()'s
 *         own @f$ n \le 0 @f$ convention.
 */
blas_mpi_iamax_result_t blas_mpi_iamax(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    blas_int            global_offset,
    MPI_Comm            comm
);

#endif /* LINE1_MPI_IAMAX_H */
