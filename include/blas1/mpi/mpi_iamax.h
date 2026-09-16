/*
 * mpi_iamax.h — MPI-parallel index of maximum absolute value
 *
 * Returns the GLOBAL 1-based index (BLAS convention) of the element
 * with the largest absolute value, across the WHOLE distributed
 * vector -- not just one rank's slice -- together with that value.
 *
 * =========================================================================
 * WHY THIS NEEDS TWO MPI COLLECTIVES, NOT ONE
 * =========================================================================
 *
 * The natural tool here is MPI_MAXLOC: it reduces (value, location)
 * pairs and returns the pair with the maximum value, breaking ties by
 * the SMALLEST location (matching blas_iamax's own "first occurrence
 * wins" tie-break, as long as ranks hold contiguous, ordered slices
 * of the global vector).
 *
 * The catch: every one of MPI's predefined MAXLOC-compatible pair
 * datatypes (MPI_DOUBLE_INT, MPI_FLOAT_INT, ...) uses a plain 32-bit
 * `int` for the location field. There is no built-in pair type with a
 * 64-bit location. blas_int, however, is deliberately int64_t (see
 * blas1/types.h) specifically so this library can address vectors
 * past 2 billion elements. Packing a 64-bit global index into
 * MAXLOC's 32-bit location field would silently truncate and return
 * a WRONG index for any global vector longer than ~2.1 billion
 * elements -- quietly defeating the entire reason blas_int is 64-bit.
 *
 * The fix used here: use MAXLOC's location field to hold the RANK
 * NUMBER instead of an element index (rank counts always fit
 * comfortably in 32 bits, even on the largest real clusters), then
 * recover the true 64-bit global index with one MPI_Bcast FROM the
 * winning rank:
 *
 *   Step 1: local_index = blas_iamax(n_local, x, incx) on this rank's
 *           slice (0 if n_local <= 0 -- see blas_iamax's own sentinel
 *           convention). local_value = |x[local_index-1]|, or -1.0
 *           if this rank's slice was empty (a value guaranteed to
 *           lose to any real |x[i]|, which is always >= 0 -- this
 *           stops an empty rank from ever "winning" the reduction).
 *   Step 2: MPI_Allreduce(..., MPI_MAXLOC, ...) on (local_value,
 *           this_rank) pairs -> (global_value, winning_rank).
 *           [FIRST SYNC POINT]
 *   Step 3: the winning rank (and ONLY the winning rank) computes
 *           global_index = global_offset + local_index -- its own
 *           local_index (1-based) plus the 0-based starting position
 *           of its slice in the global vector, which the caller
 *           supplies (this MPI layer does not track vector layout
 *           itself, matching every other function here).
 *   Step 4: MPI_Bcast(&global_index, ..., root=winning_rank, ...)
 *           hands the true int64 global index to every rank.
 *           [SECOND SYNC POINT]
 *
 * Parameters:
 *   n_local       - number of elements in THIS RANK's local slice
 *   x             - this rank's local slice (read-only)
 *   incx          - stride for x
 *   global_offset - 0-based starting position of this rank's slice
 *                   within the conceptual GLOBAL vector (e.g. if
 *                   ranks hold contiguous equal chunks of size k,
 *                   rank r's global_offset is r*k). Caller's
 *                   responsibility to compute correctly -- this
 *                   function has no way to check it.
 *   comm          - communicator to reduce and broadcast across
 *
 * Returns: a blas_mpi_iamax_result_t with the global max absolute
 * value and its 1-based global index, identical on every rank.
 * index is 0 if the global vector is empty (every rank had
 * n_local <= 0), matching blas_iamax's own n <= 0 convention.
 *
 * Collective operation, TWICE over: every rank in comm must call
 * this function, in the same relative order as every other rank.
 */

#ifndef BLAS1_MPI_IAMAX_H
#define BLAS1_MPI_IAMAX_H

#include <mpi.h>
#include "blas1/types.h"

typedef struct {
    BLAS_REAL value;  /* max(|x_global[i]|) across the whole vector */
    blas_int  index;  /* 1-based GLOBAL index of that element, or
                          0 if the global vector is empty */
} blas_mpi_iamax_result_t;

blas_mpi_iamax_result_t blas_mpi_iamax(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    blas_int            global_offset,
    MPI_Comm            comm
);

#endif /* BLAS1_MPI_IAMAX_H */
