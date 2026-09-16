/*
 * mpi_nrm2.h — MPI-parallel Euclidean norm (L2 norm)
 *
 * Computes: result = sqrt( sum over ALL ranks( x[i]^2 ) )
 *
 * =========================================================================
 * WHY THIS IS NOT "LOCAL SUM OF SQUARES, THEN ALLREDUCE, THEN SQRT"
 * =========================================================================
 *
 * That simpler version is tempting -- it's what a first pass at the
 * Phase 6 plan describes, and it produces the right answer for
 * ordinary vectors. But it silently reintroduces exactly the bug
 * blas_nrm2() (see nrm2.c) was written to avoid: if any element is
 * around 1e200, squaring it overflows to Inf before MPI ever gets
 * involved. If any element is around 1e-200, squaring it underflows
 * to 0.0 and vanishes from the sum. The serial library goes to real
 * trouble to prevent this (the Blue 1978 / LAPACK dnrm2 scaling
 * trick) -- an MPI wrapper that skips that trouble is not really
 * offering the same function in parallel, it's offering a
 * numerically weaker one that happens to agree on "nice" inputs.
 *
 * =========================================================================
 * THE ALGORITHM USED HERE -- DISTRIBUTED VERSION OF THE SAME TRICK
 * =========================================================================
 *
 * Serial nrm2 does this in two PASSES over one array:
 *   Pass 1: scale = max(|x[i]|)
 *   Pass 2: sum of (x[i]/scale)^2, Kahan-compensated
 *   Result: scale * sqrt(sum)
 *
 * The MPI version needs the SAME two quantities, but each one now
 * has to be combined ACROSS ranks, not just across elements. That
 * means two synchronisation points instead of nrm2's one:
 *
 *   Step 1: local_scale  = max(|x[i]|) on this rank's slice
 *           (reuses blas_iamax(), same as the serial code does)
 *   Step 2: MPI_Allreduce(..., MPI_MAX, ...) -> global_scale
 *           [FIRST SYNC POINT]
 *   Step 3: local sum of (x[i]/global_scale)^2, Kahan-compensated
 *           -- every rank now divides by the SAME global_scale, so
 *           every term across every rank is safely in [0, 1]
 *   Step 4: MPI_Allreduce(..., MPI_SUM, ...) -> global_sum
 *           [SECOND SYNC POINT]
 *   Step 5: return global_scale * sqrt(global_sum)
 *
 * This costs one extra MPI collective compared to the simpler
 * version, but it means blas_mpi_nrm2 is correct for the exact same
 * input range the serial blas_nrm2 is correct for -- no silent
 * precision cliff introduced by parallelising.
 *
 * Parameters:
 *   n_local - number of elements in THIS RANK's local slice
 *   x       - this rank's local slice (read-only)
 *   incx    - stride for x
 *   comm    - communicator to reduce across
 *
 * Returns: the GLOBAL Euclidean norm, identical on every rank.
 *
 * Collective operation, TWICE over: every rank in comm must call
 * this function, and must do so in the same relative order as every
 * other rank (both Allreduce calls are synchronisation points).
 */

#ifndef BLAS1_MPI_NRM2_H
#define BLAS1_MPI_NRM2_H

#include <mpi.h>
#include "blas1/types.h"

BLAS_REAL blas_mpi_nrm2(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    MPI_Comm            comm
);

#endif /* BLAS1_MPI_NRM2_H */
