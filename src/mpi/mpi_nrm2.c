/*
 * mpi_nrm2.c — MPI-parallel Euclidean norm (step 6.5)
 *
 * See mpi_nrm2.h for why this needs two MPI_Allreduce calls instead
 * of one, and for the full algorithm walkthrough. Implementation
 * notes specific to this file:
 *
 * - Step 1 reuses blas_iamax(), exactly like the serial blas_nrm2()
 *   does. blas_iamax(n<=0, ...) returns the sentinel 0 -- if this
 *   rank's local slice is empty (n_local <= 0, which can legitimately
 *   happen if a global vector is smaller than the rank count), this
 *   rank contributes local_scale = 0.0 to the MPI_MAX and stays
 *   correct without any special-casing beyond the guard below.
 *
 * - global_scale == 0.0 after the MAX reduction means EVERY rank's
 *   local slice was either empty or all-zero. Either way the global
 *   vector's norm is exactly 0.0, and dividing by global_scale in
 *   step 3 would be a division by zero -- so this is checked and
 *   returned early, before step 3 runs.
 *
 * - Kahan compensation in step 3 mirrors nrm2.c's pass 2 exactly,
 *   same -ffast-math caveat applies (see nrm2.c and the Phase 5
 *   handoff notes: -DBLAS1_STRICT_IEEE=ON if compensation must
 *   survive fast-math's reassociation).
 */

#include "blas1/mpi/mpi_nrm2.h"
#include "blas1/mpi/mpi_types.h"
#include "blas1/iamax.h"
#include "blas1/types.h"
/* fabs()/sqrt() replaced by BLAS_FABS()/BLAS_SQRT() from types.h (precision-generic) */

BLAS_REAL blas_mpi_nrm2(
    blas_int            n_local,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    MPI_Comm            comm)
{
    /* ------------------------------------------------------------------
     * Step 1 — local scale = max(|x[i]|) on this rank's slice
     * ------------------------------------------------------------------ */
    BLAS_REAL local_scale = (BLAS_REAL)0.0;

    if (n_local > 0) {
        blas_int k = blas_iamax(n_local, x, incx);
        if (k > 0) {
            local_scale = BLAS_FABS(x[(k - 1) * incx]);
        }
    }

    /* ------------------------------------------------------------------
     * Step 2 — global scale across ALL ranks [FIRST SYNC POINT]
     * ------------------------------------------------------------------ */
    BLAS_REAL global_scale;
    MPI_Allreduce(
        &local_scale,
        &global_scale,
        1,
        BLAS_MPI_REAL,
        MPI_MAX,
        comm
    );

    /* Every rank's slice was empty or all-zero -- norm is exactly 0.
     * Must check before step 3, or we divide by zero below. */
    if (global_scale == (BLAS_REAL)0.0) {
        return (BLAS_REAL)0.0;
    }

    /* ------------------------------------------------------------------
     * Step 3 — local sum of (x[i]/global_scale)^2, Kahan-compensated
     *
     * Every rank divides by the SAME global_scale, so every term
     * (from every rank) lands in [0, 1] -- squaring it can neither
     * overflow nor underflow, exactly the property the serial
     * algorithm relies on, now guaranteed across the whole
     * distributed vector rather than just one rank's slice.
     * ------------------------------------------------------------------ */
    BLAS_REAL sum = (BLAS_REAL)0.0;
    BLAS_REAL c   = (BLAS_REAL)0.0;   /* Kahan compensation */

    if (incx == 1) {
        for (blas_int i = 0; i < n_local; i++) {
            BLAS_REAL si      = x[i] / global_scale;
            BLAS_REAL t       = si * si - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
        }
    } else {
        blas_int ix = 0;
        for (blas_int i = 0; i < n_local; i++) {
            BLAS_REAL si      = x[ix] / global_scale;
            BLAS_REAL t       = si * si - c;
            BLAS_REAL new_sum = sum + t;
            c   = (new_sum - sum) - t;
            sum = new_sum;
            ix += incx;
        }
    }

    /* ------------------------------------------------------------------
     * Step 4 — global sum of scaled squares [SECOND SYNC POINT]
     * ------------------------------------------------------------------ */
    BLAS_REAL global_sum;
    MPI_Allreduce(
        &sum,
        &global_sum,
        1,
        BLAS_MPI_REAL,
        MPI_SUM,
        comm
    );

    /* ------------------------------------------------------------------
     * Step 5 — undo the scaling
     * ------------------------------------------------------------------ */
    return global_scale * BLAS_SQRT(global_sum);
}
