/*
 * mpi_types.h — Maps BLAS_REAL to its matching MPI_Datatype
 *
 * The serial library picks double or float for BLAS_REAL at compile
 * time via BLAS_USE_FLOAT (see blas1/types.h). Every MPI collective
 * call (MPI_Allreduce, MPI_Bcast, etc.) needs to be told which
 * MPI_Datatype it is moving — passing the wrong one is a silent
 * correctness bug, not a compile error, since MPI_Datatype is just an
 * opaque handle to the API.
 *
 * BLAS_MPI_REAL centralises that mapping in one place so every MPI
 * wrapper file (mpi_dot.c, mpi_nrm2.c, ...) stays consistent
 * automatically if the precision is ever switched.
 *
 * This header also defines BLAS_MPI_INT and the MAXLOC pair type
 * used by mpi_iamax.c — see the comment above each for why they
 * exist.
 */

#ifndef BLAS1_MPI_TYPES_H
#define BLAS1_MPI_TYPES_H

#include <mpi.h>
#include "blas1/types.h"

#if defined(BLAS_USE_FLOAT)
    #define BLAS_MPI_REAL MPI_FLOAT
#else
    #define BLAS_MPI_REAL MPI_DOUBLE
#endif

/*
 * BLAS_MPI_INT — the MPI_Datatype matching blas_int.
 *
 * blas_int is always int64_t (see blas1/types.h), regardless of
 * BLAS_USE_FLOAT -- it's an INDEX type, not a precision choice, so
 * there is no float/double branch here the way there is for
 * BLAS_MPI_REAL. Used by mpi_iamax.c's MPI_Bcast of the true 64-bit
 * global index (see mpi_iamax.h for why that Bcast is needed at
 * all).
 */
#define BLAS_MPI_INT MPI_INT64_T

/*
 * MAXLOC pair type — used by mpi_iamax.c's MPI_Allreduce(MPI_MAXLOC).
 *
 * MPI's built-in MAXLOC-compatible datatypes (MPI_DOUBLE_INT,
 * MPI_FLOAT_INT, ...) always pair the comparison VALUE with a plain
 * 32-bit `int` LOCATION field -- there is no built-in pair type with
 * a 64-bit location. Since blas_int is 64-bit specifically to
 * support vectors past 2 billion elements, packing a global index
 * into that 32-bit field would silently break exactly the case
 * blas_int exists for. mpi_iamax.c works around this by using the
 * location field to hold a RANK NUMBER (always safely representable
 * in 32 bits) instead of an element index, then recovering the true
 * 64-bit global index with a separate MPI_Bcast from the winning
 * rank. See mpi_iamax.h and mpi_iamax.c for the full algorithm.
 *
 * blas_mpi_maxloc_pair_t's layout (BLAS_REAL value, then int loc)
 * must exactly match what MPI_DOUBLE_INT / MPI_FLOAT_INT expect on
 * this platform -- this is the same struct-matching technique every
 * MPI program uses for MAXLOC/MINLOC, not something specific to this
 * library.
 */
typedef struct {
    BLAS_REAL value;
    int       loc;
} blas_mpi_maxloc_pair_t;

#if defined(BLAS_USE_FLOAT)
    #define BLAS_MPI_MAXLOC_TYPE MPI_FLOAT_INT
#else
    #define BLAS_MPI_MAXLOC_TYPE MPI_DOUBLE_INT
#endif

#endif /* BLAS1_MPI_TYPES_H */
