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

#endif /* BLAS1_MPI_TYPES_H */
