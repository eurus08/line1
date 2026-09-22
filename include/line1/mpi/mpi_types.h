/**
 * @file mpi_types.h
 * @ingroup mpi
 * @brief Maps BLAS_REAL to its matching MPI_Datatype, and defines the
 *        MAXLOC pair type used by blas_mpi_iamax().
 *
 * @defgroup mpi MPI Parallel LINE1 Operations
 * @brief MPI-parallel wrappers around the serial @ref serial
 *        operations. Requires building with @c -DLINE1_BUILD_MPI=ON.
 *
 * Every function in this group takes the calling rank's **local
 * slice** of a vector, not the whole global vector -- partitioning
 * the global vector across ranks is the caller's responsibility.
 * Each function is a thin wrapper that calls the existing serial
 * kernel locally, then performs the minimum number of MPI
 * collectives needed to combine results across ranks. No parallel
 * algorithm is reimplemented from scratch.
 *
 * The serial library picks @c double or @c float for @c BLAS_REAL at
 * compile time via @c BLAS_USE_FLOAT (see line1/types.h). Every MPI
 * collective call (@c MPI_Allreduce, @c MPI_Bcast, etc.) needs to be
 * told which @c MPI_Datatype it is moving -- passing the wrong one is
 * a silent correctness bug, not a compile error, since
 * @c MPI_Datatype is just an opaque handle to the API. This header
 * centralises that mapping in one place so every MPI wrapper file
 * stays consistent automatically if the precision is ever switched.
 */

#ifndef LINE1_MPI_TYPES_H
#define LINE1_MPI_TYPES_H

#include <mpi.h>
#include "line1/types.h"

/**
 * @def BLAS_MPI_REAL
 * @ingroup mpi
 * @brief The @c MPI_Datatype matching @c BLAS_REAL (@c MPI_DOUBLE or
 *        @c MPI_FLOAT depending on @c BLAS_USE_FLOAT).
 */
#if defined(BLAS_USE_FLOAT)
    #define BLAS_MPI_REAL MPI_FLOAT
#else
    #define BLAS_MPI_REAL MPI_DOUBLE
#endif

/**
 * @def BLAS_MPI_INT
 * @ingroup mpi
 * @brief The @c MPI_Datatype matching ::blas_int.
 *
 * @c blas_int is always @c int64_t regardless of @c BLAS_USE_FLOAT --
 * it's an index type, not a precision choice, so there is no
 * float/double branch here the way there is for ::BLAS_MPI_REAL.
 * Used by blas_mpi_iamax()'s @c MPI_Bcast of the true 64-bit global
 * index (see mpi_iamax.h for why that broadcast is needed at all).
 */
#define BLAS_MPI_INT MPI_INT64_T

/**
 * @struct blas_mpi_maxloc_pair_t
 * @ingroup mpi
 * @brief Value/location pair used with @c MPI_MAXLOC, for
 *        blas_mpi_iamax()'s reduction.
 *
 * MPI's built-in MAXLOC-compatible datatypes (@c MPI_DOUBLE_INT,
 * @c MPI_FLOAT_INT, ...) always pair the comparison value with a
 * plain 32-bit @c int location field -- there is no built-in pair
 * type with a 64-bit location. Since ::blas_int is 64-bit
 * specifically to support vectors past 2 billion elements, packing a
 * global index into that 32-bit field would silently break exactly
 * the case @c blas_int exists for. blas_mpi_iamax() works around this
 * by using the location field to hold a *rank number* (always safely
 * representable in 32 bits) instead of an element index, then
 * recovering the true 64-bit global index with a separate
 * @c MPI_Bcast from the winning rank -- see mpi_iamax.h for the full
 * algorithm.
 *
 * @note This struct's layout (@c BLAS_REAL value, then @c int loc)
 *       must exactly match what @c MPI_DOUBLE_INT / @c MPI_FLOAT_INT
 *       expect on the target platform. This is the same
 *       struct-matching technique every MPI program uses for
 *       MAXLOC/MINLOC, not something specific to this library.
 */
typedef struct {
    BLAS_REAL value; /**< The comparison value (max |x_i| candidate). */
    int       loc;   /**< The location paired with @c value -- holds a
                           rank number here, not an element index. */
} blas_mpi_maxloc_pair_t;

/**
 * @def BLAS_MPI_MAXLOC_TYPE
 * @ingroup mpi
 * @brief The @c MPI_Datatype to pass to @c MPI_Allreduce when
 *        reducing ::blas_mpi_maxloc_pair_t values with @c MPI_MAXLOC.
 */
#if defined(BLAS_USE_FLOAT)
    #define BLAS_MPI_MAXLOC_TYPE MPI_FLOAT_INT
#else
    #define BLAS_MPI_MAXLOC_TYPE MPI_DOUBLE_INT
#endif

#endif /* LINE1_MPI_TYPES_H */
