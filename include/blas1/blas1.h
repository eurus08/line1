/**
 * @file blas1.h
 * @brief Umbrella header for the BLAS1 library -- include this one
 *        header to get all six BLAS Level 1 operations.
 *
 * @code
 * #include <blas1/blas1.h>
 * @endcode
 *
 * Precision is selected at compile time:
 *   - Default: double precision
 *   - @c -DBLAS_USE_FLOAT : single precision
 *
 * @defgroup serial Serial BLAS1 Operations
 * @brief Single-process implementations of the six BLAS Level 1
 *        operations: dot, axpy, scal, nrm2, asum, iamax.
 *
 * Every function here operates on one process's own memory only --
 * for the MPI-parallel equivalents, see @ref mpi (requires building
 * with @c -DBLAS1_BUILD_MPI=ON).
 */

#ifndef BLAS1_H
#define BLAS1_H

/* Shared types and compiler macros — must come first */
#include "blas1/types.h"

/* The 6 BLAS Level 1 operations */
#include "blas1/dot.h"
#include "blas1/axpy.h"
#include "blas1/scal.h"
#include "blas1/nrm2.h"
#include "blas1/asum.h"
#include "blas1/iamax.h"

/**
 * @def BLAS1_VERSION_MAJOR
 * @brief Library major version.
 */
#define BLAS1_VERSION_MAJOR 1
/**
 * @def BLAS1_VERSION_MINOR
 * @brief Library minor version.
 */
#define BLAS1_VERSION_MINOR 0
/**
 * @def BLAS1_VERSION_PATCH
 * @brief Library patch version.
 */
#define BLAS1_VERSION_PATCH 0
/**
 * @def BLAS1_VERSION
 * @brief Library version encoded as a single integer for easy
 *        comparison, e.g. version 1.2.3 -> 10203.
 */
#define BLAS1_VERSION \
    (BLAS1_VERSION_MAJOR * 10000 + \
     BLAS1_VERSION_MINOR * 100   + \
     BLAS1_VERSION_PATCH)

#endif /* BLAS1_H */
