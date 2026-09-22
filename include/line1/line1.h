/**
 * @file line1.h
 * @brief Umbrella header for the LINE1 library -- include this one
 *        header to get all six BLAS Level 1 operations.
 *
 * @code
 * #include <line1/line1.h>
 * @endcode
 *
 * Precision is selected at compile time:
 *   - Default: double precision
 *   - @c -DBLAS_USE_FLOAT : single precision
 *
 * @defgroup serial Serial LINE1 Operations
 * @brief Single-process implementations of the six BLAS Level 1
 *        operations: dot, axpy, scal, nrm2, asum, iamax.
 *
 * Every function here operates on one process's own memory only --
 * for the MPI-parallel equivalents, see @ref mpi (requires building
 * with @c -DLINE1_BUILD_MPI=ON).
 */

#ifndef LINE1_H
#define LINE1_H

/* Shared types and compiler macros — must come first */
#include "line1/types.h"

/* The 6 BLAS Level 1 operations */
#include "line1/dot.h"
#include "line1/axpy.h"
#include "line1/scal.h"
#include "line1/nrm2.h"
#include "line1/asum.h"
#include "line1/iamax.h"

/**
 * @def LINE1_VERSION_MAJOR
 * @brief Library major version.
 */
#define LINE1_VERSION_MAJOR 1
/**
 * @def LINE1_VERSION_MINOR
 * @brief Library minor version.
 */
#define LINE1_VERSION_MINOR 0
/**
 * @def LINE1_VERSION_PATCH
 * @brief Library patch version.
 */
#define LINE1_VERSION_PATCH 0
/**
 * @def LINE1_VERSION
 * @brief Library version encoded as a single integer for easy
 *        comparison, e.g. version 1.2.3 -> 10203.
 */
#define LINE1_VERSION \
    (LINE1_VERSION_MAJOR * 10000 + \
     LINE1_VERSION_MINOR * 100   + \
     LINE1_VERSION_PATCH)

#endif /* LINE1_H */
