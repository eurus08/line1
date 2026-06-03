/*
 * blas1.h — Umbrella header for the BLAS1 library
 *
 * Users only need to write:
 *
 *     #include <blas1/blas1.h>
 *
 * This include gives access to all 6 BLAS 1 operations:
 *   dot, axpy, scal, nrm2, asum, iamax
 *
 * Precision is selected at compile time:
 *   Default          -> double precision
 *   -DBLAS_USE_FLOAT -> single precision
 */

#ifndef BLAS1_H
#define BLAS1_H

/* Shared types and compiler macros — must come first */
#include "types.h"

/* The 6 BLAS Level 1 operations */
#include "dot.h"
#include "axpy.h"
#include "scal.h"
#include "nrm2.h"
#include "asum.h"
#include "iamax.h"

/*
 * Library version — encoded as a single integer for easy comparison.
 * e.g. version 1.0.0 -> 10000
 *      version 1.2.3 -> 10203
 */
#define BLAS1_VERSION_MAJOR 1
#define BLAS1_VERSION_MINOR 0
#define BLAS1_VERSION_PATCH 0
#define BLAS1_VERSION \
    (BLAS1_VERSION_MAJOR * 10000 + \
     BLAS1_VERSION_MINOR * 100   + \
     BLAS1_VERSION_PATCH)

#endif /* BLAS1_H */