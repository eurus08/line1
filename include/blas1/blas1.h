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
#include "blas1/types.h"

/* The 6 BLAS Level 1 operations */
#include "blas1/dot.h"
#include "blas1/axpy.h"
#include "blas1/scal.h"
#include "blas1/nrm2.h"
#include "blas1/asum.h"
#include "blas1/iamax.h"

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