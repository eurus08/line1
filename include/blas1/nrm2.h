/*
 * nrm2.h — Euclidean norm (L2 norm) of a vector
 *
 * Computes:  result = sqrt( sum( x[i*incx]^2 ) )  for i = 0..n-1
 *
 * Uses a numerically stable algorithm to avoid overflow when
 * elements are very large, and underflow when they are very small.
 *
 * Parameters:
 *   n     - number of elements
 *   x     - input vector (read-only)
 *   incx  - stride for x
 *
 * Returns: the Euclidean norm (always >= 0)
 */

#ifndef BLAS1_NRM2_H
#define BLAS1_NRM2_H

#include "types.h"

BLAS_REAL blas_nrm2(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx
);

#endif /* BLAS1_NRM2_H */