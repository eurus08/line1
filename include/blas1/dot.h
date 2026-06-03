/*
 * dot.h — Dot product of two vectors
 *
 * Computes:  result = sum( x[i*incx] * y[i*incy] )  for i = 0..n-1
 *
 * Parameters:
 *   n     - number of elements
 *   x     - first input vector  (read-only)
 *   incx  - stride for x (1 = contiguous, 2 = every other element, ...)
 *   y     - second input vector (read-only)
 *   incy  - stride for y
 *
 * Returns: the scalar dot product (double or float depending on precision)
 */

#ifndef BLAS1_DOT_H
#define BLAS1_DOT_H

#include "types.h"

BLAS_REAL blas_dot(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    const BLAS_REAL   * BLAS_RESTRICT y,
    blas_int            incy
);

#endif /* BLAS1_DOT_H */