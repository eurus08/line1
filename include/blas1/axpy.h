/*
 * axpy.h — Scalar multiply + vector add
 *
 * Computes:  y[i*incy] += alpha * x[i*incx]  for i = 0..n-1
 *
 * The name comes from "alpha * x plus y" 
 *
 * Parameters:
 *   n     - number of elements
 *   alpha - scalar multiplier
 *   x     - input vector (read-only)
 *   incx  - stride for x
 *   y     - input/output vector (modified in place)
 *   incy  - stride for y
 */

#ifndef BLAS1_AXPY_H
#define BLAS1_AXPY_H

#include "types.h"

void blas_axpy(
    blas_int            n,
    BLAS_REAL           alpha,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    BLAS_REAL         * BLAS_RESTRICT y,
    blas_int            incy
);

#endif /* BLAS1_AXPY_H */