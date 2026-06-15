/*
 * asum.h — Sum of absolute values (L1 norm)
 *
 * Computes:  result = sum( |x[i*incx]| )  for i = 0..n-1
 *
 * Parameters:
 *   n     - number of elements
 *   x     - input vector (read-only)
 *   incx  - stride for x
 *
 * Returns: the sum of absolute values (always >= 0)
 */

#ifndef BLAS1_ASUM_H
#define BLAS1_ASUM_H

#include "blas1/types.h"

#include "blas1/types.h"


BLAS_REAL blas_asum(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx
);

#endif /* BLAS1_ASUM_H */