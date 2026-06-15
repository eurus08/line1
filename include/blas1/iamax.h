/*
 * iamax.h — Index of the element with the largest absolute value
 *
 * Computes:  result = first i where |x[i*incx]| is maximum
 *
 * Returns a 1-based index (BLAS convention inherited from Fortran).
 * Returns 0 if n <= 0 (the "no valid answer" sentinel).
 *
 * Parameters:
 *   n     - number of elements
 *   x     - input vector (read-only)
 *   incx  - stride for x
 *
 * Returns: 1-based index of the largest |x[i]|
 */

#ifndef BLAS1_IAMAX_H
#define BLAS1_IAMAX_H

#include "blas1/types.h"

#include "blas1/types.h"


blas_int blas_iamax(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx
);

#endif /* BLAS1_IAMAX_H */