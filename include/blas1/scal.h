/*
 * scal.h — Scale a vector in-place
 *
 * Computes:  x[i*incx] *= alpha  for i = 0..n-1
 *
 * Special case: alpha == 0.0 must zero the vector explicitly
 * (not multiply), to avoid NaN propagation if x contains Inf or NaN.
 *
 * Parameters:
 *   n     - number of elements
 *   alpha - scalar multiplier
 *   x     - input/output vector (modified in place)
 *   incx  - stride for x
 */

#ifndef BLAS1_SCAL_H
#define BLAS1_SCAL_H

#include "types.h"

void blas_scal(
    blas_int      n,
    BLAS_REAL     alpha,
    BLAS_REAL   * BLAS_RESTRICT x,
    blas_int      incx
);

#endif /* BLAS1_SCAL_H */