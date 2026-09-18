/**
 * @file asum.h
 * @ingroup serial
 * @brief Sum of absolute values (L1 norm) of a vector.
 */

#ifndef BLAS1_ASUM_H
#define BLAS1_ASUM_H

#include "blas1/types.h"

/**
 * @ingroup serial
 * @brief Computes the sum of absolute values of a vector's elements.
 *
 * @f[ \mathrm{result} = \sum_{i=0}^{n-1} \left| x_{i \cdot \mathrm{incx}} \right| @f]
 *
 * @param n     Number of elements.
 * @param x     Input vector (read-only).
 * @param incx  Stride for @p x.
 * @return The sum of absolute values (always &ge; 0).
 */
BLAS_REAL blas_asum(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx
);

#endif /* BLAS1_ASUM_H */
