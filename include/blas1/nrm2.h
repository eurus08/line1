/**
 * @file nrm2.h
 * @ingroup serial
 * @brief Euclidean (L2) norm of a vector.
 */

#ifndef BLAS1_NRM2_H
#define BLAS1_NRM2_H

#include "blas1/types.h"

/**
 * @ingroup serial
 * @brief Computes the Euclidean norm of a vector.
 *
 * @f[ \mathrm{result} = \sqrt{\sum_{i=0}^{n-1} x_{i \cdot \mathrm{incx}}^2} @f]
 *
 * Uses a numerically stable scaled two-pass algorithm (the
 * Blue 1978 / LAPACK @c dnrm2 technique: find the largest-magnitude
 * element first, divide every term by it before squaring, then
 * multiply back at the end) rather than the naive formula above
 * computed directly. This avoids overflow when elements are very
 * large (naive squaring can overflow to @c Inf) and underflow when
 * elements are very small (naive squaring can vanish to @c 0).
 *
 * @param n     Number of elements.
 * @param x     Input vector (read-only).
 * @param incx  Stride for @p x.
 * @return The Euclidean norm (always &ge; 0).
 */
BLAS_REAL blas_nrm2(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx
);

#endif /* BLAS1_NRM2_H */
