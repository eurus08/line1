/**
 * @file dot.h
 * @ingroup serial
 * @brief Dot product of two vectors.
 */

#ifndef BLAS1_DOT_H
#define BLAS1_DOT_H

#include "blas1/types.h"

/**
 * @ingroup serial
 * @brief Computes the dot product of two vectors.
 *
 * @f[ \mathrm{result} = \sum_{i=0}^{n-1} x_{i \cdot \mathrm{incx}} \cdot y_{i \cdot \mathrm{incy}} @f]
 *
 * @param n     Number of elements.
 * @param x     First input vector (read-only).
 * @param incx  Stride for @p x (1 = contiguous, 2 = every other element, ...).
 * @param y     Second input vector (read-only).
 * @param incy  Stride for @p y.
 * @return The scalar dot product.
 */
BLAS_REAL blas_dot(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    const BLAS_REAL   * BLAS_RESTRICT y,
    blas_int            incy
);

/**
 * @ingroup serial
 * @brief Dot product with Kahan-compensated summation.
 *
 * Identical contract to blas_dot(), but accumulates the running sum
 * with Kahan compensation to reduce floating-point rounding error.
 * This costs extra arithmetic per element, so it is a distinct
 * opt-in function rather than blas_dot()'s default behaviour --
 * callers who don't need the extra precision shouldn't pay for it
 * unknowingly.
 *
 * @note Kahan compensation can be eliminated by aggressive
 *       floating-point reassociation (e.g. @c -ffast-math). Build
 *       with @c -DBLAS1_STRICT_IEEE=ON if the compensation must
 *       survive exactly as written.
 *
 * @param n     Number of elements.
 * @param x     First input vector (read-only).
 * @param incx  Stride for @p x.
 * @param y     Second input vector (read-only).
 * @param incy  Stride for @p y.
 * @return The scalar dot product, computed with Kahan compensation.
 */
BLAS_REAL blas_dot_kahan(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    const BLAS_REAL   * BLAS_RESTRICT y,
    blas_int            incy
);

#endif /* BLAS1_DOT_H */
