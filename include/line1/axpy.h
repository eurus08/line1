/**
 * @file axpy.h
 * @ingroup serial
 * @brief Scalar multiply + vector add (y = alpha*x + y).
 */

#ifndef LINE1_AXPY_H
#define LINE1_AXPY_H

#include "line1/types.h"

/**
 * @ingroup serial
 * @brief Computes @f$ y \leftarrow \alpha x + y @f$.
 *
 * The name comes from "alpha times x plus y".
 *
 * @f[ y_{i \cdot \mathrm{incy}} \mathrel{+}= \alpha \cdot x_{i \cdot \mathrm{incx}}, \quad i = 0 \ldots n-1 @f]
 *
 * @note @f$ \alpha = 0 @f$ is a fast no-op path -- @p y is returned
 *       unmodified rather than paying for a multiply-by-zero pass
 *       over the array.
 *
 * @param n     Number of elements.
 * @param alpha Scalar multiplier.
 * @param x     Input vector (read-only).
 * @param incx  Stride for @p x. May be negative: matching reference
 *              BLAS's DAXPY, a negative @p incx walks backward
 *              through the SAME memory span @p x already points at
 *              the start of -- the caller does not need to (and
 *              should not) offset @p x itself. incx == 0 is caller
 *              error (undefined).
 * @param y     Input/output vector, modified in place.
 * @param incy  Stride for @p y. Same negative-stride convention as
 *              @p incx, independently.
 */
void blas_axpy(
    blas_int            n,
    BLAS_REAL           alpha,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    BLAS_REAL         * BLAS_RESTRICT y,
    blas_int            incy
);

#endif /* LINE1_AXPY_H */
