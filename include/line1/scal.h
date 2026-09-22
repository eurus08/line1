/**
 * @file scal.h
 * @ingroup serial
 * @brief Scale a vector in place (x = alpha*x).
 */

#ifndef LINE1_SCAL_H
#define LINE1_SCAL_H

#include "line1/types.h"

/**
 * @ingroup serial
 * @brief Scales a vector in place: @f$ x \leftarrow \alpha x @f$.
 *
 * @f[ x_{i \cdot \mathrm{incx}} \mathrel{*}= \alpha, \quad i = 0 \ldots n-1 @f]
 *
 * @warning The @f$ \alpha = 0 @f$ case explicitly zeroes every
 *          element rather than multiplying by zero. This matters:
 *          if @p x contains @c Inf or @c NaN, @c Inf*0 and @c NaN*0
 *          both produce @c NaN under IEEE 754, silently corrupting
 *          the result. Explicit zeroing avoids that propagation
 *          entirely.
 *
 * @param n     Number of elements.
 * @param alpha Scalar multiplier.
 * @param x     Input/output vector, modified in place.
 * @param incx  Stride for @p x. Must be > 0 -- matching reference BLAS's
 *              DSCAL, this does NOT support negative increments (unlike
 *              blas_dot()/blas_axpy(), which do). @p n &le; 0 or
 *              @p incx &le; 0 returns immediately without modifying
 *              @p x, rather than reading/writing out of bounds.
 */
void blas_scal(
    blas_int      n,
    BLAS_REAL     alpha,
    BLAS_REAL   * BLAS_RESTRICT x,
    blas_int      incx
);

#endif /* LINE1_SCAL_H */
