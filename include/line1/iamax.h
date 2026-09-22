/**
 * @file iamax.h
 * @ingroup serial
 * @brief Index of the element with the largest absolute value.
 */

#ifndef LINE1_IAMAX_H
#define LINE1_IAMAX_H

#include "line1/types.h"

/**
 * @ingroup serial
 * @brief Finds the index of the largest-magnitude element in a vector.
 *
 * @f[ \mathrm{result} = \operatorname*{arg\,max}_i \left| x_{i \cdot \mathrm{incx}} \right| @f]
 *
 * On ties, the first occurrence wins.
 *
 * @note Returns a **1-based** index (the BLAS convention, inherited
 *       from Fortran), not a 0-based C array index.
 *
 * @param n     Number of elements.
 * @param x     Input vector (read-only).
 * @param incx  Stride for @p x. Must be > 0 -- matching reference BLAS's
 *              IDAMAX, this does NOT support negative increments (unlike
 *              blas_dot()/blas_axpy(), which do). @p incx &le; 0 returns
 *              0 immediately, checked BEFORE the n == 1 case, so it
 *              takes precedence even when @p n == 1.
 * @return 1-based index of the largest @f$ |x_i| @f$, or @c 0 if
 *         @f$ n \le 0 @f$ or @p incx &le; 0 (the "no valid answer"
 *         sentinel).
 */
blas_int blas_iamax(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx
);

#endif /* LINE1_IAMAX_H */
