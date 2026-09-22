/**
 * @file dot.h
 * @ingroup serial
 * @brief Dot product of two vectors.
 */

#ifndef LINE1_DOT_H
#define LINE1_DOT_H

#include "line1/types.h"

/**
 * @ingroup serial
 * @brief Computes the dot product of two vectors.
 *
 * @f[ \mathrm{result} = \sum_{i=0}^{n-1} x_{i \cdot \mathrm{incx}} \cdot y_{i \cdot \mathrm{incy}} @f]
 *
 * @param n     Number of elements.
 * @param x     First input vector (read-only).
 * @param incx  Stride for @p x (1 = contiguous, 2 = every other element, ...).
 *              May be negative: matching reference BLAS's DDOT, a
 *              negative @p incx walks backward through the SAME
 *              memory span @p x already points at the start of --
 *              the caller does not need to (and should not) offset
 *              @p x itself. incx == 0 is caller error (undefined).
 * @param y     Second input vector (read-only).
 * @param incy  Stride for @p y. Same negative-stride convention as
 *              @p incx, independently.
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
 *       floating-point reassociation (e.g. @c -ffast-math). Strict
 *       IEEE 754 (no @c -ffast-math) is this project's DEFAULT build
 *       for exactly this reason; the compensation survives as written
 *       unless you explicitly opt in to @c -ffast-math with
 *       @c -DLINE1_STRICT_IEEE=OFF.
 *
 * @note Overflow: once the running sum has overflowed to &plusmn;Inf,
 *       this falls back to plain addition rather than trusting the
 *       Kahan compensation term (which can itself become infinite and
 *       produce a spurious NaN where the true answer is a
 *       well-defined &plusmn;Inf) -- except on the AVX2/NEON
 *       unit-stride SIMD paths, where this fallback is not yet
 *       implemented (known limitation; see kernel/x86/dot_avx2.c).
 *
 * @param n     Number of elements.
 * @param x     First input vector (read-only).
 * @param incx  Stride for @p x. May be negative -- same convention as
 *              blas_dot() (see its documentation).
 * @param y     Second input vector (read-only).
 * @param incy  Stride for @p y. Same negative-stride convention as
 *              @p incx, independently.
 * @return The scalar dot product, computed with Kahan compensation.
 */
BLAS_REAL blas_dot_kahan(
    blas_int            n,
    const BLAS_REAL   * BLAS_RESTRICT x,
    blas_int            incx,
    const BLAS_REAL   * BLAS_RESTRICT y,
    blas_int            incy
);

#endif /* LINE1_DOT_H */
