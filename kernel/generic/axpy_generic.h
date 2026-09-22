/*
 * kernel/generic/axpy_generic.h — Private declarations for the generic axpy kernel
 *
 * NOT part of the public API. Included only by:
 *   - kernel/generic/axpy_generic.c  (own prototype)
 *   - src/axpy.c dispatcher          (under the #else branch)
 */

#ifndef LINE1_KERNEL_GENERIC_AXPY_GENERIC_H
#define LINE1_KERNEL_GENERIC_AXPY_GENERIC_H

#include "line1/types.h"

void blas_axpy_generic(blas_int n,
                        BLAS_REAL alpha,
                        const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                              BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

#endif /* LINE1_KERNEL_GENERIC_AXPY_GENERIC_H */
