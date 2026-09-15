/*
 * kernel/arm/axpy_neon.h — Private declarations for the NEON axpy kernel
 *
 * NOT part of the public API. Included only by:
 *   - kernel/arm/axpy_neon.c  (own prototype)
 *   - src/axpy.c dispatcher   (when USE_NEON is defined)
 */

#ifndef BLAS1_KERNEL_ARM_AXPY_NEON_H
#define BLAS1_KERNEL_ARM_AXPY_NEON_H

#include "blas1/types.h"

void blas_axpy_neon(blas_int n,
                     BLAS_REAL alpha,
                     const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                           BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

#endif /* BLAS1_KERNEL_ARM_AXPY_NEON_H */
