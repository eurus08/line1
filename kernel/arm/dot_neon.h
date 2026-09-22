/*
 * kernel/arm/dot_neon.h — Private declarations for the NEON dot kernel
 *
 * NOT part of the public API. Included only by:
 *   - kernel/arm/dot_neon.c  (own prototype — satisfies -Wmissing-prototypes)
 *   - src/dot.c dispatcher   (when USE_NEON is defined)
 */

#ifndef LINE1_KERNEL_ARM_DOT_NEON_H
#define LINE1_KERNEL_ARM_DOT_NEON_H

#include "line1/types.h"

BLAS_REAL blas_dot_neon(blas_int n,
                         const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                         const BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

BLAS_REAL blas_dot_kahan_neon(blas_int n,
                               const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                               const BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

#endif /* LINE1_KERNEL_ARM_DOT_NEON_H */
