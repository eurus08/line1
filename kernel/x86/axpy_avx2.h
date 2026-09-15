/*
 * kernel/x86/axpy_avx2.h — Private declarations for the AVX2 axpy kernel
 *
 * NOT part of the public API. Included only by:
 *   - kernel/x86/axpy_avx2.c  (own prototype)
 *   - src/axpy.c dispatcher    (when USE_AVX2 is defined)
 */

#ifndef BLAS1_KERNEL_X86_AXPY_AVX2_H
#define BLAS1_KERNEL_X86_AXPY_AVX2_H

#include "blas1/types.h"

void blas_axpy_avx2(blas_int n,
                     BLAS_REAL alpha,
                     const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                           BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

#endif /* BLAS1_KERNEL_X86_AXPY_AVX2_H */
