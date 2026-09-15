/*
 * kernel/x86/dot_avx2.h — Private declarations for the AVX2 dot kernel
 *
 * NOT part of the public API. Included only by:
 *   - kernel/x86/dot_avx2.c  (own prototype — satisfies -Wmissing-prototypes)
 *   - src/dot.c dispatcher   (when USE_AVX2 is defined)
 *
 * Users of the library include <blas1/dot.h>, not this.
 */

#ifndef BLAS1_KERNEL_X86_DOT_AVX2_H
#define BLAS1_KERNEL_X86_DOT_AVX2_H

#include "blas1/types.h"

BLAS_REAL blas_dot_avx2(blas_int n,
                         const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                         const BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

BLAS_REAL blas_dot_kahan_avx2(blas_int n,
                                const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                                const BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

#endif /* BLAS1_KERNEL_X86_DOT_AVX2_H */
