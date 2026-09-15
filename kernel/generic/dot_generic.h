/*
 * kernel/generic/dot_generic.h — Private declarations for the generic dot kernel
 *
 * This header is NOT part of the public API. It is included only by
 * kernel/generic/dot_generic.c (to satisfy -Wmissing-prototypes) and
 * by src/dot.c's dispatcher (instead of raw forward declarations).
 *
 * Users of the library never include this — they use <blas1/dot.h>.
 */

#ifndef BLAS1_KERNEL_GENERIC_DOT_GENERIC_H
#define BLAS1_KERNEL_GENERIC_DOT_GENERIC_H

#include "blas1/types.h"

BLAS_REAL blas_dot_generic(blas_int n,
                            const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                            const BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

BLAS_REAL blas_dot_kahan_generic(blas_int n,
                                  const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                                  const BLAS_REAL * BLAS_RESTRICT y, blas_int incy);

#endif /* BLAS1_KERNEL_GENERIC_DOT_GENERIC_H */
