/*
 * kernel/x86/axpy_avx2.c — AVX2 + FMA explicit intrinsics kernel for blas_axpy
 *
 * Step 5.5 of the build plan (AVX2 part). Mirrors step 5.3 in the build plan.
 *
 * Compiled ONLY when USE_AVX2=1. Requires -mavx2 -mfma on this translation
 * unit (applied via set_source_files_properties in CMakeLists.txt).
 *
 * =========================================================================
 * AXPY vs DOT — WHAT'S DIFFERENT FOR AVX2
 * =========================================================================
 *
 * dot:  load x, load y, multiply, accumulate into register → no store
 * axpy: load x, load y, FMA into y, STORE y back → 3 memory ops/element
 *
 * Key implications:
 *
 *   1. Alpha is broadcast once into a __m256d register before the loop.
 *      We reuse the same alpha register every iteration — no reload needed.
 *      _mm256_set1_pd(alpha) broadcasts a scalar double into all 4 lanes.
 *
 *   2. We use _mm256_fmadd_pd(valpha, x, y) which computes alpha*x + y
 *      for all 4 lanes in one instruction, then store the result back to y.
 *
 *   3. 4 independent streams (acc0..acc3) prevent the CPU from stalling on
 *      store-forwarding dependencies — each stream works on a different
 *      cache line, so loads and stores can be pipelined by the memory unit.
 *
 *   4. Unaligned loads AND stores (_mm256_storeu_pd) are used — same
 *      alignment-agnostic approach as dot_avx2.c.
 *
 * The caller (src/axpy.c dispatcher) guarantees alpha != 0 when we arrive
 * here — the alpha==0 fast path is handled before the kernel is called.
 *
 * =========================================================================
 * FLOAT PRECISION GUARD
 * =========================================================================
 *
 * Same as dot_avx2.c — _pd intrinsics require double. Float builds must
 * not reach this file.
 */

#include <immintrin.h>

#include "blas1/types.h"
#include "axpy_avx2.h"   /* own prototype — satisfies -Wmissing-prototypes */

#ifndef BLAS_USE_FLOAT
/* double build — proceed */
#else
#  error "axpy_avx2.c uses _pd (packed double) intrinsics and cannot be compiled \
for float precision (-DBLAS_USE_FLOAT)."
#endif

/* -------------------------------------------------------------------------
 * blas_axpy_avx2 — y += alpha*x with explicit AVX2 + FMA
 *
 * Processes 16 doubles per iteration (4 vectors × 4 lanes).
 * Alpha is broadcast once before the loop into valpha.
 * Scalar tail handles the last 0..15 elements.
 * Strided path falls back to scalar — same gather-is-slow reasoning as dot.
 * ------------------------------------------------------------------------- */
void blas_axpy_avx2(blas_int n,
                     BLAS_REAL alpha,
                     const BLAS_REAL * BLAS_RESTRICT x, blas_int incx,
                           BLAS_REAL * BLAS_RESTRICT y, blas_int incy)
{
    /* Strided fallback — supports negative incx/incy (blas_stride_start). */
    if (incx != 1 || incy != 1) {
        blas_int ix = blas_stride_start(n, incx);
        blas_int iy = blas_stride_start(n, incy);
        for (blas_int i = 0; i < n; i++) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
        return;
    }

    /* ----------------------------------------------------------------
     * Unit-stride AVX2 path
     *
     * Broadcast alpha into all 4 lanes of a __m256d register once.
     * Inner loop: load x chunk, load y chunk, FMA, store y chunk.
     * 4 independent streams hide load/store latency.
     * ---------------------------------------------------------------- */
    __m256d valpha = _mm256_set1_pd(alpha);   /* [α, α, α, α] */

    blas_int i   = 0;
    blas_int n16 = (n / 16) * 16;

    for (; i < n16; i += 16) {
        /* Load 4 chunks of x (read-only) */
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d x2 = _mm256_loadu_pd(x + i + 8);
        __m256d x3 = _mm256_loadu_pd(x + i + 12);

        /* Load 4 chunks of y (read-modify-write) */
        __m256d y0 = _mm256_loadu_pd(y + i);
        __m256d y1 = _mm256_loadu_pd(y + i + 4);
        __m256d y2 = _mm256_loadu_pd(y + i + 8);
        __m256d y3 = _mm256_loadu_pd(y + i + 12);

        /* FMA: y += alpha * x  (fused, one instruction per vector) */
        y0 = _mm256_fmadd_pd(valpha, x0, y0);
        y1 = _mm256_fmadd_pd(valpha, x1, y1);
        y2 = _mm256_fmadd_pd(valpha, x2, y2);
        y3 = _mm256_fmadd_pd(valpha, x3, y3);

        /* Store results back to y */
        _mm256_storeu_pd(y + i,      y0);
        _mm256_storeu_pd(y + i + 4,  y1);
        _mm256_storeu_pd(y + i + 8,  y2);
        _mm256_storeu_pd(y + i + 12, y3);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        y[i] += alpha * x[i];
    }
}
