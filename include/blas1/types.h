/*
 * types.h — Shared type definitions for the BLAS1 library
 *
 * Every source file in this project includes this header.
 * It defines:
 *   - Integer and floating-point typedefs
 *   - Compiler hint macros (inline, restrict, likely/unlikely)
 *   - A precision-selection mechanism (float vs double)
 */

#ifndef BLAS1_TYPES_H
#define BLAS1_TYPES_H

#include <stdint.h>   /* int64_t, int32_t */
#include <stddef.h>   /* size_t           */

/* ------------------------------------------------------------------ */
/*  1. Integer type                                                     */
/*     blas_int is used for vector lengths and stride values.           */
/*     We use int64_t so we can handle vectors with more than 2 billion */
/*     elements — common on modern HPC systems.                         */
/* ------------------------------------------------------------------ */
typedef int64_t  blas_int;

/* ------------------------------------------------------------------ */
/*  2. Floating-point types                                             */
/*     The library supports both single (float) and double precision.   */
/*     BLAS_REAL resolves to one of them at compile time depending on   */
/*     whether BLAS_USE_FLOAT is defined.                               */
/*                                                                      */
/*     Default: double (the HPC standard)                               */
/*     Override: compile with -DBLAS_USE_FLOAT for single precision     */
/* ------------------------------------------------------------------ */
typedef float   blas_float;
typedef double  blas_double;

#if defined(BLAS_USE_FLOAT)
    typedef blas_float  BLAS_REAL;
#else
    typedef blas_double BLAS_REAL;
#endif

/* ------------------------------------------------------------------ */
/*  3. BLAS_INLINE                                                      */
/*     Tells the compiler: "please inline this function at the call     */
/*     site." Avoids function-call overhead in tight loops.             */
/*     __attribute__((always_inline)) is a GCC/Clang extension that     */
/*     makes this a hard request, not a suggestion.                     */
/* ------------------------------------------------------------------ */
#if defined(__GNUC__) || defined(__clang__)
    #define BLAS_INLINE static inline __attribute__((always_inline))
#else
    #define BLAS_INLINE static inline
#endif

/* ------------------------------------------------------------------ */
/*  4. BLAS_RESTRICT                                                    */
/*     Tells the compiler that two pointers do NOT point to             */
/*     overlapping memory. This is the single most important hint for   */
/*     enabling auto-vectorisation — without it the compiler must        */
/*     assume aliasing and generates far more conservative code.        */
/*                                                                      */
/*     Usage:  void axpy(blas_int n,                                    */
/*                       const double * BLAS_RESTRICT x,               */
/*                       double       * BLAS_RESTRICT y);               */
/* ------------------------------------------------------------------ */
#if defined(__GNUC__) || defined(__clang__)
    #define BLAS_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define BLAS_RESTRICT __restrict
#else
    #define BLAS_RESTRICT
#endif

/* ------------------------------------------------------------------ */
/*  5. BLAS_LIKELY / BLAS_UNLIKELY                                      */
/*     Branch prediction hints. Tell the CPU which branch is the        */
/*     "normal" path so it can pipeline instructions more efficiently.  */
/*                                                                      */
/*     Usage:  if (BLAS_UNLIKELY(n <= 0)) return 0.0;                  */
/* ------------------------------------------------------------------ */
#if defined(__GNUC__) || defined(__clang__)
    #define BLAS_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define BLAS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define BLAS_LIKELY(x)   (x)
    #define BLAS_UNLIKELY(x) (x)
#endif

/* ------------------------------------------------------------------ */
/*  6. BLAS_UNUSED                                                      */
/*     Silences "unused parameter" warnings for parameters that are     */
/*     intentionally unused (e.g. in stub implementations).            */
/* ------------------------------------------------------------------ */
#define BLAS_UNUSED(x) ((void)(x))

#endif /* BLAS1_TYPES_H */