/**
 * @file types.h
 * @brief Shared type definitions, precision selection, and compiler
 *        hint macros used throughout the BLAS1 library.
 *
 * Every source file in this project includes this header, directly
 * or transitively. It defines:
 *   - Integer and floating-point typedefs
 *   - Compiler hint macros (inline, restrict, likely/unlikely)
 *   - A precision-selection mechanism (float vs double)
 */

#ifndef BLAS1_TYPES_H
#define BLAS1_TYPES_H

#include <stdint.h>   /* int64_t, int32_t */
#include <stddef.h>   /* size_t           */

/**
 * @typedef blas_int
 * @brief Index/length/stride type used throughout the library.
 *
 * @c int64_t is used (not @c int) so that vectors with more than 2
 * billion elements can be addressed correctly -- a real constraint
 * on modern HPC systems, not a hypothetical one.
 */
typedef int64_t  blas_int;

/** @brief Single-precision floating-point type. */
typedef float   blas_float;

/** @brief Double-precision floating-point type. */
typedef double  blas_double;

/**
 * @typedef BLAS_REAL
 * @brief The library's working floating-point type.
 *
 * Resolves to ::blas_float if @c BLAS_USE_FLOAT is defined at compile
 * time, otherwise to ::blas_double (the default, and the HPC
 * standard). Every public function signature in this library uses
 * @c BLAS_REAL rather than @c float or @c double directly, so the
 * whole library's precision can be switched with a single define.
 */
#if defined(BLAS_USE_FLOAT)
    typedef blas_float  BLAS_REAL;
#else
    typedef blas_double BLAS_REAL;
#endif

/**
 * @def BLAS_INLINE
 * @brief Requests aggressive inlining at the call site.
 *
 * On GCC/Clang this expands to @c __attribute__((always_inline)),
 * making the request a hard requirement rather than a hint --
 * avoiding function-call overhead in tight loops. Falls back to a
 * plain @c static @c inline on other compilers.
 */
#if defined(__GNUC__) || defined(__clang__)
    #define BLAS_INLINE static inline __attribute__((always_inline))
#else
    #define BLAS_INLINE static inline
#endif

/**
 * @def BLAS_RESTRICT
 * @brief Marks a pointer as non-aliasing.
 *
 * Tells the compiler that this pointer does not point to memory that
 * any other pointer in the same function can also reach. This is the
 * single most important hint for enabling auto-vectorisation --
 * without it, the compiler must conservatively assume aliasing and
 * generates far less efficient code.
 *
 * @par Example
 * @code
 * void axpy(blas_int n,
 *           const double * BLAS_RESTRICT x,
 *           double       * BLAS_RESTRICT y);
 * @endcode
 */
#if defined(__GNUC__) || defined(__clang__)
    #define BLAS_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define BLAS_RESTRICT __restrict
#else
    #define BLAS_RESTRICT
#endif

/**
 * @def BLAS_LIKELY
 * @brief Branch-prediction hint: this condition is expected to be true.
 * @param x The condition being hinted about.
 */
/**
 * @def BLAS_UNLIKELY
 * @brief Branch-prediction hint: this condition is expected to be false.
 * @param x The condition being hinted about.
 *
 * @par Example
 * @code
 * if (BLAS_UNLIKELY(n <= 0)) return 0.0;
 * @endcode
 */
#if defined(__GNUC__) || defined(__clang__)
    #define BLAS_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define BLAS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define BLAS_LIKELY(x)   (x)
    #define BLAS_UNLIKELY(x) (x)
#endif

/**
 * @def BLAS_UNUSED
 * @brief Silences "unused parameter" warnings for parameters that are
 *        intentionally unused.
 * @param x The parameter or variable being marked as intentionally unused.
 */
#define BLAS_UNUSED(x) ((void)(x))

#endif /* BLAS1_TYPES_H */
