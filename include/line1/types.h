/**
 * @file types.h
 * @brief Shared type definitions, precision selection, and compiler
 *        hint macros used throughout the LINE1 library.
 *
 * Every source file in this project includes this header, directly
 * or transitively. It defines:
 *   - Integer and floating-point typedefs
 *   - Compiler hint macros (inline, restrict, likely/unlikely)
 *   - A precision-selection mechanism (float vs double)
 */

#ifndef LINE1_TYPES_H
#define LINE1_TYPES_H

#include <stdint.h>   /* int64_t, int32_t, uint32_t, uint64_t */
#include <stddef.h>   /* size_t           */
#include <string.h>   /* memcpy() -- see blas_is_inf() below  */
#include <math.h>     /* fabs()/fabsf(), sqrt()/sqrtf() -- see BLAS_FABS/BLAS_SQRT below */
#include <float.h>    /* DBL_MAX, FLT_MAX -- see BLAS_REAL_MAX below */

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
 * @def BLAS_FABS
 * @brief Precision-generic absolute value on BLAS_REAL.
 *
 * <math.h>'s fabs()/sqrt() take and return @c double; calling them
 * directly on a @c float (BLAS_REAL when BLAS_USE_FLOAT is set)
 * silently promotes the argument to double and truncates the result
 * back to float on return -- extra work, a -Wdouble-promotion warning
 * at the call, and a -Wfloat-conversion/-Wconversion warning on the
 * implicit narrowing back to BLAS_REAL. Every call site operating on
 * a BLAS_REAL value should use this (and ::BLAS_SQRT) instead of
 * fabs()/sqrt() directly, so the single-precision build actually stays
 * in float.
 *
 * @param x A @c BLAS_REAL value.
 * @return  The absolute value of @p x, as a @c BLAS_REAL.
 */
/**
 * @def BLAS_SQRT
 * @brief Precision-generic square root on BLAS_REAL.
 *
 * See ::BLAS_FABS immediately above for why this exists.
 *
 * @param x A @c BLAS_REAL value.
 * @return  The square root of @p x, as a @c BLAS_REAL.
 */
#if defined(BLAS_USE_FLOAT)
    #define BLAS_FABS(x)  fabsf(x)
    #define BLAS_SQRT(x)  sqrtf(x)
#else
    #define BLAS_FABS(x)  fabs(x)
    #define BLAS_SQRT(x)  sqrt(x)
#endif

/**
 * @def BLAS_REAL_MAX
 * @brief Largest finite representable BLAS_REAL value (DBL_MAX / FLT_MAX).
 *
 * Do NOT use this to detect "is this value infinite" (`x > BLAS_REAL_MAX`) --
 * that comparison does not survive -ffast-math. See blas_is_inf() below.
 */
#if defined(BLAS_USE_FLOAT)
    #define BLAS_REAL_MAX FLT_MAX
#else
    #define BLAS_REAL_MAX DBL_MAX
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
 * @brief Runtime-robust check for "this BLAS_REAL is +Inf or -Inf",
 * even in a translation unit built with -ffast-math.
 *
 * Do NOT replace this with isinf() or a direct comparison such as
 * `x > BLAS_REAL_MAX` or `x == x * 2` -- none of them survive
 * -ffast-math. That flag implies -ffinite-math-only, under which the
 * compiler is entitled to assume no floating-point value is EVER
 * infinite. This is not just a theoretical risk: confirmed directly
 * against this project's own Release flags (-O3 -march=native
 * -ffast-math) by inspecting the generated assembly, GCC acts on that
 * assumption by dead-code-eliminating an ordinary
 * `if (x > BLAS_REAL_MAX) ...` guard entirely -- not merely folding
 * isinf() to a constant false, but removing the branch outright, even
 * though the runtime value genuinely is infinite.
 *
 * This function instead reinterprets the value's raw bits as an
 * unsigned integer (via memcpy(), the well-defined, strict-aliasing-
 * safe way to do this in C -- NOT a union or pointer cast) and does
 * an ordinary INTEGER comparison against the IEEE 754 bit pattern for
 * infinity (sign bit ignored, since +Inf and -Inf should be treated
 * alike here). Integer comparisons are not covered by
 * -ffinite-math-only's "assume no value is ever infinite" license --
 * the compiler has no equivalent belief about arbitrary integers --
 * so this check cannot be constant-folded or eliminated. Verified
 * (again via the generated assembly, not just reasoned about) to
 * survive this project's Release flags intact.
 *
 * @param x The value to test.
 * @return  Non-zero if @p x is +Inf or -Inf, zero otherwise (including
 *          for NaN and every finite value).
 */
BLAS_INLINE int blas_is_inf(BLAS_REAL x)
{
#if defined(BLAS_USE_FLOAT)
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7FFFFFFFu) == 0x7F800000u;
#else
    uint64_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7FFFFFFFFFFFFFFFULL) == 0x7FF0000000000000ULL;
#endif
}

/**
 * @brief Compute the correct 0-based starting offset for a BLAS
 * stride, for routines that support negative increments (matching
 * reference BLAS's own convention for DAXPY/DDOT).
 *
 * Reference BLAS (e.g. DAXPY's Fortran source) initializes its loop
 * index as `IX = 1; IF (INCX.LT.0) IX = (-N+1)*INCX + 1` (1-based).
 * This is the 0-based equivalent: `kx = (1-n)*incx` for incx < 0,
 * else 0.
 *
 * For incx > 0, this returns 0 -- walk forward from the given
 * pointer, as always. For incx < 0, this returns the offset of the
 * LAST logical element, (n-1)*(-incx), so that starting there and
 * stepping by incx (negative) walks backward through exactly the
 * same n elements the caller's pointer spans, staying within
 * [0, (n-1)*|incx|] throughout -- rather than starting at offset 0
 * and immediately stepping to a negative (out-of-bounds) offset.
 *
 * Only meaningful for callers that have already validated n > 0.
 *
 * @param n    Element count. Must be > 0 -- not validated here.
 * @param incx The stride, positive, negative, or (degenerately) zero.
 * @return     The 0-based offset from the caller's pointer at which
 *             to start iterating: 0 for @p incx >= 0, or the offset of
 *             the last logical element for @p incx < 0.
 */
BLAS_INLINE blas_int blas_stride_start(blas_int n, blas_int incx)
{
    return (incx < 0) ? (1 - n) * incx : 0;
}

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

#endif /* LINE1_TYPES_H */
