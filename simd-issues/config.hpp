/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for basic RAJA configuration options.
 *
 *          This file contains platform-specific parameters that control
 *          aspects of compilation of application code using RAJA. These
 *          parameters specify: SIMD unit width, data alignment information,
 *          inline directives, etc.
 *
 *          IMPORTANT: These options are set by CMake and depend on the options
 *          passed to it.
 *
 *          IMPORTANT: Exactly one e RAJA_COMPILER_* option must be defined to
 *          ensure correct behavior.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_config_HPP
#define RAJA_config_HPP

#cmakedefine ENABLE_FT

#define @RAJA_FP@
#define @RAJA_PTR@

/*
 * Programming models
 */

#cmakedefine RAJA_ENABLE_OPENMP
#cmakedefine RAJA_ENABLE_TARGET_OPENMP
#cmakedefine RAJA_ENABLE_TBB
#cmakedefine RAJA_ENABLE_CUDA
#cmakedefine RAJA_ENABLE_CLANG_CUDA
#cmakedefine RAJA_ENABLE_CHAI

/*
 * Timer options
 */

#cmakedefine RAJA_USE_CHRONO
#cmakedefine RAJA_USE_GETTIME
#cmakedefine RAJA_USE_CLOCK
#cmakedefine RAJA_USE_CYCLE

/*
 * Detect the host C++ compiler we are using.
 */
#if defined(__INTEL_COMPILER)
#define RAJA_COMPILER_INTEL
#elif defined(__xlc__)
#define RAJA_COMPILER_XLC
#elif defined(__clang__)
#define RAJA_COMPILER_CLANG
#elif defined(__PGI)
#define RAJA_COMPILER_PGI
#elif defined(_WIN32)
#define RAJA_COMPILER_MSVC
#elif defined(__GNUC__)
#define RAJA_COMPILER_GNU
#endif


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief RAJA software version number.
 *
 ******************************************************************************
 */
const int RAJA_VERSION_MAJOR = @RAJA_VERSION_MAJOR@;
const int RAJA_VERSION_MINOR = @RAJA_VERSION_MINOR@;
const int RAJA_VERSION_PATCHLEVEL = @RAJA_VERSION_PATCHLEVEL@;


/*!
 ******************************************************************************
 *
 * \brief Useful macros.
 *
 ******************************************************************************
 */

//
//  Platform-specific constants for range index set and data alignment:
//
//     RANGE_ALIGN - alignment of begin/end indices in range segments
//                   (i.e., starting index and length of range segments
//                    constructed by index set builder methods will
//                    be multiples of this value)
//
//     RANGE_MIN_LENGTH - used in index set builder methods
//                        as min length of range segments (an integer multiple
//                        of RANGE_ALIGN)
//
//     DATA_ALIGN - used in compiler-specific intrinsics and typedefs
//                  to specify alignment of data, loop bounds, etc.;
//                  units of "bytes"

const int RANGE_ALIGN = @RAJA_RANGE_ALIGN@;
const int RANGE_MIN_LENGTH = @RAJA_RANGE_MIN_LENGTH@;
const int DATA_ALIGN = @RAJA_DATA_ALIGN@;

#if defined (_WIN32)
#define RAJA_RESTRICT __restrict
#else
#define RAJA_RESTRICT __restrict__
#endif


//
//  Compiler-specific definitions for inline directives, data alignment
//  intrinsics, and SIMD vector pragmas
//
//  Variables for compiler instrinsics, directives, typedefs
//
//     RAJA_INLINE - macro to enforce method inlining
//
//     RAJA_ALIGN_DATA(<variable>) - macro to express alignment of data,
//                              loop bounds, etc.
//
//     RAJA_SIMD - macro to express SIMD vectorization pragma to force
//                 loop vectorization
//
//     RAJA_ALIGNED_ATTR(<alignment>) - macro to express type or variable alignments
//

#if defined(RAJA_COMPILER_GNU)
#define RAJA_ALIGNED_ATTR(N) __attribute__((aligned(N)))
#else
#define RAJA_ALIGNED_ATTR(N) alignas(N)
#endif


#if defined(RAJA_COMPILER_INTEL)
//
// Configuration options for Intel compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(ENABLE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else
#define RAJA_ALIGN_DATA(d) __assume_aligned(d, DATA_ALIGN)
#endif

#if defined(_OPENMP) && (_OPENMP >= 201307)
#define RAJA_SIMD  _Pragma("omp simd")
#define RAJA_NO_SIMD _Pragma("novector")
#else
#define RAJA_SIMD _Pragma("simd")
#define RAJA_NO_SIMD  _Pragma("novector")
#endif


#elif defined(RAJA_COMPILER_GNU)
//
// Configuration options for GNU compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(ENABLE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else
#define RAJA_ALIGN_DATA(d) __builtin_assume_aligned(d, DATA_ALIGN)
#endif

#if defined(_OPENMP) && (_OPENMP >= 201307)
#define RAJA_SIMD  _Pragma("omp simd")
#define RAJA_NO_SIMD 
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && \
      ( ( (__GNUC__ == 4) && (__GNUC_MINOR__ == 9) ) || (__GNUC__ >= 5) )
#define RAJA_SIMD    _Pragma("GCC ivdep")
#define RAJA_NO_SIMD 
#else
#define RAJA_SIMD
#define RAJA_NO_SIMD
#endif


#elif defined(RAJA_COMPILER_XLC)
//
// Configuration options for xlc compiler (i.e., bgq/sequoia).
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#define RAJA_ALIGN_DATA(d) __alignx(DATA_ALIGN, d)

//#define RAJA_SIMD  _Pragma("simd_level(10)")
#if defined(_OPENMP) && (_OPENMP >= 201307)
#define RAJA_SIMD  _Pragma("omp simd")
#define RAJA_NO_SIMD _Pragma("nosimd")
#else
// This pragma is unreliable.  It may not work on Blue Gene/Q or POWER7.
#define RAJA_SIMD  _Pragma("ibm independent_loop")
#define RAJA_NO_SIMD  _Pragma("nosimd")
#endif


#elif defined(RAJA_COMPILER_CLANG)
//
// Configuration options for clang compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(ENABLE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else
#define RAJA_ALIGN_DATA(d) __builtin_assume_aligned(d, DATA_ALIGN)
#endif

#if defined(_OPENMP) && (_OPENMP >= 201307)
#define RAJA_SIMD  _Pragma("omp simd")
#define RAJA_NO_SIMD _Pragma("clang loop vectorize(disable)")
#else

#if ( (__clang_major__ >= 4 ) ||  (__clang_major__ >= 3 && __clang_minor__ > 7) )
#define RAJA_SIMD    _Pragma("clang loop vectorize(assume_safety)")
#else 
#define RAJA_SIMD    _Pragma("clang loop vectorize(enable)")
#endif

#define RAJA_NO_SIMD  _Pragma("clang loop vectorize(disable)")
#endif

#else

#pragma message("RAJA_COMPILER unknown, using default empty macros.")

#define RAJA_INLINE inline
#define RAJA_ALIGN_DATA(d)
#define RAJA_SIMD
#define RAJA_NO_SIMD

#endif

#cmakedefine RAJA_HAVE_POSIX_MEMALIGN
#cmakedefine RAJA_HAVE_ALIGNED_ALLOC
#cmakedefine RAJA_HAVE_MM_MALLOC

}  // closing brace for RAJA namespace

#endif // closing endif for header file include guard
