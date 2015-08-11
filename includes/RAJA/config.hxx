/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for basic RAJA configuration.
 * 
 *          This file contains platform-specific parameters that control 
 *          aspects of compilation of application code using RAJA. These
 *          parameters specify: SIMD unit width, data alignment information,
 *          inline directives, etc. 
 *
 *          Items in this file can be set by editing the file or specified 
 *          using "-D" compilation definitions.
 *
 *          IMPORTANT: Exactly one RAJA_PLATFORM_* and one RAJA_COMPILER_* 
 *                     option must be defined to ensure correct behavior.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL 
 *
 ******************************************************************************
 */

#ifndef RAJA_config_HXX
#define RAJA_config_HXX


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief RAJA software version number.
 *
 ******************************************************************************
 */
#define RAJA_VERSION_MAJOR 1
#define RAJA_VERSION_MINOR 0
#define RAJA_VERSION_PATCHLEVEL 0


//
//  Platform-specific constants for range index set and data alignment:
//
//     RANGE_ALIGN - alignment of begin/end indices in range segments
//                   (i.e., starting index and length of range segments 
//                    constructed by index set builder methods will 
//                    be multiples of this value; units of "real" data type
//
//     RANGE_MIN_LENGTH - used in index set builder methods
//                        as min length of range segments (an integer multiple
//                        of RANGE_ALIGN); units of "real" data type
//
//     DATA_ALIGN - used in compiler-specific intrinsics and typedefs
//                  to specify alignment of data, loop bounds, etc.;
//                  units of "bytes" 


#if defined(RAJA_PLATFORM_X86_SSE)
//
// Configuration for Intel platforms with SSE vector instructions.
//
const int RANGE_ALIGN = 4;

#if defined(RAJA_USE_CUDA)
const int RANGE_MIN_LENGTH = 32;
#else
const int RANGE_MIN_LENGTH = 2*RANGE_ALIGN;
#endif
#define COHERENCE_BLOCK_SIZE 64

const int DATA_ALIGN = 32;


#elif defined(RAJA_PLATFORM_X86_AVX)
//
// Configuration for Intel platforms with AVX vector instructions.
//
const int RANGE_ALIGN = 4;

#if defined(RAJA_USE_CUDA)
const int RANGE_MIN_LENGTH = 32;
#else
const int RANGE_MIN_LENGTH = 2*RANGE_ALIGN;
#endif
#define COHERENCE_BLOCK_SIZE 64

const int DATA_ALIGN = 32;

#elif defined(RAJA_PLATFORM_BGQ)
//
// Configuration for IBM BG/Q systems (e.g., sequoia)
//

const int RANGE_ALIGN = 4;

const int RANGE_MIN_LENGTH = 2*RANGE_ALIGN;

const int DATA_ALIGN = 32;

#define COHERENCE_BLOCK_SIZE 64

#else
#error RAJA platform is undefined!

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

#if defined(RAJA_COMPILER_ICC)
//
// Configuration options for Intel compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(RAJA_USE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else

#if __ICC < 1300  // use alignment intrinsic
#define RAJA_ALIGN_DATA(d) __assume_aligned(d, DATA_ALIGN)
#else
#define RAJA_ALIGN_DATA(d)  // TODO: Define this...
#endif

#endif

#define RAJA_SIMD  // TODO: Define this...


#elif defined(RAJA_COMPILER_GNU) 
//
// Configuration options for GNU compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(RAJA_USE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else

#define RAJA_ALIGN_DATA(d) __builtin_assume_aligned(d, DATA_ALIGN)

#endif

#define RAJA_SIMD  // TODO: Define this...


#elif defined(RAJA_COMPILER_XLC12)
//
// Configuration options for xlc v12 compiler (i.e., bgq/sequoia).
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#define RAJA_ALIGN_DATA(d) __alignx(DATA_ALIGN, d)

//#define RAJA_SIMD  _Pragma("simd_level(10)")
#define RAJA_SIMD   // TODO: Define this... 


#elif defined(RAJA_COMPILER_CLANG)
//
// Configuration options for clang compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(RAJA_USE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else

#define RAJA_ALIGN_DATA(d) // TODO: Define this...

#endif

#define RAJA_SIMD  // TODO: Define this...


#else
#error RAJA compiler is undefined!

#endif


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
