/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Main RAJA header file.
 *     
 *          This is the main header file to include in code that uses RAJA.
 *          It includes other RAJA headers files that define types, index
 *          sets, ieration methods, etc.
 *
 *          IMPORTANT: If changes are made to this file, note that contents
 *                     of some header files require that they are included
 *                     in the order found here.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_HXX
#define RAJA_HXX


#include "config.hxx"

#include "int_datatypes.hxx"
#include "real_datatypes.hxx"

#include "execpolicy.hxx"

#include "reducers.hxx"

#include "RangeSegment.hxx"
#include "ListSegment.hxx"
#include "IndexSet.hxx"

#include "IndexSetUtils.hxx"

//
//////////////////////////////////////////////////////////////////////
//
// These contents of the header files included here define index set 
// iteration policies whose implementations are compiler-dependent.
//
//////////////////////////////////////////////////////////////////////
//

#if defined(RAJA_COMPILER_ICC)

#include "forall_simd.hxx"
#include "forall_omp.hxx"
#include "forall_cilk.hxx"


#elif defined(RAJA_COMPILER_GNU)


#include "forall_simd.hxx"
#include "forall_omp.hxx"


#elif defined(RAJA_COMPILER_XLC12) 

#include "forall_simd.hxx"
#include "forall_omp.hxx"


#elif defined(RAJA_COMPILER_CLANG)

#include "forall_simd.hxx"
#include "forall_omp.hxx"


#else
#error RAJA compiler macro is undefined!

#endif


#if defined(RAJA_USE_CUDA)

#include "forall_cuda.hxx"

#endif


//
// All platforms must support sequential execution.  
//
// NOTE: This file includes sequential segment iteration over segments in
//       an index set which may require definitions in the above 
//       headers for segment execution.
//
#include "forall_seq.hxx"


//
// Generic iteration templates that require specializations defined 
// in the files included above.
//
#include "forall_generic.hxx"


#endif  // closing endif for header file include guard
