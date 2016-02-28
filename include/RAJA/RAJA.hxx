/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

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


//
//////////////////////////////////////////////////////////////////////
//
// These contents of the header files included here define index set 
// iteration policies whose implementations are compiler-dependent.
//
//////////////////////////////////////////////////////////////////////
//

#if defined(RAJA_COMPILER_ICC)

#include "exec-simd/raja_simd.hxx"
#include "exec-openmp/raja_openmp.hxx"
#include "exec-cilk/raja_cilk.hxx"


#elif defined(RAJA_COMPILER_GNU)


#include "exec-simd/raja_simd.hxx"
#include "exec-openmp/raja_openmp.hxx"


#elif defined(RAJA_COMPILER_XLC12) 

#include "exec-simd/raja_simd.hxx"
#include "exec-openmp/raja_openmp.hxx"


#elif defined(RAJA_COMPILER_CLANG)

#include "exec-simd/raja_simd.hxx"
#include "exec-openmp/raja_openmp.hxx"


#else
#error RAJA compiler macro is undefined!

#endif


#if defined(RAJA_USE_CUDA)

#include "exec-cuda/raja_cuda.hxx"

#endif


//
// All platforms must support sequential execution.  
//
// NOTE: These files include sequential segment iteration over segments in
//       an index set which may require definitions in the above 
//       headers for segment execution.
//
#include "exec-sequential/raja_sequential.hxx"


//
// Generic iteration templates require specializations defined 
// in the files included above.
//
#include "forall_generic.hxx"


#include "IndexSetUtils.hxx"

#endif  // closing endif for header file include guard
