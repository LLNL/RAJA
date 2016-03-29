/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing utility methods used in CUDA operations.
 *
 *          These methods work only on platforms that support CUDA. 
 *
 ******************************************************************************
 */

#ifndef RAJA_raja_cudaerrchk_HXX
#define RAJA_raja_cudaerrchk_HXX

#if defined(RAJA_USE_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>


namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Utility method used in CUDA operations.
//
//////////////////////////////////////////////////////////////////////
//
#define cudaErrchk(ans) { RAJA::cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, 
                       bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDAassert: %s %s %d\n", 
              cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

}  // closing brace for RAJA namespace


#endif  // closing endif for if defined(RAJA_USE_CUDA)

#endif  // closing endif for header file include guard

