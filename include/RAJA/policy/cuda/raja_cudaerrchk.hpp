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

#ifndef RAJA_raja_cudaerrchk_HPP
#define RAJA_raja_cudaerrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "RAJA/util/defines.hpp"

namespace RAJA
{

///
///////////////////////////////////////////////////////////////////////
///
/// Utility assert method used in CUDA operations to report CUDA
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define cudaErrchk(ans)                            \
  {                                                \
    ::RAJA::cudaAssert((ans), __FILE__, __LINE__); \
  }

inline void cudaAssert(cudaError_t code,
                       const char *file,
                       int line,
                       bool abort = true)
{
  if (code != cudaSuccess) {
    fprintf(
        stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) RAJA_ABORT_OR_THROW("CUDAassert");
  }
}

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
