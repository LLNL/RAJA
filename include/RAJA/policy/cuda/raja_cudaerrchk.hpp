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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cudaerrchk_HPP
#define RAJA_cudaerrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "RAJA/util/macros.hpp"

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
#define cudaErrchk(ans)                                                        \
  {                                                                            \
    ::RAJA::cudaAssert((ans), __FILE__, __LINE__);                             \
  }

inline void
cudaAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    if (abort)
    {
      std::string msg;
      msg += "CUDAassert: ";
      msg += cudaGetErrorString(code);
      msg += " ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      throw std::runtime_error(msg);
    }
    else
    {
      fprintf(
          stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file,
          line);
    }
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
