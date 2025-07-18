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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cudaerrchk_HPP
#define RAJA_cudaerrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <iostream>
#include <utility>
#include <tuple>
#include <string>
#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/for_each.hpp"

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
#define cudaErrchk(func, ...)                                             \
  {                                                                       \
    ::RAJA::cudaAssert(func(__VA_ARGS__),                                 \
                       RAJA_STRINGIFY(func),                              \
                       RAJA_STRINGIFY(__VA_ARGS__),                       \
                       std::tie(__VA_ARGS__),                             \
                       __FILE__, __LINE__);                               \
  }

inline void cudaAssert(cudaError_t code,
                       const char* file,
                       int line,
                       bool abort = true)
{
  if (code != cudaSuccess)
  {
    std::ostringstream str;
    str << "CUDAassert: ";
    str << cudaGetErrorString(code);
    str << " ";
    str << func_name;
    str << "(";
    str << args_name;
    str << ")(";
    for_each_tuple(args, [&, first=true](auto const& arg) mutable {
      if (!first) {
        str << " ";
      } else {
        first = false;
      }
      str << arg;
    });
    str << ") ";
    str << file;
    str << ":";
    str << line;
    auto msg{str.str()};
    if (abort)
    {
      throw std::runtime_error(msg);
    }
    else
    {
      std::cerr << msg;
    }
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
