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

#include "RAJA/util/Printing.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/for_each.hpp"

#include "cub/util_type.cuh"

namespace RAJA
{

namespace detail
{

template < >
struct StreamInsertHelper<CUDA_DIM_T>
{
  CUDA_DIM_T const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.x
               << "," << m_val.y
               << "," << m_val.z
               << "}";
  }
};

template < typename R >
struct StreamInsertHelper<::cub::DoubleBuffer<R>>
{
  ::cub::DoubleBuffer<R> const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.Current() << "," << m_val.Alternate() << "}";
  }
};

}  // namespace detail

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
                       ::RAJA::ctie(__VA_ARGS__),                         \
                       __FILE__, __LINE__);                               \
  }

template < typename Tuple >
RAJA_INLINE void cudaAssert(cudaError_t code,
                            const char* func_name,
                            const char* args_name,
                            Tuple const& args,
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
        str << ", ";
      } else {
        first = false;
      }
      str << ::RAJA::detail::StreamInsertHelper{arg};
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
