/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing utility methods used in HIP operations.
 *
 *          These methods work only on platforms that support HIP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hiperrchk_HPP
#define RAJA_hiperrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <iostream>
#include <utility>
#include <tuple>
#include <string>
#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include "RAJA/util/Printing.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/for_each.hpp"

#if defined(__HIPCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/types.hpp"
#elif defined(__CUDACC__)
#include "cub/util_type.cuh"
#endif

namespace RAJA
{

namespace detail
{

template < >
struct StreamInsertHelper<HIP_DIM_T&>
{
  HIP_DIM_T& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.x
               << "," << m_val.y
               << "," << m_val.z
               << "}";
  }
};
///
template < >
struct StreamInsertHelper<HIP_DIM_T const&>
{
  HIP_DIM_T const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.x
               << "," << m_val.y
               << "," << m_val.z
               << "}";
  }
};

#if defined(__HIPCC__)
template < typename R >
struct StreamInsertHelper<::rocprim::double_buffer<R>&>
{
  ::rocprim::double_buffer<R>& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.current() << "," << m_val.alternate() << "}";
  }
};
///
template < typename R >
struct StreamInsertHelper<::rocprim::double_buffer<R> const&>
{
  ::rocprim::double_buffer<R> const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.current() << "," << m_val.alternate() << "}";
  }
};
#elif defined(__CUDACC__)
template < typename R >
struct StreamInsertHelper<::cub::DoubleBuffer<R>&>
{
  ::cub::DoubleBuffer<R>& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.Current() << "," << m_val.Alternate() << "}";
  }
};
///
template < typename R >
struct StreamInsertHelper<::cub::DoubleBuffer<R> const&>
{
  ::cub::DoubleBuffer<R> const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.Current() << "," << m_val.Alternate() << "}";
  }
};
#endif

}  // namespace detail

///
///////////////////////////////////////////////////////////////////////
///
/// Utility assert method used in HIP operations to report HIP
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define hipErrchk(func, ...)                                              \
  {                                                                       \
    ::RAJA::hipAssert(func(__VA_ARGS__),                                  \
                      RAJA_STRINGIFY(func),                               \
                      RAJA_STRINGIFY(__VA_ARGS__),                        \
                      std::forward_as_tuple(__VA_ARGS__),                 \
                      __FILE__, __LINE__);                                \
  }

template < typename Tuple >
RAJA_INLINE void hipAssert(hipError_t code,
                           const char* func_name,
                           const char* args_name,
                           Tuple const& args,
                           const char* file,
                           int line,
                           bool abort = true)
{
  if (code != hipSuccess)
  {
    std::ostringstream str;
    str << "HIPassert: ";
    str << hipGetErrorString(code);
    str << " ";
    str << func_name;
    str << "(";
    str << args_name;
    str << ")(";
    for_each_tuple(args, [&, first=true](auto&& arg) mutable {
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

#endif  // closing endif for if defined(RAJA_ENABLE_HIP)

#endif  // closing endif for header file include guard
