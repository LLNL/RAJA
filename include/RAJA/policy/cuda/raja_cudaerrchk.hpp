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
#include <algorithm>
#include <tuple>
#include <array>
#include <string_view>
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
struct StreamInsertHelper<CUDA_DIM_T&>
{
  CUDA_DIM_T& m_val;

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
struct StreamInsertHelper<CUDA_DIM_T const&>
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

///
///////////////////////////////////////////////////////////////////////
///
/// Utility assert method used in CUDA operations to report CUDA
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define RAJA_INTERNAL_CUDA_CHECK_API_CALL(func, ...)                      \
  {                                                                       \
    ::RAJA::detail::cudaAssert(func(__VA_ARGS__),                         \
                       RAJA_STRINGIFY(func),                              \
                       []{static constexpr auto arg_names = ::RAJA::detail::cuda_api_arg_names(RAJA_STRINGIFY(func)); return arg_names; }(), \
                       std::forward_as_tuple(__VA_ARGS__),                \
                       __FILE__, __LINE__);                               \
  }

// returns a space separated string of the arguments to the given function
// returns an empty string if func is unknown
// no leading or trailing spaces
constexpr std::string_view cuda_api_arg_names(std::string_view func)
{
  using storage_type = std::pair<std::string_view, std::string_view>;
  constexpr std::array<storage_type, 18> known_functions{{
    storage_type{"cudaDeviceSynchronize",                         ""},
    storage_type{"cudaGetDevice",                                 "device"},
    storage_type{"cudaGetDeviceProperties",                       "prop device"},
    storage_type{"cudaStreamSynchronize",                         "stream"},
    storage_type{"cudaHostAlloc",                                 "pHost size flags"},
    storage_type{"cudaHostFree",                                  "ptr"},
    storage_type{"cudaMalloc",                                    "devPtr size"},
    storage_type{"cudaFree",                                      "devPtr"},
    storage_type{"cudaMemset",                                    "devPtr value count"},
    storage_type{"cudaMemcpy",                                    "dst src count kind"},
    storage_type{"cudaMemsetAsync",                               "devPtr value count stream"},
    storage_type{"cudaMemcpyAsync",                               "dst src count kind stream"},
    storage_type{"cudaLaunchKernel",                              "func gridDim blockDim args sharedMem stream"},
    storage_type{"cudaPeekAtLastError",                           ""},
    storage_type{"cudaGetLastError",                              ""},
    storage_type{"cudaFuncGetAttributes",                         "attr func"},
    storage_type{"cudaOccupancyMaxPotentialBlockSize",            "minGridSize blockSize func dynamicSMemSize blockSizeLimit"},
    storage_type{"cudaOccupancyMaxActiveBlocksPerMultiprocessor", "numBlocks func blockSize dynamicSMemSize"}
  }};
  for (auto [api_name, api_args] : known_functions) {
    if (func == api_name) {
      return api_args;
    }
  }
  return "";
}

template < typename Tuple >
RAJA_INLINE void cudaAssert(cudaError_t code,
                            std::string_view func_name,
                            std::string_view args_name,
                            Tuple const& args,
                            std::string_view file,
                            int line,
                            bool abort = true)
{
  if (code != cudaSuccess)
  {
    std::ostringstream str;
    str << "CUDA error: ";
    str << cudaGetErrorString(code);
    str << " ";
    str << func_name;
    str << "(";
    const auto args_end = args_name.end();
    ::RAJA::for_each_tuple(args, [&, first=true, args_current=args_name.begin()](auto&& arg) mutable {
      if (!first) {
        str << ", ";
      } else {
        first = false;
      }
      if (args_current != args_end) {
        auto args_current_end = std::find(args_current, args_end, ' ');
        str << std::string_view{args_current, size_t(args_current_end-args_current)} << "=";
        if (args_current_end != args_end) {
          ++args_current_end; // skip space
        }
        args_current = args_current_end;
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

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
