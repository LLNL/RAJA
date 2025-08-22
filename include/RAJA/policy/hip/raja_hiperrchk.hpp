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
#include <algorithm>
#include <tuple>
#include <array>
#include <string_view>
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
                      []{static constexpr auto arg_names = ::RAJA::hip_api_arg_names(RAJA_STRINGIFY(func)); return arg_names; }(), \
                      std::forward_as_tuple(__VA_ARGS__),                 \
                      __FILE__, __LINE__);                                \
  }

// returns a space separated string of the arguments to the given function
// returns an empty string if func is unknown
// no leading or trailing spaces
constexpr std::string_view hip_api_arg_names(std::string_view func)
{
  using storage_type = std::pair<std::string_view, std::string_view>;
  constexpr std::array<storage_type, 19> known_functions{{
    storage_type{"hipDeviceSynchronize",                         ""},
    storage_type{"hipGetDevice",                                 "device"},
    storage_type{"hipGetDeviceProperties",                       "prop device"},
    storage_type{"hipGetDevicePropertiesR0600",                  "prop device"},
    storage_type{"hipStreamSynchronize",                         "stream"},
    storage_type{"hipHostMalloc",                                "pHost size flags"},
    storage_type{"hipHostFree",                                  "ptr"},
    storage_type{"hipMalloc",                                    "devPtr size"},
    storage_type{"hipFree",                                      "devPtr"},
    storage_type{"hipMemset",                                    "devPtr value count"},
    storage_type{"hipMemcpy",                                    "dst src count kind"},
    storage_type{"hipMemsetAsync",                               "devPtr value count stream"},
    storage_type{"hipMemcpyAsync",                               "dst src count kind stream"},
    storage_type{"hipLaunchKernel",                              "func gridDim blockDim args sharedMem stream"},
    storage_type{"hipPeekAtLastError",                           ""},
    storage_type{"hipGetLastError",                              ""},
    storage_type{"hipFuncGetAttributes",                         "attr func"},
    storage_type{"hipOccupancyMaxPotentialBlockSize",            "minGridSize blockSize func dynamicSMemSize blockSizeLimit"},
    storage_type{"hipOccupancyMaxActiveBlocksPerMultiprocessor", "numBlocks func blockSize dynamicSMemSize"}
  }};
  for (auto [api_name, api_args] : known_functions) {
    if (func == api_name) {
      return api_args;
    }
  }
  return "";
}

template < typename Tuple >
RAJA_INLINE void hipAssert(hipError_t code,
                           std::string_view func_name,
                           std::string_view args_name,
                           Tuple const& args,
                           std::string_view file,
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
    const auto args_end = args_name.end();
    for_each_tuple(args, [&, first=true, args_current=args_name.begin()](auto&& arg) mutable {
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

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_HIP)

#endif  // closing endif for header file include guard
