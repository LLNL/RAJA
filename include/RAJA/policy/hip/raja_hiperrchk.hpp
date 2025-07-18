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

#include "RAJA/util/macros.hpp"
#include "RAJA/util/for_each.hpp"

namespace RAJA
{

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
                      std::tie(__VA_ARGS__),                              \
                      __FILE__, __LINE__);                                \
  }

template < typename... Ts >
RAJA_INLINE void hipAssert(const char* file,
                      int line,
                      bool abort,
                      int code,
                      std::pair<const char*, Ts> const& ...args
                      )
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

#endif  // closing endif for if defined(RAJA_ENABLE_HIP)

#endif  // closing endif for header file include guard
