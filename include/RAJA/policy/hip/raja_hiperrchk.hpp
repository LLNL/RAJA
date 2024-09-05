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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hiperrchk_HPP
#define RAJA_hiperrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <iostream>
#include <string>

#include <hip/hip_runtime.h>

#include "RAJA/util/macros.hpp"

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
#define hipErrchk(ans)                                                         \
  {                                                                            \
    ::RAJA::hipAssert((ans), __FILE__, __LINE__);                              \
  }

inline void
hipAssert(hipError_t code, const char* file, int line, bool abort = true)
{
  if (code != hipSuccess)
  {
    if (abort)
    {
      std::string msg;
      msg += "HIPassert: ";
      msg += hipGetErrorString(code);
      msg += " ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      throw std::runtime_error(msg);
    }
    else
    {
      fprintf(stderr, "HIPassert: %s %s %d\n", hipGetErrorString(code), file,
              line);
    }
  }
}

} // namespace RAJA

#endif // closing endif for if defined(RAJA_ENABLE_HIP)

#endif // closing endif for header file include guard
