/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for common RAJA internal definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_INTERNAL_DEFINES_HPP
#define RAJA_INTERNAL_DEFINES_HPP

#include "RAJA/config.hpp"

#include <cstdlib>
#include <stdexcept>

//
// Macros for decorating host/device functions for CUDA kernels.
// We need a better solution than this as it is a pain to manage
// this stuff in an application.
//
#if defined(RAJA_ENABLE_CUDA) && defined(__CUDACC__)

#define RAJA_HOST_DEVICE __host__ __device__
#define RAJA_DEVICE __device__

#if defined(RAJA_ENABLE_CLANG_CUDA)
#define RAJA_SUPPRESS_HD_WARN
#else
#if defined(_WIN32)  // windows is non-compliant, yay
#define RAJA_SUPPRESS_HD_WARN __pragma(nv_exec_check_disable)
#else
#define RAJA_SUPPRESS_HD_WARN _Pragma("nv_exec_check_disable")
#endif
#endif

#else

#define RAJA_HOST_DEVICE
#define RAJA_DEVICE
#define RAJA_SUPPRESS_HD_WARN
#endif

/*!
 *******************************************************************************
 * \def RAJA_USED_ARG(x)
 *
 * \brief Macro for silencing compiler warnings in methods with unused
 *  arguments.
 *
 * \note The intent is to use this macro in the function signature. For example:
 *
 * \code
 *
 *  void my_function(int x, int RAJA_UNUSED_ARG(y))
 *  {
 *    // my implementation that doesn't use 'y'
 *  }
 *
 * \endcode
 *******************************************************************************
 */
#define RAJA_UNUSED_ARG(x)

/*!
 *******************************************************************************
 * \def RAJA_UNUSED_VAR(x)
 *
 * \brief Macro used to silence compiler warnings about variables
 *        that are defined but not used.
 *
 * \iote The intent is to use this macro for variables that are only used
 *       for debugging purposes (e.g. in debug assertions). For example:
 *
 * \code
 *
 *  double myVar = ...
 *
 *  cassert(myVar > 0)  // variable used in assertion that may be compiled out
 *  RAJA_UNUSED_VAR(myVar);
 *
 * \endcode
 *******************************************************************************
 */
template < typename... T >
RAJA_HOST_DEVICE RAJA_INLINE void RAJA_UNUSED_VAR(T&&...) noexcept {}

/*!
 * \def RAJA_STRINGIFY_HELPER(x)
 *
 * Helper for RAJA_STRINGIFY_MACRO
 */
#define RAJA_STRINGIFY_HELPER(x) #x

/*!
 * \def RAJA_STRINGIFY_MACRO(x)
 *
 * Used in static_assert macros to print values of defines
 */
#define RAJA_STRINGIFY_MACRO(x) RAJA_STRINGIFY_HELPER(x)

#define RAJA_DIVIDE_CEILING_INT(dividend, divisor) \
  (((dividend) + (divisor)-1) / (divisor))


inline void RAJA_ABORT_OR_THROW(const char *str)
{
  if (std::getenv("RAJA_NO_EXCEPT") != nullptr) {
    std::abort();
  } else {
    throw std::runtime_error(str);
  }
}

//! Macros for marking deprecated features in RAJA
/*!
 * To deprecate a function, place immediately before the return type
 * To deprecate a member of a class or struct, place immediately before the declaration
 * To deprecate a typedef, place immediately before the declaration.
 * To deprecate a struct or class, place immediately after the 'struct'/'class' keyword
 */

#if ( __cplusplus >= 201402L )
# define RAJA_HAS_CXX14 1
# define RAJA_HAS_CXX_ATTRIBUTE_DEPRECATED 1
#elif defined(__has_cpp_attribute)
# if __has_cpp_attribute(deprecated)
#  define RAJA_HAS_CXX_ATTRIBUTE_DEPRECATED 1
# endif
#endif

#ifdef RAJA_HAS_CXX_ATTRIBUTE_DEPRECATED

// When using a C++14 compiler, use the standard-specified deprecated attribute
# define RAJA_DEPRECATE(Msg) [[deprecated(Msg)]]
# define RAJA_DEPRECATE_ALIAS(Msg) [[deprecated(Msg)]]

#elif defined(_MSC_VER)

// for MSVC, use __declspec
# define RAJA_DEPRECATE(Msg) __declspec(deprecated(Msg))
# define RAJA_DEPRECATE_ALIAS(Msg)

#else

// else use __attribute__(deprecated("Message"))
#define RAJA_DEPRECATE(Msg) __attribute__((deprecated(Msg)))
#define RAJA_DEPRECATE_ALIAS(Msg)

#endif

#endif /* RAJA_INTERNAL_DEFINES_HPP */
