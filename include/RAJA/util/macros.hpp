/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for common RAJA internal macro definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_INTERNAL_MACROS_HPP
#define RAJA_INTERNAL_MACROS_HPP

#include "RAJA/config.hpp"

#include <cstdlib>
#include <stdexcept>
#include <stdio.h>

#if defined(RAJA_HIP_ACTIVE)
#include <hip/hip_runtime.h>
#endif

//
// Macros for decorating host/device functions for CUDA and HIP kernels.
// We need a better solution than this as it is a pain to manage
// this stuff in an application.
//
#if (defined(RAJA_ENABLE_CUDA) && defined(__CUDA_ARCH__)) \
  || (defined(RAJA_ENABLE_HIP) && defined(__HIP_DEVICE_COMPILE__)) \
  || (defined(RAJA_ENABLE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
#define RAJA_DEVICE_CODE
#endif

#if defined(RAJA_ENABLE_CUDA) && defined(__CUDACC__)
#define RAJA_HOST_DEVICE __host__ __device__
#define RAJA_DEVICE __device__
#define RAJA_HOST __host__

#if defined(RAJA_ENABLE_CLANG_CUDA)
#define RAJA_SUPPRESS_HD_WARN
#else
#define RAJA_SUPPRESS_HD_WARN RAJA_PRAGMA(nv_exec_check_disable)
#endif

#elif defined(RAJA_ENABLE_HIP) && defined(__HIPCC__)
#define RAJA_HOST_DEVICE __host__ __device__
#define RAJA_DEVICE __device__
#define RAJA_HOST __host__
#define RAJA_SUPPRESS_HD_WARN

#else

#define RAJA_HOST_DEVICE
#define RAJA_DEVICE
#define RAJA_HOST
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
template <typename... T>
RAJA_HOST_DEVICE RAJA_INLINE void RAJA_UNUSED_VAR(T &&...) noexcept
{
}

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


RAJA_HOST_DEVICE
inline void RAJA_ABORT_OR_THROW(const char *str)
{
#if defined(__SYCL_DEVICE_ONLY__)
  abort();
#else
  printf ( "%s\n", str );
#if defined(RAJA_ENABLE_TARGET_OPENMP) && (_OPENMP >= 201511)
  // seg faulting here instead of calling std::abort for omp target
  const char * errtemp = nullptr;
  errtemp = str;
#elif defined(__CUDA_ARCH__)
  asm ("trap;");

#elif defined(__HIP_DEVICE_COMPILE__)
  abort();

#else
#ifdef RAJA_COMPILER_MSVC
  fflush(stdout);
  char *value;
  size_t len;
  bool no_except = false;
  if(_dupenv_s(&value, &len, "RAJA_NO_EXCEPT") == 0 && value != nullptr){
    no_except = true;
    free(value);
  }

#else
  bool no_except = std::getenv("RAJA_NO_EXCEPT") != nullptr;
#endif

  fflush(stdout);
  if (no_except) {
    std::abort();
  } else {
    throw std::runtime_error(str);
  }
#endif
#endif
}

//! Macros for marking deprecated features in RAJA
/*!
 * To deprecate a function, place immediately before the return type
 * To deprecate a member of a class or struct, place immediately before the
 * declaration
 * To deprecate a typedef, place immediately before the declaration.
 * To deprecate a struct or class, place immediately after the 'struct'/'class'
 * keyword
 */

#if (__cplusplus >= 201402L)
#define RAJA_HAS_CXX14 1
#define RAJA_HAS_CXX_ATTRIBUTE_DEPRECATED 1
#elif defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define RAJA_HAS_CXX_ATTRIBUTE_DEPRECATED 1
#endif
#endif

#if defined(RAJA_HAS_CXX_ATTRIBUTE_DEPRECATED)
// When using a C++14 compiler, use the standard-specified deprecated attribute
#define RAJA_DEPRECATE(Msg) [[deprecated(Msg)]]
#define RAJA_DEPRECATE_ALIAS(Msg) [[deprecated(Msg)]]

#elif defined(_MSC_VER)

// for MSVC, use __declspec
#define RAJA_DEPRECATE(Msg) __declspec(deprecated(Msg))
#define RAJA_DEPRECATE_ALIAS(Msg)

#else

// else use __attribute__(deprecated("Message"))
#define RAJA_DEPRECATE(Msg) __attribute__((deprecated(Msg)))
#define RAJA_DEPRECATE_ALIAS(Msg)

#endif

#endif /* RAJA_INTERNAL_MACROS_HPP */
