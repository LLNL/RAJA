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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
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

#ifdef RAJA_HAS_CXX_ATTRIBUTE_DEPRECATED

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

#endif /* RAJA_INTERNAL_DEFINES_HPP */
