/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for reinterpreting type conversions.
 *
 *          These conversions are needed to pass N-bit floating point values
 *          as integral types for certain API's that have limited type support.
 *          These conversions are used heavily by the atomic operators.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_util_TypeConvert_HPP
#define RAJA_util_TypeConvert_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"


namespace RAJA
{
namespace util
{


/*!
 * Reinterpret any datatype as another datatype of the same size
 */
template <typename A, typename B>
RAJA_INLINE RAJA_HOST_DEVICE constexpr B reinterp_A_as_B(A const &val)
{
  static_assert(sizeof(A) == sizeof(B), "A and B must be same size");
  return reinterpret_cast<B const volatile &>(val);
}

template <typename A, typename B>
RAJA_INLINE RAJA_HOST_DEVICE constexpr B reinterp_A_as_B(A volatile const &val)
{
  static_assert(sizeof(A) == sizeof(B), "A and B must be same size");
  return reinterpret_cast<B const volatile &>(val);
}


}  // namespace util
}  // namespace RAJA

#endif  // closing endif for header file include guard
