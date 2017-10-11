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

#ifndef RAJA_util_TypeConvert_HPP
#define RAJA_util_TypeConvert_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"


namespace RAJA
{
namespace util
{


/*!
 * Reinterpret any datatype as another datatype of the same size
 */
template<typename A, typename B>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
B reinterp_A_as_B(A const &val){
  static_assert(sizeof(A)==sizeof(B), "A and B must be same size");
  return reinterpret_cast<B const volatile &>(val);
}

template<typename A, typename B>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
B reinterp_A_as_B(A volatile const &val){
  static_assert(sizeof(A)==sizeof(B), "A and B must be same size");
  return reinterpret_cast<B const volatile &>(val);
}


}  // closing brace for util namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
