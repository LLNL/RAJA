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

#ifndef RAJA_util_TypeConvert_HPP
#define RAJA_util_TypeConvert_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
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

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"


namespace RAJA
{
namespace util
{
//
// For all of these to work (and frankly the CUDA API) we need to ensure
// that the C++ types are the correct sizes.
//
// If we run into a case where these assertions don't hold, we will need to 
// update our implementation to handle it
//

static_assert(sizeof(unsigned) == 4, "unsigned must be 32-bits");
static_assert(sizeof(unsigned long long) == 8, "unsigned long long must be 64-bits");


/*!
 * Reinterpret any 32-bit datatype as an "unsigned"
 */
template<typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
unsigned reinterp_T_as_u(T const &val){
  static_assert(sizeof(T)==4, "T must be 32-bit");
  return reinterpret_cast<unsigned const volatile &>(val);
}


/*!
 * Reinterpret a "unsigned" as any 32-bit datatype.
 */
template<typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
T reinterp_u_as_T(unsigned const &val){
  static_assert(sizeof(T)==4, "T must be 32-bit");
  return reinterpret_cast<T const &>(val);
}


/*!
 * Reinterpret any 64-bit datatype as an "unsigned long long"
 */
template<typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
unsigned long long reinterp_T_as_ull(T const &val){
  static_assert(sizeof(T)==8, "T must be 64-bit");
  return reinterpret_cast<unsigned long long const volatile &>(val);
}


/*!
 * Reinterpret a "unsigned long long" as any 64-bit datatype.
 */
template<typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
T reinterp_ull_as_T(unsigned long long const &val){
  static_assert(sizeof(T)==8, "T must be 64-bit");
  return reinterpret_cast<T const &>(val);
}




}  // closing brace for util namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
