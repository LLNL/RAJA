/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX__

#ifndef RAJA_policy_vector_register_avx_HPP
#define RAJA_policy_vector_register_avx_HPP

#include<RAJA/pattern/vector.hpp>

namespace RAJA {
  struct avx_register {};
}


#endif

#include<RAJA/policy/vector/register/avx/avx_int64.hpp>
#include<RAJA/policy/vector/register/avx/avx_int32.hpp>
#include<RAJA/policy/vector/register/avx/avx_float.hpp>
#include<RAJA/policy/vector/register/avx/avx_double.hpp>


#endif // __AVX__
