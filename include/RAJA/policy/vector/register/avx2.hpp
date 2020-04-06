/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX2
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX2__

#ifndef RAJA_policy_vector_register_avx2_HPP
#define RAJA_policy_vector_register_avx2_HPP

#include<RAJA/pattern/vector.hpp>

namespace RAJA {
  struct avx2_register {};
}

#endif // guard

#include<RAJA/policy/vector/register/avx2/avx2_int32.hpp>
#include<RAJA/policy/vector/register/avx2/avx2_int64.hpp>
#include<RAJA/policy/vector/register/avx2/avx2_float.hpp>
#include<RAJA/policy/vector/register/avx2/avx2_double.hpp>


#endif // __AVX2__
