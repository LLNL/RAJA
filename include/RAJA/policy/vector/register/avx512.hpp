/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX512
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Check if the base AVX512 instructions are present
#ifdef __AVX512F__

#ifndef RAJA_policy_vector_register_avx512_HPP
#define RAJA_policy_vector_register_avx512_HPP

#include<RAJA/pattern/vector.hpp>

namespace RAJA {
  struct avx512_register {};
}

#endif // guard

//#include<RAJA/policy/vector/register/avx512/avx512_int32.hpp>
//#include<RAJA/policy/vector/register/avx512/avx512_int64.hpp>
#include<RAJA/policy/vector/register/avx512/avx512_float.hpp>
#include<RAJA/policy/vector/register/avx512/avx512_double.hpp>


#endif // __AVX512F__
