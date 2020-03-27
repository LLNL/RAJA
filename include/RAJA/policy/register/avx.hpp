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

#include<RAJA/pattern/register.hpp>

namespace RAJA {
  struct avx_register {};

  template<typename T>
  struct RegisterTraits<avx_register, T>{
      using register_type = avx_register;
      using element_type = T;


      static constexpr size_t s_bit_width = 256;
      static constexpr size_t s_byte_width = 32;
      static constexpr size_t s_num_elem = s_byte_width / sizeof(T);

  };

}


#endif

#include<RAJA/policy/register/avx/avx_int64.hpp>
#include<RAJA/policy/register/avx/avx_int32.hpp>
#include<RAJA/policy/register/avx/avx_float.hpp>
#include<RAJA/policy/register/avx/avx_double.hpp>


#endif // __AVX__
