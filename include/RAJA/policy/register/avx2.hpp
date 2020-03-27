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

#include<RAJA/pattern/register.hpp>

namespace RAJA {
  struct avx2_register {};

  template<typename T>
  struct RegisterTraits<avx2_register, T>{
      using register_type = avx2_register;
      using element_type = T;

      static constexpr size_t s_bit_width = 256;
      static constexpr size_t s_byte_width = 32;
      static constexpr size_t s_num_elem = s_byte_width / sizeof(T);

  };


}

#endif // guard

#include<RAJA/policy/register/avx2/avx2_int32.hpp>
#include<RAJA/policy/register/avx2/avx2_int64.hpp>
#include<RAJA/policy/register/avx2/avx2_float.hpp>
#include<RAJA/policy/register/avx2/avx2_double.hpp>


#endif // __AVX2__
