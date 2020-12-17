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

namespace RAJA {
  struct avx_register {};

  template<typename T>
  struct RegisterTraits<avx_register, T>{
      using element_type = T;
      using register_policy = avx_register;
      static constexpr camp::idx_t s_num_bits = 256;
      static constexpr camp::idx_t s_num_elem = s_num_bits / 8 / sizeof(T);
  };
}


#endif

#include<RAJA/policy/tensor/arch/avx/avx_int64.hpp>
#include<RAJA/policy/tensor/arch/avx/avx_int32.hpp>
#include<RAJA/policy/tensor/arch/avx/avx_float.hpp>
#include<RAJA/policy/tensor/arch/avx/avx_double.hpp>


#endif // __AVX__
