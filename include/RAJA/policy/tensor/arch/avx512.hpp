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

namespace RAJA {
  struct avx512_register {};

  template<typename T>
  struct RegisterTraits<avx512_register, T>{
      using element_type = T;
      using register_policy = avx512_register;
      static constexpr camp::idx_t s_num_bits = 512;
      static constexpr camp::idx_t s_num_elem = s_num_bits / 8 / sizeof(T);
  };
}

#endif // guard

#include<RAJA/policy/tensor/arch/avx512/avx512_int32.hpp>
#include<RAJA/policy/tensor/arch/avx512/avx512_int64.hpp>
#include<RAJA/policy/tensor/arch/avx512/avx512_float.hpp>
#include<RAJA/policy/tensor/arch/avx512/avx512_double.hpp>


#endif // __AVX512F__
