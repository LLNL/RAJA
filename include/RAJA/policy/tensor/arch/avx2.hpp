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

namespace RAJA {

  template<typename T>
  struct RegisterTraits<avx2_register, T>{
      using element_type = T;
      using register_policy = avx2_register;
      static constexpr camp::idx_t s_num_bits = 256;
      static constexpr camp::idx_t s_num_elem = s_num_bits / 8 / sizeof(T);
  };
}

#endif // guard

#include<RAJA/policy/tensor/arch/avx2/avx2_int32.hpp>
#include<RAJA/policy/tensor/arch/avx2/avx2_int64.hpp>
#include<RAJA/policy/tensor/arch/avx2/avx2_float.hpp>
#include<RAJA/policy/tensor/arch/avx2/avx2_double.hpp>


#endif // __AVX2__
