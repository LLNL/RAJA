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

#include<RAJA/policy/vector/register/scalar.hpp>

namespace RAJA {
  struct vector_avx2_register {};

  template<typename T>
  struct RegisterTraits<vector_avx2_register, T>{
      using register_type = vector_avx2_register;
      using element_type = T;

      static constexpr size_t s_bit_width = 256;
      static constexpr size_t s_byte_width = 32;
      static constexpr size_t s_num_elem = s_byte_width / sizeof(T);

  };

  // Use the vector_scalar_register for a 1-wide vector
  template<typename T>
  class Register<vector_avx2_register, T, 1> :
  public internal::ScalarRegister<vector_avx2_register, T> {

      using Base = internal::ScalarRegister<vector_avx2_register, T>;
      using Base::Base;

  };

}

#endif // guard

#include<RAJA/policy/vector/register/avx2_double2.hpp>
#include<RAJA/policy/vector/register/avx2_double3.hpp>
#include<RAJA/policy/vector/register/avx2_double4.hpp>


#endif // __AVX2__
