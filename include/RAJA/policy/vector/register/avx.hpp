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

#include<RAJA/policy/vector/register/scalar.hpp>


namespace RAJA {
  struct vector_avx_register {};

  template<typename T>
  struct RegisterTraits<vector_avx_register, T>{
      using register_type = vector_avx_register;
      using element_type = T;


      static constexpr size_t s_bit_width = 256;
      static constexpr size_t s_byte_width = 32;
      static constexpr size_t s_num_elem = s_byte_width / sizeof(T);

  };

}


#endif

#include<RAJA/policy/vector/register/avx_double1.hpp>
#include<RAJA/policy/vector/register/avx_double2.hpp>
#include<RAJA/policy/vector/register/avx_double3.hpp>
#include<RAJA/policy/vector/register/avx_double4.hpp>


#endif // __AVX__
