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

#ifndef RAJA_policy_vector_register_altivec_HPP
#define RAJA_policy_vector_register_altivec_HPP

#include "RAJA/config.hpp"
#ifdef RAJA_ALTIVEC

#include<RAJA/policy/vector/register/scalar.hpp>

#include<altivec.h>


namespace RAJA {
  struct vector_altivec_register {};

  template<typename T>
  struct RegisterTraits<vector_altivec_register, T>{

      static
      vector double foo(){
        return (vector double)(0);
      }

      using register_type = decltype(foo());
      using element_type = T;

      static constexpr size_t s_byte_width = sizeof(register_type);
      static constexpr size_t s_bit_width = s_byte_width * 8;
      static constexpr size_t s_num_elem = s_byte_width / sizeof(T);

  };


  // Use the vector_scalar_register for a 1-wide vector
  template<typename T>
  class Register<vector_altivec_register, T, 1> :
  public Register<vector_scalar_register, T, 1> {};


}

#include<RAJA/policy/vector/register/altivec_double.hpp>

#endif // RAJA_ALTIVEC
#endif // guard
