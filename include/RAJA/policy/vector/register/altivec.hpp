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

#include<RAJA/pattern/vector.hpp>
#include<altivec.h>


namespace RAJA {
  struct altivec_register {};

  template<typename T>
  struct RegisterTraits<altivec_register, T>{

      static
      vector T foo(){
        return (vector T)(0);
      }

      using register_type = decltype(foo());
      using element_type = T;


      RAJA_INLINE
      static constexpr
      camp::idx_t num_elem(){return byte_width()/sizeof(T);}

      RAJA_INLINE
      static constexpr
      camp::idx_t byte_width(){return sizeof(register_type);}

      RAJA_INLINE
      static constexpr
      camp::idx_t bit_width(){return byte_width()*8;}

  };



}

#include<RAJA/policy/vector/register/altivec/altivec_int32.hpp>
#include<RAJA/policy/vector/register/altivec/altivec_int64.hpp>
#include<RAJA/policy/vector/register/altivec/altivec_float.hpp>
#include<RAJA/policy/vector/register/altivec/altivec_double.hpp>


#endif // RAJA_ALTIVEC
#endif // guard
