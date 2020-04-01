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

#include<RAJA/pattern/vector.hpp>

namespace RAJA {
  struct avx2_register {};

  template<typename T>
  struct RegisterTraits<avx2_register, T>{
      using register_type = avx2_register;
      using element_type = T;

      RAJA_INLINE
      static constexpr
      camp::idx_t num_elem(){return 32 / sizeof(T);}

      RAJA_INLINE
      static constexpr
      camp::idx_t byte_width(){return 32;}

      RAJA_INLINE
      static constexpr
      camp::idx_t bit_width(){return 256;}

  };


}

#endif // guard

#include<RAJA/policy/vector/register/avx2/avx2_int32.hpp>
#include<RAJA/policy/vector/register/avx2/avx2_int64.hpp>
#include<RAJA/policy/vector/register/avx2/avx2_float.hpp>
#include<RAJA/policy/vector/register/avx2/avx2_double.hpp>


#endif // __AVX2__
