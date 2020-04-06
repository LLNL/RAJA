/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_vector_register_HPP
#define RAJA_pattern_vector_register_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{


/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */


  template<typename REGISTER_POLICY, typename T>
  struct RegisterTraits{
      using register_policy = REGISTER_POLICY;
      using element_type = camp::decay<T>;

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static constexpr
      camp::idx_t num_elem(){return 1;}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static constexpr
      camp::idx_t byte_width(){return sizeof(element_type);}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static constexpr
      camp::idx_t bit_width(){return 8*sizeof(element_type);}

  };

  template<typename REGISTER_POLICY,
           typename T,
           camp::idx_t NUM_ELEM = RegisterTraits<REGISTER_POLICY,T>::num_elem()>
  class Register;


  template<typename ST, typename REGISTER_POLICY, typename RT, camp::idx_t NUM_ELEM>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator+(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    using register_t = Register<REGISTER_POLICY, RT, NUM_ELEM>;
    return register_t(x).add(y);
  }

  template<typename ST, typename REGISTER_POLICY, typename RT, camp::idx_t NUM_ELEM>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator-(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    using register_t = Register<REGISTER_POLICY, RT, NUM_ELEM>;
    return register_t(x).subtract(y);
  }

  template<typename ST, typename REGISTER_POLICY, typename RT, camp::idx_t NUM_ELEM>
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator*(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    using register_t = Register<REGISTER_POLICY, RT, NUM_ELEM>;
    return register_t(x).multiply(y);
  }

  template<typename ST, typename REGISTER_POLICY, typename RT, camp::idx_t NUM_ELEM>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator/(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    using register_t = Register<REGISTER_POLICY, RT, NUM_ELEM>;
    return register_t(x).divide(y);
  }


}  // namespace RAJA


// Bring in RegisterBase, which simplified creating Register specializations
#include "RAJA/pattern/vector/internal/RegisterBase.hpp"

// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/vector/register.hpp"


#endif
