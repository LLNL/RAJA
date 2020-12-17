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

#ifndef RAJA_pattern_register_Register_HPP
#define RAJA_pattern_register_Register_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{


  template<typename REGISTER_POLICY,
           typename T,
           int SKEW = 0>
  class Register;


  template<typename REGISTER_POLICY, typename RT, int SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT, SKEW>
  operator+(RT x, Register<REGISTER_POLICY, RT, SKEW> const &y){
    using register_t = Register<REGISTER_POLICY, RT, SKEW>;
    return register_t(x).add(y);
  }

  template<typename REGISTER_POLICY, typename RT, int SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT, SKEW>
  operator-(RT x, Register<REGISTER_POLICY, RT, SKEW> const &y){
    using register_t = Register<REGISTER_POLICY, RT, SKEW>;
    return register_t(x).subtract(y);
  }

  template<typename REGISTER_POLICY, typename RT, int SKEW>
  Register<REGISTER_POLICY, RT, SKEW>
  operator*(RT x, Register<REGISTER_POLICY, RT, SKEW> const &y){
    using register_t = Register<REGISTER_POLICY, RT, SKEW>;
    return register_t(x).multiply(y);
  }

  template<typename REGISTER_POLICY, typename RT, int SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT, SKEW>
  operator/(RT x, Register<REGISTER_POLICY, RT, SKEW> const &y){
    using register_t = Register<REGISTER_POLICY, RT, SKEW>;
    return register_t(x).divide(y);
  }


}  // namespace RAJA


// Bring in RegisterBase, which simplified creating Register specializations
#include "RAJA/pattern/simd_register/RegisterBase.hpp"

// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/simd_register/arch.hpp"


#endif
