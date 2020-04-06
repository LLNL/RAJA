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


  template<typename REGISTER_POLICY,
           typename T>
  class Register;


  template<typename ST, typename REGISTER_POLICY, typename RT>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT>
  operator+(ST x, Register<REGISTER_POLICY, RT> const &y){
    using register_t = Register<REGISTER_POLICY, RT>;
    return register_t(x).add(y);
  }

  template<typename ST, typename REGISTER_POLICY, typename RT>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT>
  operator-(ST x, Register<REGISTER_POLICY, RT> const &y){
    using register_t = Register<REGISTER_POLICY, RT>;
    return register_t(x).subtract(y);
  }

  template<typename ST, typename REGISTER_POLICY, typename RT>
  Register<REGISTER_POLICY, RT>
  operator*(ST x, Register<REGISTER_POLICY, RT> const &y){
    using register_t = Register<REGISTER_POLICY, RT>;
    return register_t(x).multiply(y);
  }

  template<typename ST, typename REGISTER_POLICY, typename RT>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Register<REGISTER_POLICY, RT>
  operator/(ST x, Register<REGISTER_POLICY, RT> const &y){
    using register_t = Register<REGISTER_POLICY, RT>;
    return register_t(x).divide(y);
  }


}  // namespace RAJA


// Bring in RegisterBase, which simplified creating Register specializations
#include "RAJA/pattern/vector/internal/RegisterBase.hpp"

// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/vector/register.hpp"


#endif
