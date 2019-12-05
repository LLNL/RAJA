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

#ifndef RAJA_pattern_register_HPP
#define RAJA_pattern_register_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{


/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */

  template<typename REGISTER_POLICY, typename T, size_t NUM_ELEM>
  class Register;

  template<typename REGISTER_POLICY, typename T>
  struct RegisterTraits{
      using register_type = REGISTER_POLICY;
      using element_type = T;

      static constexpr size_t s_num_elem = 1;
      static constexpr size_t s_byte_width = sizeof(T);
      static constexpr size_t s_bit_width = s_byte_width*8;
  };


  template<typename ST, typename REGISTER_POLICY, typename RT, size_t NUM_ELEM>
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator+(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    return Register<REGISTER_POLICY, RT, NUM_ELEM>(x) + y;
  }

  template<typename ST, typename REGISTER_POLICY, typename RT, size_t NUM_ELEM>
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator-(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    return Register<REGISTER_POLICY, RT, NUM_ELEM>(x) - y;
  }

  template<typename ST, typename REGISTER_POLICY, typename RT, size_t NUM_ELEM>
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator*(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    return Register<REGISTER_POLICY, RT, NUM_ELEM>(x) * y;
  }

  template<typename ST, typename REGISTER_POLICY, typename RT, size_t NUM_ELEM>
  Register<REGISTER_POLICY, RT, NUM_ELEM>
  operator/(ST x, Register<REGISTER_POLICY, RT, NUM_ELEM> const &y){
    return Register<REGISTER_POLICY, RT, NUM_ELEM>(x) / y;
  }

}  // namespace RAJA




#endif
