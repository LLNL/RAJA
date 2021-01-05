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

#ifndef RAJA_pattern_tensor_TensorRegister_HPP
#define RAJA_pattern_tensor_TensorRegister_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorLayout.hpp"
#include "RAJA/pattern/tensor/internal/ET/TensorRef.hpp"

namespace RAJA
{
  template<typename REGISTER_POLICY,
           typename T,
           typename LAYOUT,
           typename SIZES,
           typename VAL_SEQ>
  class TensorRegister;


  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>
  operator+(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>;
    return register_t(x).add(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>
  operator-(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>;
    return register_t(x).subtract(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ>
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>
  operator*(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>;
    return register_t(x).multiply(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>
  operator/(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ>;
    return register_t(x).divide(y);
  }

}  // namespace RAJA


#include "RAJA/pattern/tensor/internal/TensorRegisterBase.hpp"

#endif
