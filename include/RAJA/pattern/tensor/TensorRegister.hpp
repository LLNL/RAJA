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
           typename VAL_SEQ,
           camp::idx_t SKEW>
  class TensorRegister;


  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ, camp::idx_t SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>
  operator+(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>;
    return register_t(x).add(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ, camp::idx_t SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>
  operator-(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>;
    return register_t(x).subtract(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ, camp::idx_t SKEW>
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>
  operator*(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>;
    return register_t(x).multiply(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, typename VAL_SEQ, camp::idx_t SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>
  operator/(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, VAL_SEQ, SKEW>;
    return register_t(x).divide(y);
  }

}  // namespace RAJA


#include "RAJA/pattern/tensor/internal/TensorRegisterBase.hpp"

#endif
