/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA simd policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_register_arch_HPP
#define RAJA_policy_register_arch_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/TensorRegister.hpp"

namespace RAJA
{
    template<typename REGISTER_POLICY, typename T>
    struct RegisterTraits;
    /*
     * using element_type = T;
     * using register_policy = REGISTER_POLICY;
     * static constexpr camp::idx s_num_bits = X;
     * static constexpr camp::idx s_num_elem = Y;
     *
     */

}

//
//////////////////////////////////////////////////////////////////////
//
// SIMD register types and policies
//
//////////////////////////////////////////////////////////////////////
//

#ifdef __AVX512F__
#include<RAJA/policy/tensor/arch/avx512.hpp>
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::avx512_register
#endif
#endif


#ifdef __AVX2__
#include<RAJA/policy/tensor/arch/avx2.hpp>
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::avx2_register
#endif
#endif


#ifdef __AVX__
#include<RAJA/policy/tensor/arch/avx.hpp>
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::avx_register
#endif
#endif



// The scalar register is always supported (doesn't require any SIMD/SIMT)
#include<RAJA/policy/tensor/arch/scalar/scalar.hpp>
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::scalar_register
#endif


namespace RAJA
{
  // This sets the default SIMD register that will be used
  using default_register = RAJA_TENSOR_REGISTER_TYPE;

  // Convenience to describe VectorTensors
  template<typename REGISTER_POLICY, typename T>
  using VectorRegister = TensorRegister<avx_register, double, VectorLayout, camp::idx_seq<RegisterTraits<REGISTER_POLICY,T>::s_num_elem>, 0>;

} // namespace RAJA


#endif
