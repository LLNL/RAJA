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

#ifndef RAJA_policy_tensor_arch_HPP
#define RAJA_policy_tensor_arch_HPP

#include "RAJA/config.hpp"

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

//
//////////////////////////////////////////////////////////////////////
//
// SIMD register types and policies
//
//////////////////////////////////////////////////////////////////////
//


#ifdef __AVX512F__
struct avx512_register {};
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::avx512_register
#endif
#endif


#ifdef __AVX2__
struct avx2_register {};
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::avx2_register
#endif
#endif


#ifdef __AVX__
struct avx_register {};
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::avx_register
#endif
#endif


#if defined(RAJA_CUDA_ACTIVE)

/*!
 * A CUDA warp distributed vector register
 */

struct cuda_warp_register {};

#endif

// The scalar register is always supported (doesn't require any SIMD/SIMT)
struct scalar_register {};
#ifndef RAJA_TENSOR_REGISTER_TYPE
#define RAJA_TENSOR_REGISTER_TYPE RAJA::scalar_register
#endif


  // This sets the default SIMD register that will be used
  using default_register = RAJA_TENSOR_REGISTER_TYPE;

} // namespace RAJA


#endif
