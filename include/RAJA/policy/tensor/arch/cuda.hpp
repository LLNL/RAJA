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

#ifdef RAJA_ENABLE_CUDA


#ifndef RAJA_policy_tensor_arch_cuda_HPP
#define RAJA_policy_tensor_arch_cuda_HPP


namespace RAJA {

  template<typename T>
  struct RegisterTraits<cuda_warp_register, T>{
      using element_type = T;
      using register_policy = cuda_warp_register;
      static constexpr camp::idx_t s_num_elem = 32;
      static constexpr camp::idx_t s_num_bits = sizeof(T) * s_num_elem;
  };
}

#include<RAJA/pattern/tensor.hpp>

#include<RAJA/policy/tensor/arch/cuda/cuda_warp.hpp>


#endif


#endif // RAJA_ENABLE_CUDA
