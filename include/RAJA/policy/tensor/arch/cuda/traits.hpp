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
namespace expt {
namespace internal {

  template<typename T>
  struct RegisterTraits<cuda_warp_register, T>{
      using element_type = T;
      using register_policy = cuda_warp_register;
      static constexpr camp::idx_t s_num_elem = 32;
      static constexpr camp::idx_t s_num_bits = sizeof(T) * s_num_elem;
      using int_element_type = int32_t;
  };

} // namespace internal
} // namespace expt
} // namespace RAJA



#endif


#endif // RAJA_ENABLE_CUDA
