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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_tensor_arch_cuda_traits_HPP
#define RAJA_policy_tensor_arch_cuda_traits_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_CUDA_ACTIVE)


namespace RAJA
{
namespace internal
{
namespace expt
{

template<typename T>
struct RegisterTraits<RAJA::expt::cuda_warp_register, T>
{
  using element_type                      = T;
  using register_policy                   = RAJA::expt::cuda_warp_register;
  static constexpr camp::idx_t s_num_elem = RAJA_CUDA_WARPSIZE;
  static constexpr camp::idx_t s_num_bits = sizeof(T) * s_num_elem;
  using int_element_type                  = int32_t;
};

}  // namespace expt
}  // namespace internal
}  // namespace RAJA


#endif  // RAJA_CUDA_ACTIVE

#endif
