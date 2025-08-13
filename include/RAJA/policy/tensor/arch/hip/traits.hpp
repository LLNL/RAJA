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

#ifndef RAJA_policy_tensor_arch_hip_traits_HPP
#define RAJA_policy_tensor_arch_hip_traits_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_HIP_ACTIVE)


namespace RAJA
{
namespace internal
{
namespace expt
{

template<typename T>
struct RegisterTraits<RAJA::expt::hip_wave_register, T>
{
  using element_type                      = T;
  using register_policy                   = RAJA::expt::hip_wave_register;
  static constexpr camp::idx_t s_num_elem = RAJA_HIP_WAVESIZE;
  static constexpr camp::idx_t s_num_bits = sizeof(T) * s_num_elem;
  using int_element_type                  = int32_t;
};

}  // namespace expt
}  // namespace internal
}  // namespace RAJA


#endif  // RAJA_HIP_ACTIVE

#endif
