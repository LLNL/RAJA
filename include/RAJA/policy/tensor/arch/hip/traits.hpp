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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef RAJA_ENABLE_HIP


#ifndef RAJA_policy_tensor_arch_hip_traits_HPP
#define RAJA_policy_tensor_arch_hip_traits_HPP

namespace RAJA {
namespace internal {
namespace expt {

  template<typename T>
  struct RegisterTraits<RAJA::expt::hip_wave_register, T>{
      using element_type = T;
      using register_policy = RAJA::expt::hip_wave_register;
      static constexpr camp::idx_t s_num_elem = 64;
      static constexpr camp::idx_t s_num_bits = sizeof(T) * s_num_elem;
      using int_element_type = int32_t;
  };

} // namespace internal
} // namespace expt
} // namespace RAJA



#endif


#endif // RAJA_ENABLE_HIP
