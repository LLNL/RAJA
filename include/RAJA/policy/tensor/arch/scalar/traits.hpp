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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_tensor_arch_scalar_traits_HPP
#define RAJA_policy_tensor_arch_scalar_traits_HPP

namespace RAJA {
namespace internal {
namespace expt {


  template<>
  struct RegisterTraits<RAJA::expt::scalar_register, int32_t>{
      using element_type = int32_t;
      using register_policy = RAJA::expt::scalar_register;
      static constexpr camp::idx_t s_num_bits = sizeof(element_type)*8;
      static constexpr camp::idx_t s_num_elem = 1;
      using int_element_type = int32_t;
  };

  template<>
  struct RegisterTraits<RAJA::expt::scalar_register, int64_t>{
      using element_type = int64_t;
      using register_policy = RAJA::expt::scalar_register;
      static constexpr camp::idx_t s_num_bits = sizeof(element_type)*8;
      static constexpr camp::idx_t s_num_elem = 1;
      using int_element_type = int64_t;
  };

  template<>
  struct RegisterTraits<RAJA::expt::scalar_register, float>{
      using element_type = float;
      using register_policy = RAJA::expt::scalar_register;
      static constexpr camp::idx_t s_num_bits = sizeof(element_type)*8;
      static constexpr camp::idx_t s_num_elem = 1;
      using int_element_type = int32_t;
  };

  template<>
  struct RegisterTraits<RAJA::expt::scalar_register, double>{
      using element_type = double;
      using register_policy = RAJA::expt::scalar_register;
      static constexpr camp::idx_t s_num_bits = sizeof(element_type)*8;
      static constexpr camp::idx_t s_num_elem = 1;
      using int_element_type = int64_t;
  };


}
}
}

#endif


