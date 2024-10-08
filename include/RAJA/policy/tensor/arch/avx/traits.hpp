/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX__

#ifndef RAJA_policy_tensor_arch_avx_traits_HPP
#define RAJA_policy_tensor_arch_avx_traits_HPP

namespace RAJA {
namespace internal {
namespace expt {

  template<>
  struct RegisterTraits<RAJA::expt::avx_register, int32_t>{
      using element_type = int32_t;
      using register_policy = RAJA::expt::avx_register;
      static constexpr camp::idx_t s_num_bits = 256;
      static constexpr camp::idx_t s_num_elem = 8;
      using int_element_type = int32_t;
  };

  template<>
  struct RegisterTraits<RAJA::expt::avx_register, int64_t>{
      using element_type = int64_t;
      using register_policy = RAJA::expt::avx_register;
      static constexpr camp::idx_t s_num_bits = 256;
      static constexpr camp::idx_t s_num_elem = 4;
      using int_element_type = int64_t;
  };

  template<>
  struct RegisterTraits<RAJA::expt::avx_register, float>{
      using element_type = float;
      using register_policy = RAJA::expt::avx_register;
      static constexpr camp::idx_t s_num_bits = 256;
      static constexpr camp::idx_t s_num_elem = 8;
      using int_element_type = int32_t;
  };

  template<>
  struct RegisterTraits<RAJA::expt::avx_register, double>{
      using element_type = double;
      using register_policy = RAJA::expt::avx_register;
      static constexpr camp::idx_t s_num_bits = 256;
      static constexpr camp::idx_t s_num_elem = 4;
      using int_element_type = int64_t;
  };

} // namespace intenral
} // namespace expt
} // namespace RAJA


#endif


#endif // __AVX__
