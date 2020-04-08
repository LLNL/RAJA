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

#ifndef RAJA_policy_cuda_register_HPP
#define RAJA_policy_cuda_register_HPP

#include<RAJA/pattern/vector.hpp>
#include<RAJA/policy/cuda/register/cuda_warp.hpp>

namespace RAJA
{


  template<typename T, camp::idx_t LANE_BITS = 5, camp::idx_t UNROLL = 1>
  using CudaWarpStreamVector = StreamVector<
      T, UNROLL, cuda_warp_register<LANE_BITS>>;

  template<typename T, camp::idx_t NUM_ELEM, camp::idx_t LANE_BITS = 5>
  using CudaWarpFixedVector = FixedVector<
      T, NUM_ELEM, cuda_warp_register<LANE_BITS>>;

}

#endif
