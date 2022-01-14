//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_WARP_THREAD_RESOURCE_REDUCESUM_HPP__
#define __TEST_WARP_THREAD_RESOURCE_REDUCESUM_HPP__

#include "warp-thread-ReduceSum-impl.hpp"

//
//
// Setup the Warp Reduction ReduceSum g-tests.
//
//
TYPED_TEST_SUITE_P(KernelWarpThreadReduceSumTest);
template <typename T>
class KernelWarpThreadReduceSumTest : public ::testing::Test {};

TYPED_TEST_P(KernelWarpThreadReduceSumTest, WarpThreadReduceSumKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using REDUCE_POL = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<2>>::type;

  // Attain the loop depth type from execpol data.
  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;

  // Get List of loop exec policies.
  using LOOP_POLS = typename EXEC_POL_DATA::type;

  // Build proper basic kernel exec policy type.
  using EXEC_POLICY = typename WarpThreadExec<LOOP_TYPE, REDUCE_POL, LOOP_POLS>::type;

  constexpr bool USE_RES = true;

  // For double nested loop tests the third arg is ignored.
  KernelWarpThreadTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RES>( LOOP_TYPE(), 2345);
}

REGISTER_TYPED_TEST_SUITE_P(KernelWarpThreadReduceSumTest,
                            WarpThreadReduceSumKernel);

#endif  // __TEST_WARP_THREAD_RESOURCE_REDUCESUM_HPP__
