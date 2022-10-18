//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_MULTI_LAMBDA_HPP__
#define __TEST_KERNEL_NESTED_LOOP_MULTI_LAMBDA_HPP__

#include "nested-loop-MultiLambda-impl.hpp"

//
//
// Setup the Nested Loop Multi Lambda g-tests.
//
//
TYPED_TEST_SUITE_P(KernelNestedLoopMultiLambdaTest);
template <typename T>
class KernelNestedLoopMultiLambdaTest : public ::testing::Test {};

TYPED_TEST_P(KernelNestedLoopMultiLambdaTest, NestedLoopMultiLambdaKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<1>>::type;

  // Attain the loop depth type from execpol data.
  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;

  // Get List of loop exec policies.
  using LOOP_POLS = typename EXEC_POL_DATA::type;

  // Build proper basic kernel exec policy type.
  using EXEC_POLICY = typename MultiLambdaNestedLoopExec<LOOP_TYPE, LOOP_POLS>::type;

  constexpr bool USE_RES = false;

  // For double nested loop tests the third arg is ignored.
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RES>();
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopMultiLambdaTest,
                            NestedLoopMultiLambdaKernel);

#endif  // __TEST_KERNEL_NESTED_LOOP_MULTI_LAMBDA_HPP__
