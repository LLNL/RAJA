//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_RESOURCE_REDUCE_CONDITIONAL_LOOP_SEGMENTS_HPP__
#define __TEST_KERNEL_RESOURCE_REDUCE_CONDITIONAL_LOOP_SEGMENTS_HPP__

#include "reduce-conditional-loop-impl.hpp"

TYPED_TEST_SUITE_P(KernelReduceConditionalLoopTest);
template <typename T>
class KernelReduceConditionalLoopTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelReduceConditionalLoopTest, ReduceConditionalLoopKernel)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  WORKING_RES working_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_working_res{working_res};

  constexpr bool USE_RES = true;

  // Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 37 );
  KernelReduceConditionalLoopTestImpl<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY, WORKING_RES,
                                RAJA::TypedRangeSegment<IDX_TYPE>, USE_RES>(
                                  r1, working_res, erased_working_res);

  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 2057 );
  KernelReduceConditionalLoopTestImpl<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY, WORKING_RES,
                                RAJA::TypedRangeSegment<IDX_TYPE>, USE_RES>(
                                  r2, working_res, erased_working_res);

}

REGISTER_TYPED_TEST_SUITE_P(KernelReduceConditionalLoopTest,
                            ReduceConditionalLoopKernel);

#endif  // __TEST_KERNEL_RESOURCE_REDUCE_CONDITIONAL_LOOP_SEGMENTS_HPP__
