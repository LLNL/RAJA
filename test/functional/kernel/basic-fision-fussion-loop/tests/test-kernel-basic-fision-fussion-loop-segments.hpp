//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_BASIC_FISION_FUSSION_LOOP_HPP__
#define __TEST_KERNEL_BASIC_FISION_FUSSION_LOOP_HPP__

#include "basic-fision-fussion-loop-impl.hpp"

TYPED_TEST_SUITE_P(KernelBasicFisionFussionLoopTest);
template <typename T>
class KernelBasicFisionFussionLoopTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelBasicFisionFussionLoopTest, BasicFisionFussionLoopSegmentKernel)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  WORKING_RES working_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_working_res{working_res};

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 37 );

  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedRangeSegment<IDX_TYPE>>(
                                      r1, working_res, erased_working_res);


  
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 2057 );
  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedRangeSegment<IDX_TYPE>>(
                                      r2, working_res, erased_working_res);

  // test zero-length range segment
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 5, 5 );
  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedRangeSegment<IDX_TYPE>>(
                                      r3, working_res, erased_working_res);

// Range-stride segment tests
  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs1( 0, 188, 2 );
  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedRangeStrideSegment<IDX_TYPE>>(
                                      rs1, working_res, erased_working_res);

  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs2( 2, 1029, 3 );
  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedRangeStrideSegment<IDX_TYPE>>(
                                      rs2, working_res, erased_working_res);

  // test zero-length range-stride segment
  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs3( 2, 2, 3 );
  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedRangeStrideSegment<IDX_TYPE>>(
                                      rs3, working_res, erased_working_res);

// List segment tests
  std::vector<IDX_TYPE> seg_idx;
  IDX_TYPE last = IDX_TYPE(10567);
  srand( time(NULL) );
  for (IDX_TYPE i = IDX_TYPE(0); i < last; ++i) {
    IDX_TYPE randval = IDX_TYPE( rand() % RAJA::stripIndexType(last) );
    if ( i < randval ) {
      seg_idx.push_back(i);
    }
  }
  RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(), erased_working_res);
  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedListSegment<IDX_TYPE>>(
                                      l1, working_res, erased_working_res);

  // test zero-length list segment
  seg_idx.clear();
  RAJA::TypedListSegment<IDX_TYPE> l2( nullptr, seg_idx.size(), erased_working_res);
  KernelBasicFisionFussionLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                      RAJA::TypedListSegment<IDX_TYPE>>(
                                      l2, working_res, erased_working_res);
}

REGISTER_TYPED_TEST_SUITE_P(KernelBasicFisionFussionLoopTest,
                            BasicFisionFussionLoopSegmentKernel);
#endif  // __TEST_KERNEL_BASIC_FISION_FUSSION_LOOP_HPP__
