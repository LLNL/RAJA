//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_BASIC_SINGLE_LOOP_SEGMENTS_HPP__
#define __TEST_KERNEL_BASIC_SINGLE_LOOP_SEGMENTS_HPP__

#include "basic-single-loop-segments-impl.hpp"

TYPED_TEST_SUITE_P(KernelBasicSingleLoopTest);
template <typename T>
class KernelBasicSingleLoopTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelBasicSingleLoopTest, BasicSingleLoopSegmentKernel)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  WORKING_RES working_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_working_res{working_res};

  constexpr bool USE_RES = false;

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 37 );
  RAJA::getIndices(seg_idx, r1);

  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedRangeSegment<IDX_TYPE>, USE_RES>(
                                  r1, seg_idx, working_res, erased_working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 2057 );
  RAJA::getIndices(seg_idx, r2);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedRangeSegment<IDX_TYPE>, USE_RES>(
                                  r2, seg_idx, working_res, erased_working_res);

  // test zero-length range segment
  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 5, 5 );
  RAJA::getIndices(seg_idx, r3);

  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedRangeSegment<IDX_TYPE>, USE_RES>(
                                  r3, seg_idx, working_res, erased_working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs1( 0, 188, 2 );
  RAJA::getIndices(seg_idx, rs1);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedRangeStrideSegment<IDX_TYPE>, USE_RES>(
                                  rs1, seg_idx, working_res, erased_working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs2( 2, 1029, 3 );
  RAJA::getIndices(seg_idx, rs2);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedRangeStrideSegment<IDX_TYPE>, USE_RES>(
                                  rs2, seg_idx, working_res, erased_working_res);

  // test zero-length range-stride segment
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs3( 2, 2, 3 );
  RAJA::getIndices(seg_idx, rs3);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedRangeStrideSegment<IDX_TYPE>, USE_RES>(
                                  rs3, seg_idx, working_res, erased_working_res);

// List segment tests
  seg_idx.clear();
  IDX_TYPE last = IDX_TYPE(10567);
  srand( time(NULL) );
  for (IDX_TYPE i = IDX_TYPE(0); i < last; ++i) {
    IDX_TYPE randval = IDX_TYPE( rand() % RAJA::stripIndexType(last) );
    if ( i < randval ) {
      seg_idx.push_back(i);
    }
  }
  RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(), erased_working_res);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedListSegment<IDX_TYPE>, USE_RES>(
                                  l1, seg_idx, working_res, erased_working_res);

  // test zero-length list segment
  seg_idx.clear();
  RAJA::TypedListSegment<IDX_TYPE> l2( nullptr, seg_idx.size(), erased_working_res);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY, WORKING_RES,
                                RAJA::TypedListSegment<IDX_TYPE>, USE_RES>(
                                  l2, seg_idx, working_res, erased_working_res);
}

REGISTER_TYPED_TEST_SUITE_P(KernelBasicSingleLoopTest,
                            BasicSingleLoopSegmentKernel);

#endif  // __TEST_KERNEL_BASIC_SINGLE_LOOP_SEGMENTS_HPP__
