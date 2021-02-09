//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_BASIC_SINGLE_LOOP_SEGMENTS_HPP__
#define __TEST_KERNEL_BASIC_SINGLE_LOOP_SEGMENTS_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename EXEC_POLICY, 
          typename SEG_TYPE>
void KernelBasicSingleLoopTestImpl(const SEG_TYPE& seg, 
                                   const std::vector<IDX_TYPE>& seg_idx,
                                   camp::resources::Resource& working_res)
{
  IDX_TYPE data_len = seg_idx[seg_idx.size() - 1] + 1;
  IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );

  IDX_TYPE* working_array;
  IDX_TYPE* check_array;
  IDX_TYPE* test_array;

  allocateForallTestData<IDX_TYPE>(data_len,
                                   working_res,
                                   &working_array,
                                   &check_array,
                                   &test_array);

  for (IDX_TYPE i = IDX_TYPE(0); i < data_len; ++i) {
    test_array[RAJA::stripIndexType(i)] = static_cast<IDX_TYPE>(0);
  }
 
  working_res.memcpy(working_array, test_array, 
                     sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  for (IDX_TYPE i = IDX_TYPE(0); i < idx_len; ++i) {
    test_array[ RAJA::stripIndexType(seg_idx[RAJA::stripIndexType(i)]) ] = 
       seg_idx[RAJA::stripIndexType(i)];
  }
 
  RAJA::kernel<EXEC_POLICY>( RAJA::make_tuple( seg ), 
    [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      working_array[RAJA::stripIndexType(idx)] = idx;
    }
  );

  working_res.memcpy(check_array, working_array, 
                     sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  for (IDX_TYPE i = IDX_TYPE(0); i < data_len; ++i) {
    ASSERT_EQ( test_array[RAJA::stripIndexType(i)], 
               check_array[RAJA::stripIndexType(i)] );
  }

  deallocateForallTestData<IDX_TYPE>(working_res,
                                     working_array,
                                     check_array,
                                     test_array);
}


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

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 37 );
  RAJA::getIndices(seg_idx, r1);

  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY,
                                RAJA::TypedRangeSegment<IDX_TYPE>>(
                                  r1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 2057 );
  RAJA::getIndices(seg_idx, r2);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY,
                                RAJA::TypedRangeSegment<IDX_TYPE>>(
                                  r2, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs1( 0, 188, 2 );
  RAJA::getIndices(seg_idx, rs1);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY,
                                RAJA::TypedRangeStrideSegment<IDX_TYPE>>(
                                  rs1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> rs2( 2, 1029, 3 );
  RAJA::getIndices(seg_idx, rs2);
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY,
                                RAJA::TypedRangeStrideSegment<IDX_TYPE>>(
                                  rs2, seg_idx, working_res);

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
  RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(),
                                       working_res );
  KernelBasicSingleLoopTestImpl<IDX_TYPE, EXEC_POLICY,
                                RAJA::TypedListSegment<IDX_TYPE>>(
                                  l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(KernelBasicSingleLoopTest,
                            BasicSingleLoopSegmentKernel);

#endif  // __TEST_KERNEL_BASIC_SINGLE_LOOP_SEGMENTS_HPP__
