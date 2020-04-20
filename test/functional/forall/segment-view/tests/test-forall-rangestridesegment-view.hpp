//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESTRIDESEGMENT_VIEW_HPP__
#define __TEST_FORALL_RANGESTRIDESEGMENT_VIEW_HPP__

#include "test-forall-segment-view.hpp"

template <typename INDEX_TYPE, typename DIFF_TYPE, 
          typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeStrideSegmentViewTest(INDEX_TYPE first, INDEX_TYPE last, 
                                      DIFF_TYPE stride)
{
  RAJA::TypedRangeStrideSegment<INDEX_TYPE> r1(first, last, stride);
  INDEX_TYPE N = r1.size();

  camp::resources::Resource working_res{WORKING_RES()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  memset( test_array, 0, sizeof(INDEX_TYPE) * N );

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * N);

  INDEX_TYPE idx = first;
  for (INDEX_TYPE i = 0; i < N; ++i) {
    test_array[ (idx-first)/stride ] = idx;
    idx += stride;
  }

  using view_type = RAJA::View< INDEX_TYPE, RAJA::Layout<1, INDEX_TYPE, 0> >;

  RAJA::Layout<1> layout(N);
  view_type work_view(working_array, layout);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( (idx-first)/stride ) = idx;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * N);

  for (INDEX_TYPE i = 0; i < N; i++) {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<INDEX_TYPE>::value>::type* = nullptr>
void runNegativeStrideViewTests()
{
}

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<INDEX_TYPE>::value>::type* = nullptr>
void runNegativeStrideViewTests()
{
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(-10, -1, 2);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(-5, 0, 2);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(-5, 5, 3);

// Test negative strides
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(10, -1, -1);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(10, 0, -2);
}


TYPED_TEST_P(ForallSegmentViewTest, RangeStrideSegmentForallView)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;
  using DIFF_TYPE   = typename std::make_signed<INDEX_TYPE>::type;

  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(0, 20, 1);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(1, 20, 1);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(0, 20, 2);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(1, 20, 2);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(0, 21, 2);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(1, 21, 2);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(1, 255, 2);

// Test size zero segments
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(0, 20, -2);
  ForallRangeStrideSegmentViewTest<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(1, 20, -2);

  runNegativeStrideViewTests<INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>();
}

#endif  // __TEST_FORALL_RANGESTRIDESEGMENT_VIEW_HPP__
