//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESEGMENT_VIEW_HPP__
#define __TEST_FORALL_RANGESEGMENT_VIEW_HPP__

#include "test-forall-segment-view.hpp"

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegmentViewTest(INDEX_TYPE first, INDEX_TYPE last)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(first, last);
  INDEX_TYPE N = r1.end() - r1.begin();

  camp::resources::Resource working_res{WORKING_RES()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  const INDEX_TYPE rbegin = *r1.begin();

  std::iota(test_array, test_array + N, rbegin);

  using view_type = RAJA::View< INDEX_TYPE, RAJA::Layout<1, INDEX_TYPE, 0> >;
 
  RAJA::Layout<1> layout(N);
  view_type work_view(working_array, layout);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( idx - rbegin ) = idx;
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

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegmentOffsetViewTest(INDEX_TYPE first, INDEX_TYPE last, 
                                      INDEX_TYPE offset)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(first+offset, last+offset);
  INDEX_TYPE N = r1.end() - r1.begin();

  camp::resources::Resource working_res{WORKING_RES()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  const INDEX_TYPE rbegin = *r1.begin();

  std::iota(test_array, test_array + N, rbegin);

  using view_type = RAJA::View< INDEX_TYPE, RAJA::OffsetLayout<1, INDEX_TYPE> >;

  INDEX_TYPE f_offset = first + offset;
  INDEX_TYPE l_offset = last + offset;
  view_type work_view(working_array, 
                      RAJA::make_offset_layout<1, INDEX_TYPE>({{f_offset}},
                                                              {{l_offset}}));

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    work_view( idx ) = idx;
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

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<INDEX_TYPE>::value>::type* = nullptr>
void runNegativeViewTests()
{
}

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<INDEX_TYPE>::value>::type* = nullptr>
void runNegativeViewTests()
{
  ForallRangeSegmentViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 0);
  ForallRangeSegmentViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 5);

  ForallRangeSegmentOffsetViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 0, 1);
  ForallRangeSegmentOffsetViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(-5, 5, 2);
}


TYPED_TEST_P(ForallSegmentViewTest, RangeSegmentForallView)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  ForallRangeSegmentViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 5);
  ForallRangeSegmentViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 5);
  ForallRangeSegmentViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 255);

  ForallRangeSegmentOffsetViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 5, 1);
  ForallRangeSegmentOffsetViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 5, 2);
  ForallRangeSegmentOffsetViewTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 255, 3);

  runNegativeViewTests<INDEX_TYPE, WORKING_RES, EXEC_POLICY>();
}

#endif  // __TEST_FORALL_RANGESEGMENT_VIEW_HPP__
