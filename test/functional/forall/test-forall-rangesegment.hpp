//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESEGMENT_HPP__
#define __TEST_FORALL_RANGESEGMENT_HPP__

#include "test-forall.hpp"
#include <numeric>

using namespace camp::resources;
using namespace camp;

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegmentFunctionalTest(INDEX_TYPE first, INDEX_TYPE last)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(first, last);
  INDEX_TYPE N = r1.end() - r1.begin();

  Resource working_res{WORKING_RES()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  std::iota(test_array, test_array + N, *r1.begin());

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    working_array[idx - *r1.begin()] = idx;
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


TYPED_TEST_P(ForallFunctionalTest, RangeSegmentForall)
{
  using INDEX_TYPE       = typename at<TypeParam, num<0>>::type;
  using WORKING_RESOURCE = typename at<TypeParam, num<1>>::type;
  using EXEC_POLICY      = typename at<TypeParam, num<2>>::type;

  ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(0, 5);
  ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(1, 5);
  ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(1, 255);

  if (std::is_signed<INDEX_TYPE>::value) {
#if !defined(__CUDA_ARCH__) && !defined(RAJA_ENABLE_TBB)
    ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(-5, 0);
    ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(-5, 5);
#endif
  }
}

REGISTER_TYPED_TEST_SUITE_P(ForallFunctionalTest, RangeSegmentForall);

#endif  // __TEST_FORALL_RANGESEGMENT_HPP__
