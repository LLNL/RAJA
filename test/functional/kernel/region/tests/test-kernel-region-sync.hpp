//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REGION_SYNC_HPP__
#define __TEST_KERNEL_REGION_SYNC_HPP__

#include "test-kernel-region-utils.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelRegionSyncFunctionalTest(INDEX_TYPE first, INDEX_TYPE last)
{
  camp::resources::Resource host_res{camp::resources::Host()};
  camp::resources::Resource work_res{WORKING_RES()};

  const INDEX_TYPE N = last - first;
  
  INDEX_TYPE* work_array1;
  INDEX_TYPE* work_array2;
  INDEX_TYPE* work_array3;

  INDEX_TYPE* check_array;

  allocRegionTestData(N,
                      work_res,
                      &work_array1, &work_array2, &work_array3,
                      host_res,
                      &check_array);

  work_res.memset( work_array1, 0, sizeof(INDEX_TYPE) * N );
  work_res.memset( work_array2, 0, sizeof(INDEX_TYPE) * N );
  work_res.memset( work_array3, 0, sizeof(INDEX_TYPE) * N );

  host_res.memset( check_array, 0, sizeof(INDEX_TYPE) * N );

  //
  // Create a list segment with indices in reverse order from range
  // segment below. In the test kernel below, the first and third
  // lambda expressions are run in loops using the range segment. The
  // second lambda is run in a loop using the list segment. This makes
  // it so that parallel threads must be synchronized between the loops.
  //
  std::vector<INDEX_TYPE> idx_array(N);
  std::iota(idx_array.begin(), idx_array.end(), first);
  std::reverse(idx_array.begin(), idx_array.end());
  RAJA::TypedListSegment<INDEX_TYPE> lseg(&idx_array[0], N,
                                          work_res);

  RAJA::TypedRangeSegment<INDEX_TYPE> rseg(first, last);

  RAJA::kernel<EXEC_POLICY>(

    RAJA::make_tuple(rseg, lseg),

    [=] (INDEX_TYPE i) {
      work_array1[i - first] = 50;
    },

    [=] (INDEX_TYPE i) {
      work_array2[i - first] = 100;
    },

    [=] (INDEX_TYPE i) {
      work_array3[i - first] = work_array1[i - first] + 
                               work_array2[i - first] + 1;
    }

  );
  
  work_res.memcpy(check_array, work_array3, sizeof(INDEX_TYPE) * N);

  for (INDEX_TYPE i = 0; i < N; i++) {
    ASSERT_EQ(check_array[i], 151);
  }

  deallocRegionTestData(work_res,
                        work_array1, work_array2, work_array3,
                        host_res,
                        check_array);
}

TYPED_TEST_P(KernelRegionFunctionalTest, RegionSyncKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  KernelRegionSyncFunctionalTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 25);
  KernelRegionSyncFunctionalTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 153);
  KernelRegionSyncFunctionalTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(3, 2556);
}

REGISTER_TYPED_TEST_SUITE_P(KernelRegionFunctionalTest,
                            RegionSyncKernel);

#endif  // __TEST_KERNEL_REGION_SYNC_HPP__
