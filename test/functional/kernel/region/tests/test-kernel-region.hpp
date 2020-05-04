//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REGION_HPP__
#define __TEST_KERNEL_REGION_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

#include "gtest/gtest.h"

TYPED_TEST_SUITE_P(KernelRegionTest);
template <typename T>
class KernelRegionTest : public ::testing::Test
{
};

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelBasicRegionTest(INDEX_TYPE first, INDEX_TYPE last)
{
  camp::resources::Resource work_res{WORKING_RES()};

  //
  // Set some local variables and create some segments for using in tests
  //
  const INDEX_TYPE N = last - first;
  
  RAJA::TypedRangeSegment<INDEX_TYPE> rseg(first, last);

  INDEX_TYPE* work_array1 = work_res.allocate<INDEX_TYPE>(N);
  INDEX_TYPE* work_array2 = work_res.allocate<INDEX_TYPE>(N);
  INDEX_TYPE* work_array3 = work_res.allocate<INDEX_TYPE>(N);

  work_res.memset( work_array1, 0, sizeof(INDEX_TYPE) * N );
  work_res.memset( work_array2, 0, sizeof(INDEX_TYPE) * N );
  work_res.memset( work_array3, 0, sizeof(INDEX_TYPE) * N );

  camp::resources::Resource host_res{camp::resources::Host()};

  INDEX_TYPE* check_array = host_res.allocate<INDEX_TYPE>(N);

  host_res.memset( check_array, 0, sizeof(INDEX_TYPE) * N );

  RAJA::kernel<EXEC_POLICY>(

    RAJA::make_tuple(rseg),

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

  work_res.deallocate(work_array1);
  work_res.deallocate(work_array2);
  work_res.deallocate(work_array3);

  host_res.deallocate(check_array);
}

TYPED_TEST_P(KernelRegionTest, RegionSegmentKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  KernelBasicRegionTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(0, 25);
  KernelBasicRegionTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(1, 153);
  KernelBasicRegionTest<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(3, 2556);
}

REGISTER_TYPED_TEST_SUITE_P(KernelRegionTest,
                            RegionSegmentKernel);

#endif  // __TEST_KERNEL_REGION_HPP__
