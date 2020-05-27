//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REGION_UTILS_HPP__
#define __TEST_KERNEL_REGION_UTILS_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

#include "gtest/gtest.h"

template <typename T>
class KernelRegionFunctionalTest : public ::testing::Test
{
};

template <typename T>
void allocRegionTestData(int N,
                         camp::resources::Resource& work_res,
                         T** work1, T** work2, T** work3,
                         camp::resources::Resource& host_res,
                         T** check)
{
  *work1 = work_res.allocate<T>(N);
  *work2 = work_res.allocate<T>(N);
  *work3 = work_res.allocate<T>(N);

  *check = host_res.allocate<T>(N);
}

template <typename T>
void deallocRegionTestData(camp::resources::Resource& work_res,
                           T* work1, T* work2, T* work3,
                           camp::resources::Resource& host_res,
                           T* check)
{
  work_res.deallocate(work1);
  work_res.deallocate(work2);
  work_res.deallocate(work3);

  host_res.deallocate(check);
}

TYPED_TEST_SUITE_P(KernelRegionFunctionalTest);

#endif  // __TEST_KERNEL_REGION_UTILS_HPP__
