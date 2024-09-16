//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "RAJA_gtest.hpp"

GPU_TEST(SynchronizeUnitTest, HIP)
{

  double* managed_data = (double*)malloc(sizeof(double) * 50);
  double* d_managed_data;
  hipMalloc(&d_managed_data, sizeof(double) * 50);

  RAJA::forall<RAJA::hip_exec_async<256>>(
      RAJA::RangeSegment(0, 50), [=] RAJA_HOST_DEVICE(RAJA::Index_type i)
      { d_managed_data[i] = 1.0 * i; });
  RAJA::synchronize<RAJA::hip_synchronize>();

  hipMemcpy(
      managed_data, d_managed_data, sizeof(double) * 50, hipMemcpyDeviceToHost);

  RAJA::forall<RAJA::seq_exec>(
      RAJA::RangeSegment(0, 50),
      [=](RAJA::Index_type i) { EXPECT_EQ(managed_data[i], 1.0 * i); });

  free(managed_data);
  hipFree(d_managed_data);
}
