//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REGION_HPP__
#define __TEST_KERNEL_REGION_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index.hpp"

TYPED_TEST_SUITE_P(KernelRegionBasicTest);
template <typename T>
class KernelRegionBasicTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(KernelRegionSyncTest);
template <typename T>
class KernelRegionSyncTest : public ::testing::Test
{
};

#include "tests/test-kernel-region-basic.hpp"
#include "tests/test-kernel-region-sync.hpp"

REGISTER_TYPED_TEST_SUITE_P(KernelRegionBasicTest,
                            RegionBasicKernel);

REGISTER_TYPED_TEST_SUITE_P(KernelRegionSyncTest,
                            RegionSyncKernel);

#endif  // __TEST_KERNEL_REGION_HPP__
