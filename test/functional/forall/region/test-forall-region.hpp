//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REGION_HPP__
#define __TEST_FORALL_REGION_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index-types.hpp"

#include "RAJA_test-forall-data.hpp"
#include "RAJA_test-forall-execpol.hpp"

TYPED_TEST_SUITE_P(ForallRegionTest);
template <typename T>
class ForallRegionTest : public ::testing::Test
{
};

#include "tests/test-forall-region-basic.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallRegionTest,
                            RegionBasicSegmentForall);

#endif  // __TEST_FORALL_REGION_HPP__
