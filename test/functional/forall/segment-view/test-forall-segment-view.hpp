//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_SEGMENT_VIEW_HPP__
#define __TEST_FORALL_SEGMENT_VIEW_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index-types.hpp"

#include "RAJA_test-forall-data.hpp"
#include "RAJA_test-forall-execpol.hpp"

TYPED_TEST_SUITE_P(ForallSegmentViewTest);
template <typename T>
class ForallSegmentViewTest : public ::testing::Test
{
};

#include "tests/test-forall-rangesegment-view.hpp"
#include "tests/test-forall-rangesegment-2Dview.hpp"
#include "tests/test-forall-rangestridesegment-view.hpp"
#include "tests/test-forall-listsegment-view.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallSegmentViewTest,
                            RangeSegmentForallView,
                            RangeSegmentForall2DView,
                            RangeStrideSegmentForallView,
                            ListSegmentForallView);

#endif  // __TEST_FORALL_SEGMENT_VIEW_HPP__
