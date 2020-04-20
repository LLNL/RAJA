//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_SEGMENT_VIEW_HPP__
#define __TEST_FORALL_SEGMENT_VIEW_HPP__

#include "RAJA/RAJA.hpp"

#include "../../test-forall-utils.hpp"

TYPED_TEST_SUITE_P(ForallSegmentViewTest);
template <typename T>
class ForallSegmentViewTest : public ::testing::Test
{
};

#include "test-forall-rangesegment-view.hpp"
#include "test-forall-rangesegment-2Dview.hpp"
#include "test-forall-rangestridesegment-view.hpp"
#include "test-forall-listsegment-view.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallSegmentViewTest,
                            RangeSegmentForallView,
                            RangeSegmentForall2DView,
                            RangeStrideSegmentForallView,
                            ListSegmentForallView);

#endif  // __TEST_FORALL_SEGMENT_VIEW_HPP__
