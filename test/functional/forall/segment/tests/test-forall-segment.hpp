//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_SEGMENT_HPP__
#define __TEST_FORALL_SEGMENT_HPP__

#include "RAJA/RAJA.hpp"

#include "../../test-forall-utils.hpp"

TYPED_TEST_SUITE_P(ForallSegmentTest);
template <typename T>
class ForallSegmentTest : public ::testing::Test
{
};

#include "test-forall-rangesegment.hpp"
#include "test-forall-rangestridesegment.hpp"
#include "test-forall-listsegment.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallSegmentTest,
                            RangeSegmentForall,
                            RangeStrideSegmentForall,
                            ListSegmentForall);

#endif  // __TEST_FORALL_SEGMENT_HPP__
