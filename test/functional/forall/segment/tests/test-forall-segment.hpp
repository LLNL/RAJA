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

//
// Strongly typed indexes
//
RAJA_INDEX_VALUE(IndexType, "IndexType");
RAJA_INDEX_VALUE_T(StrongInt, int, "StrongIntType");
RAJA_INDEX_VALUE_T(StrongULL, unsigned long long , "StrongULLType");

//
// Index types for segments
//
using SegmentIdxTypeList = camp::list<RAJA::Index_type,
                                      int,
                                      IndexType,
#if defined(RAJA_TEST_EXHAUSTIVE)
                                      StrongInt,
                                      unsigned int,
                                      short,
                                      unsigned short,
                                      long int,
                                      unsigned long,
                                      long long,
#endif
                                      StrongULL,
                                      unsigned long long>;

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
