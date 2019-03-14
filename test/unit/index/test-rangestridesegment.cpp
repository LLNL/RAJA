//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

TEST(RangeStrideSegment, Constructors)
{
    RAJA::RangeStrideSegment first(0, 10, 2);
    RAJA::RangeStrideSegment copied(first);
    ASSERT_EQ(first, copied);
    RAJA::RangeStrideSegment moved(std::move(first));
    ASSERT_EQ(moved, copied);
}

TEST(RangeStrideSegment, Assignments)
{
    auto r = RAJA::make_strided_range(RAJA::Index_type(), 5, 3);
    RAJA::RangeStrideSegment seg1 = r;
    ASSERT_EQ(r, seg1);
    RAJA::RangeStrideSegment seg2 = std::move(r);
    ASSERT_EQ(seg2, seg1);
}

TEST(RangeStrideSegment, Swaps)
{
    RAJA::RangeStrideSegment r1(0, 5, 2);
    RAJA::RangeStrideSegment r2(1, 6, 1);
    RAJA::RangeStrideSegment r3(r1);
    RAJA::RangeStrideSegment r4(r2);
    std::swap(r1, r2);
    ASSERT_EQ(r1, r4);
    ASSERT_EQ(r2, r3);
}

TEST(RangeStrideSegment, Iterators)
{
    RAJA::RangeStrideSegment r1(0, 100, 4);
    ASSERT_EQ(0, *r1.begin());
    ASSERT_EQ(96, *(--r1.end()));
    ASSERT_EQ(25, r1.end() - r1.begin());
    ASSERT_EQ(25, std::distance(r1.begin(), r1.end()));
    ASSERT_EQ(25, r1.size());
}

TEST(RangeStrideSegment, Sizes)
{
  RAJA::RangeStrideSegment segment1(0, 20, 1);
  ASSERT_EQ(segment1.size(), 20);

  RAJA::RangeStrideSegment segment2(0, 20, 2);
  ASSERT_EQ(segment2.size(), 10);

  RAJA::RangeStrideSegment segment3(0, 20, 4);
  ASSERT_EQ(segment3.size(), 5);

  RAJA::RangeStrideSegment segment4(0, 20, 5);
  ASSERT_EQ(segment4.size(), 4);

  RAJA::RangeStrideSegment segment5(0, 20, 10);
  ASSERT_EQ(segment5.size(), 2);

  RAJA::RangeStrideSegment segment6(0, 20, 20);
  ASSERT_EQ(segment6.size(), 1);

  // ROUNDOFFS
  RAJA::RangeStrideSegment segment7(0, 21, 2);
  ASSERT_EQ(segment7.size(), 11);

  RAJA::RangeStrideSegment segment8(0, 21, 4);
  ASSERT_EQ(segment8.size(), 6);

  RAJA::RangeStrideSegment segment9(0, 21, 5);
  ASSERT_EQ(segment9.size(), 5);

  RAJA::RangeStrideSegment segment10(0, 21, 10);
  ASSERT_EQ(segment10.size(), 3);

  RAJA::RangeStrideSegment segment11(0, 21, 20);
  ASSERT_EQ(segment11.size(), 2);

  // PRIMES
  RAJA::RangeStrideSegment segment12(0, 7, 3);  // should produce 0,3,6
  ASSERT_EQ(segment12.size(), 3);

  RAJA::RangeStrideSegment segment13(0, 13, 3);  // should produce 0,3,6,9,12
  ASSERT_EQ(segment13.size(), 5);

  RAJA::RangeStrideSegment segment14(0, 17, 5);  // should produce 0,5,10,15
  ASSERT_EQ(segment14.size(), 4);
}

TEST(RangeStrideSegment, Slices)
{
  auto r = RAJA::RangeStrideSegment(0, 20, 2);
  auto s = r.slice(0, 5);

  ASSERT_EQ(5, s.size());
  ASSERT_EQ(0, *s.begin());
  ASSERT_EQ(10, *s.end());
}

TEST(RangeStrideSegment, Equality)
{
  auto r1 = RAJA::RangeStrideSegment(0, 10, 1);
  auto r2 = RAJA::RangeStrideSegment(0, 10, 1);

  ASSERT_EQ(r1, r2);

  auto r3 = RAJA::RangeStrideSegment(1, 10, 1);

  ASSERT_TRUE( !(r1 == r3));
}
