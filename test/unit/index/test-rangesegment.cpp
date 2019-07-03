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

template <class T>
void RangeSegmentConstructorTest()
{
  RAJA::TypedRangeSegment<T> first(0, 10);
  RAJA::TypedRangeSegment<T> copied(first);

  ASSERT_EQ(first, copied);

  RAJA::TypedRangeSegment<T> moved(std::move(first));

  ASSERT_EQ(moved, copied);

  RAJA::TypedRangeSegment<T> r1(-10, 7);
  RAJA::TypedRangeSegment<T> r2(0, -50);
}

TEST(RangeSegmentTest, Constructors)
{
  // Default Range Segment Test
  RangeSegmentConstructorTest<RAJA::Index_type>();

  // Typed Range Segment Tests
  RangeSegmentConstructorTest<int>();
  RangeSegmentConstructorTest<float>();
  RangeSegmentConstructorTest<double>();
  RangeSegmentConstructorTest<short>();
  RangeSegmentConstructorTest<long>();
  RangeSegmentConstructorTest<unsigned int>();
}

TEST(RangeSegmentTest, Assignments)
{
  auto r = RAJA::RangeSegment(RAJA::Index_type(), 5);
  RAJA::RangeSegment seg1 = r;
  ASSERT_EQ(r, seg1);
  RAJA::RangeSegment seg2 = std::move(r);
  ASSERT_EQ(seg2, seg1);
}

TEST(RangeSegmentTest, Swaps)
{
  RAJA::RangeSegment r1(0, 5);
  RAJA::RangeSegment r2(1, 6);
  RAJA::RangeSegment r3(r1);
  RAJA::RangeSegment r4(r2);
  std::swap(r1, r2);
  ASSERT_EQ(r1, r4);
  ASSERT_EQ(r2, r3);
}

TEST(RangeSegmentTest, Iterators)
{
  RAJA::RangeSegment r1(0, 100);
  ASSERT_EQ(0, *r1.begin());
  ASSERT_EQ(99, *(--r1.end()));
  ASSERT_EQ(100, r1.end() - r1.begin());
  ASSERT_EQ(100, std::distance(r1.begin(), r1.end()));
  ASSERT_EQ(100, r1.size());
}

TEST(RangeSegmentTest, Slices)
{
  auto r = RAJA::RangeSegment(0, 125);

  auto s = r.slice(10,100);

  ASSERT_EQ(10, *s.begin());
  ASSERT_EQ(110, *(s.end()));
}

TEST(RangeSegmentTest, Equality)
{
  auto r1 = RAJA::RangeSegment(0, 125);
  auto r2 = RAJA::RangeSegment(0, 125);

  ASSERT_EQ(r1, r2);

  auto r3 = RAJA::RangeSegment(10,15);

  ASSERT_NE(r1, r3);
}
