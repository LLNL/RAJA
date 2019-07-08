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

template<typename T>
class RangeSegmentTest : public ::testing::Test {};

using MyTypes = ::testing::Types<RAJA::Index_type,
                                 char, 
                                 unsigned char,
				 short,
				 unsigned short,
				 int, 
				 unsigned int,
				 long,
				 unsigned long,
				 long int,
				 unsigned long int,
				 long long,
				 unsigned long long,
				 double,
				 float>;

TYPED_TEST_CASE(RangeSegmentTest, MyTypes);

TYPED_TEST(RangeSegmentTest, Constructors)
{
  RAJA::TypedRangeSegment<TypeParam> first(0, 10);
  RAJA::TypedRangeSegment<TypeParam> copied(first);

  ASSERT_EQ(first, copied);

  RAJA::TypedRangeSegment<TypeParam> moved(std::move(first));

  ASSERT_EQ(moved, copied);

  if(std::is_signed<TypeParam>::value){
    RAJA::TypedRangeSegment<TypeParam> r1(-10, 7);
    RAJA::TypedRangeSegment<TypeParam> r2(0, -50);
    std::cout << r2.size() << std::endl;
    ASSERT_EQ(17, r1.size());
    ASSERT_EQ(-50, r2.size());
  }
}

TYPED_TEST(RangeSegmentTest, Assignments)
{
  auto r = RAJA::TypedRangeSegment<TypeParam>(RAJA::Index_type(), 5);
  RAJA::TypedRangeSegment<TypeParam> seg1 = r;
  ASSERT_EQ(r, seg1);
  RAJA::TypedRangeSegment<TypeParam> seg2 = std::move(r);
  ASSERT_EQ(seg2, seg1);
}

TYPED_TEST(RangeSegmentTest, Swaps)
{
  RAJA::TypedRangeSegment<TypeParam> r1(0, 5);
  RAJA::TypedRangeSegment<TypeParam> r2(1, 6);
  RAJA::TypedRangeSegment<TypeParam> r3(r1);
  RAJA::TypedRangeSegment<TypeParam> r4(r2);
  std::swap(r1, r2);
  ASSERT_EQ(r1, r4);
  ASSERT_EQ(r2, r3);
}

TYPED_TEST(RangeSegmentTest, Iterators)
{
  RAJA::TypedRangeSegment<TypeParam> r1(0, 100);
  ASSERT_EQ(0, *r1.begin());
  ASSERT_EQ(99, *(--r1.end()));
  ASSERT_EQ(100, r1.end() - r1.begin());
  ASSERT_EQ(100, std::distance(r1.begin(), r1.end()));
  ASSERT_EQ(100, r1.size());
}

TYPED_TEST(RangeSegmentTest, Slices)
{
  auto r = RAJA::TypedRangeSegment<TypeParam>(0, 125);
  
  auto s = r.slice(10,100);

  ASSERT_EQ(10, *s.begin());
  ASSERT_EQ(110, *(s.end()));
}

TYPED_TEST(RangeSegmentTest, Equality)
{
  auto r1 = RAJA::TypedRangeSegment<TypeParam>(0, 125);
  auto r2 = RAJA::TypedRangeSegment<TypeParam>(0, 125);

  ASSERT_EQ(r1, r2);

  auto r3 = RAJA::TypedRangeSegment<TypeParam>(10,15);

  ASSERT_NE(r1, r3);
}
