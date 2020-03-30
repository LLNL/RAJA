//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for RangeSegment
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

template<typename T>
class RangeSegmentUnitTest : public ::testing::Test {};

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
                                 unsigned long long>;

TYPED_TEST_SUITE(RangeSegmentUnitTest, MyTypes);

template< typename T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
void NegativeRangeSegConstructorsTest()
{
}

template< typename T, typename std::enable_if<std::is_signed<T>::value>::type* = nullptr>
void NegativeRangeSegConstructorsTest()
{
  RAJA::TypedRangeSegment<T> r1(-10, 7);
  RAJA::TypedRangeSegment<T> r3(-13, -1);
  ASSERT_EQ(17, r1.size());
  ASSERT_EQ(12, r3.size());
#if !defined(RAJA_ENABLE_CUDA) && !defined(RAJA_ENABLE_HIP)
  ASSERT_ANY_THROW(RAJA::TypedRangeSegment<T> r2(T(0), T(-50)));
#endif
}

TYPED_TEST(RangeSegmentUnitTest, Constructors)
{
  RAJA::TypedRangeSegment<TypeParam> first(0, 10);
  RAJA::TypedRangeSegment<TypeParam> copied(first);

  ASSERT_EQ(first, copied);

  RAJA::TypedRangeSegment<TypeParam> moved(std::move(first));

  ASSERT_EQ(moved, copied);

  // Test exception when begin > end
#if !defined(RAJA_ENABLE_CUDA) && !defined(RAJA_ENABLE_HIP)
  ASSERT_ANY_THROW(RAJA::TypedRangeSegment<TypeParam> neg(20, 19));
#endif

  NegativeRangeSegConstructorsTest<TypeParam>();
}

TYPED_TEST(RangeSegmentUnitTest, Assignments)
{
  auto r = RAJA::TypedRangeSegment<TypeParam>(RAJA::Index_type(), 5);
  RAJA::TypedRangeSegment<TypeParam> seg1 = r;
  ASSERT_EQ(r, seg1);
  RAJA::TypedRangeSegment<TypeParam> seg2 = std::move(r);
  ASSERT_EQ(seg2, seg1);
}

TYPED_TEST(RangeSegmentUnitTest, Swaps)
{
  RAJA::TypedRangeSegment<TypeParam> r1(0, 5);
  RAJA::TypedRangeSegment<TypeParam> r2(1, 6);
  RAJA::TypedRangeSegment<TypeParam> r3(r1);
  RAJA::TypedRangeSegment<TypeParam> r4(r2);
  std::swap(r1, r2);
  ASSERT_EQ(r1, r4);
  ASSERT_EQ(r2, r3);
}

template< typename T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
void NegativeRangeSegIteratorsTest()
{
}

template< typename T, typename std::enable_if<std::is_signed<T>::value>::type* = nullptr>
void NegativeRangeSegIteratorsTest()
{
  RAJA::TypedRangeSegment<T> r3(-2, 100);
  ASSERT_EQ(T(-2), *r3.begin());
}

TYPED_TEST(RangeSegmentUnitTest, Iterators)
{
  RAJA::TypedRangeSegment<TypeParam> r1(0, 100);
  ASSERT_EQ(TypeParam(0), *r1.begin());
  ASSERT_EQ(TypeParam(99), *(--r1.end()));
  ASSERT_EQ(TypeParam(100), r1.end() - r1.begin());
  using difftype_t = decltype(std::distance(r1.begin(), r1.end()));
  ASSERT_EQ(difftype_t(100), std::distance(r1.begin(), r1.end()));
  ASSERT_EQ(difftype_t(100), r1.size());

  NegativeRangeSegIteratorsTest<TypeParam>();
}

TYPED_TEST(RangeSegmentUnitTest, Slices)
{
  auto r = RAJA::TypedRangeSegment<TypeParam>(0, 125);
  
  auto s = r.slice(10,100);

  ASSERT_EQ(TypeParam(10), *s.begin());
  ASSERT_EQ(TypeParam(110), *(s.end()));
}

TYPED_TEST(RangeSegmentUnitTest, Equality)
{
  auto r1 = RAJA::TypedRangeSegment<TypeParam>(0, 125);
  auto r2 = RAJA::TypedRangeSegment<TypeParam>(0, 125);

  ASSERT_EQ(r1, r2);

  auto r3 = RAJA::TypedRangeSegment<TypeParam>(10,15);

  ASSERT_NE(r1, r3);
}
