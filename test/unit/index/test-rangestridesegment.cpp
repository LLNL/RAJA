//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for RangeSTrideSegment
///

#include "RAJA_test-base.hpp"

#include "RAJA_unit-test-types.hpp"

template<typename T>
class RangeStrideSegmentUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(RangeStrideSegmentUnitTest, UnitIndexTypes);


TYPED_TEST(RangeStrideSegmentUnitTest, Constructors)
{
    RAJA::TypedRangeStrideSegment<TypeParam> first(0, 10, 2);
    RAJA::TypedRangeStrideSegment<TypeParam> copied(first);
    ASSERT_EQ(first, copied);
    RAJA::TypedRangeStrideSegment<TypeParam> moved(std::move(first));
    ASSERT_EQ(moved, copied);
}

TYPED_TEST(RangeStrideSegmentUnitTest, Assignments)
{
    auto r = RAJA::make_strided_range<TypeParam>(static_cast<TypeParam>(0), 
                                                 static_cast<TypeParam>(5), 
                                                 static_cast<typename std::make_signed<TypeParam>::type>(3));
    RAJA::TypedRangeStrideSegment<TypeParam> seg1 = r;
    ASSERT_EQ(r, seg1);
    RAJA::TypedRangeStrideSegment<TypeParam> seg2 = std::move(r);
    ASSERT_EQ(seg2, seg1);
}

TYPED_TEST(RangeStrideSegmentUnitTest, Swaps)
{
    RAJA::TypedRangeStrideSegment<TypeParam> r1(0, 5, 2);
    RAJA::TypedRangeStrideSegment<TypeParam> r2(1, 6, 1);
    RAJA::TypedRangeStrideSegment<TypeParam> r3(r1);
    RAJA::TypedRangeStrideSegment<TypeParam> r4(r2);
    std::swap(r1, r2);
    ASSERT_EQ(r1, r4);
    ASSERT_EQ(r2, r3);
}

TYPED_TEST(RangeStrideSegmentUnitTest, Iterators)
{
    RAJA::TypedRangeStrideSegment<TypeParam> r1(0, 100, 4);
    ASSERT_EQ(TypeParam(0), *r1.begin());
    ASSERT_EQ(TypeParam(96), *(--r1.end()));
    using difftype_t = decltype(std::distance(r1.begin(), r1.end()));
    ASSERT_EQ(difftype_t(25), r1.end() - r1.begin());
    ASSERT_EQ(difftype_t(25), std::distance(r1.begin(), r1.end()));
    ASSERT_EQ(difftype_t(25), r1.size());
}

template< typename T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
void NegativeRangeStrideTestSizes()
{
}

template< typename T, typename std::enable_if<std::is_signed<T>::value>::type* = nullptr>
void NegativeRangeStrideTestSizes()
{
  RAJA::TypedRangeStrideSegment<T> segment16(-10, -2, 2);
  using difftype_t = decltype(std::distance(segment16.begin(), segment16.end()));
  ASSERT_EQ(segment16.size(), difftype_t(4));

  RAJA::TypedRangeStrideSegment<T> segment17(-5, 5, 2);
  ASSERT_EQ(segment17.size(), difftype_t(5));

  RAJA::TypedRangeStrideSegment<T> segment18(0, -5, 1);
  ASSERT_EQ(segment18.size(), difftype_t(0));
}

TYPED_TEST(RangeStrideSegmentUnitTest, Sizes)
{
  RAJA::TypedRangeStrideSegment<TypeParam> segment1(0, 20, 1);
  using difftype_t = decltype(std::distance(segment1.begin(), segment1.end()));
  ASSERT_EQ(segment1.size(), difftype_t(20));

  RAJA::TypedRangeStrideSegment<TypeParam> segment2(0, 20, 2);
  ASSERT_EQ(segment2.size(), difftype_t(10));

  RAJA::TypedRangeStrideSegment<TypeParam> segment3(0, 20, 4);
  ASSERT_EQ(segment3.size(), difftype_t(5));

  RAJA::TypedRangeStrideSegment<TypeParam> segment4(0, 20, 5);
  ASSERT_EQ(segment4.size(), difftype_t(4));

  RAJA::TypedRangeStrideSegment<TypeParam> segment5(0, 20, 10);
  ASSERT_EQ(segment5.size(), difftype_t(2));

  RAJA::TypedRangeStrideSegment<TypeParam> segment6(0, 20, 20);
  ASSERT_EQ(segment6.size(), difftype_t(1));

  // ROUNDOFFS
  RAJA::TypedRangeStrideSegment<TypeParam> segment7(0, 21, 2);
  ASSERT_EQ(segment7.size(), difftype_t(11));

  RAJA::TypedRangeStrideSegment<TypeParam> segment8(0, 21, 4);
  ASSERT_EQ(segment8.size(), difftype_t(6));

  RAJA::TypedRangeStrideSegment<TypeParam> segment9(0, 21, 5);
  ASSERT_EQ(segment9.size(), difftype_t(5));

  RAJA::TypedRangeStrideSegment<TypeParam> segment10(0, 21, 10);
  ASSERT_EQ(segment10.size(), difftype_t(3));

  RAJA::TypedRangeStrideSegment<TypeParam> segment11(0, 21, 20);
  ASSERT_EQ(segment11.size(), difftype_t(2));

  // PRIMES
  RAJA::TypedRangeStrideSegment<TypeParam> segment12(0, 7, 3);  // should produce 0,3,6
  ASSERT_EQ(segment12.size(), difftype_t(3));

  RAJA::TypedRangeStrideSegment<TypeParam> segment13(0, 13, 3);  // should produce 0,3,6,9,12
  ASSERT_EQ(segment13.size(), difftype_t(5));

  RAJA::TypedRangeStrideSegment<TypeParam> segment14(0, 17, 5);  // should produce 0,5,10,15
  ASSERT_EQ(segment14.size(), difftype_t(4));

  // NEGATIVE STRIDE
  RAJA::TypedRangeStrideSegment<TypeParam> segment15(0, 20, -2);
  ASSERT_EQ(segment15.size(), difftype_t(0));

  // NEGATIVE INDICES
  NegativeRangeStrideTestSizes<TypeParam>();
}

TYPED_TEST(RangeStrideSegmentUnitTest, Slices)
{
  auto r = RAJA::TypedRangeStrideSegment<TypeParam>(0, 20, 2);
  auto s = r.slice(0, 5);

  ASSERT_EQ(size_t(5), size_t(s.size()));
  ASSERT_EQ(TypeParam(0), *s.begin());
  ASSERT_EQ(TypeParam(10), *s.end());
}

TYPED_TEST(RangeStrideSegmentUnitTest, Equality)
{
  auto r1 = RAJA::TypedRangeStrideSegment<TypeParam>(0, 10, 1);
  auto r2 = RAJA::TypedRangeStrideSegment<TypeParam>(0, 10, 1);

  ASSERT_EQ(r1, r2);

  auto r3 = RAJA::TypedRangeStrideSegment<TypeParam>(1, 10, 1);

  ASSERT_TRUE( !(r1 == r3));
}
