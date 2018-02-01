//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

///
/// Source file containing tests for RAJA index set mechanics.
///

#include "gtest/gtest.h"
#include "RAJA/RAJA.hpp"

#include <iostream>

namespace RAJA
{
  template <typename T>
  void PrintTo(const TypedRangeSegment<T>& s, ::std::ostream* os) {
    *os << '[' << (*s.begin()) << ',' << (*(s.end() - 1)) << ')';
  }

  template <typename T>
  void PrintTo(const TypedRangeStrideSegment<T>& s, ::std::ostream* os) {
    *os << '[' << (*s.begin()) << ',' << (*(s.end() - 1)) << ')' << " by " << (*(s.begin() + 1) - *s.begin());
  }

  template <typename T>
  void PrintTo(const TypedListSegment<T>& s, ::std::ostream* os) {
    *os << "Address: " << &(*s.begin()) << "; Size: " << s.size() << "; Ownership: " << (s.getIndexOwnership() == RAJA::Owned ? "Owned" : "Unowned");
  }

}

TEST(RangeStrideSegmentTest, sizes_no_roundoff)
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
}


TEST(RangeStrideSegmentTest, sizes_roundoff1)
{
  RAJA::RangeStrideSegment segment2(0, 21, 2);
  ASSERT_EQ(segment2.size(), 11);

  RAJA::RangeStrideSegment segment3(0, 21, 4);
  ASSERT_EQ(segment3.size(), 6);

  RAJA::RangeStrideSegment segment4(0, 21, 5);
  ASSERT_EQ(segment4.size(), 5);

  RAJA::RangeStrideSegment segment5(0, 21, 10);
  ASSERT_EQ(segment5.size(), 3);

  RAJA::RangeStrideSegment segment6(0, 21, 20);
  ASSERT_EQ(segment6.size(), 2);
}


TEST(RangeStrideSegmentTest, sizes_primes)
{
  RAJA::RangeStrideSegment segment1(0, 7, 3);  // should produce 0,3,6
  ASSERT_EQ(segment1.size(), 3);

  RAJA::RangeStrideSegment segment2(0, 13, 3); // should produce 0,3,6,9,12
  ASSERT_EQ(segment2.size(), 5);

  RAJA::RangeStrideSegment segment3(0, 17, 5); // should produce 0,5,10,15
  ASSERT_EQ(segment3.size(), 4);
}

TEST(RangeStrideSegmentTest, basic_types)
{
  RAJA::TypedRangeStrideSegment<signed char> segment1(0, 31, 3);
  ASSERT_EQ(segment1.size(), 11);

  RAJA::TypedRangeStrideSegment<short> segment2(0, 31, 3);
  ASSERT_EQ(segment2.size(), 11);

  RAJA::TypedRangeStrideSegment<int> segment3(0, 31, 3);
  ASSERT_EQ(segment3.size(), 11);

  RAJA::TypedRangeStrideSegment<long> segment4(0, 31, 3);
  ASSERT_EQ(segment3.size(), 11);

  RAJA::TypedRangeStrideSegment<long long> segment5(0, 31, 3);
  ASSERT_EQ(segment3.size(), 11);
}

RAJA_INDEX_VALUE(StrongType, "StrongType");

TEST(RangeStrideSegmentTest, strongly_typed)
{
  RAJA::TypedRangeStrideSegment<StrongType> segment1(0, 7, 3);  // should produce 0,3,6
  ASSERT_EQ(segment1.size(), 3);

  RAJA::TypedRangeStrideSegment<StrongType> segment2(0, 13, 3); // should produce 0,3,6,9,12
  ASSERT_EQ(segment2.size(), 5);

  RAJA::TypedRangeStrideSegment<StrongType> segment3(0, 17, 5); // should produce 0,5,10,15
  ASSERT_EQ(segment3.size(), 4);

  std::vector<int> values(7, 0);
  RAJA::forall<RAJA::seq_exec>(segment1,
      [&](StrongType i){
      values[*i] = 1;
  });

  ASSERT_EQ(values[0], 1);
  ASSERT_EQ(values[1], 0);
  ASSERT_EQ(values[2], 0);
  ASSERT_EQ(values[3], 1);
  ASSERT_EQ(values[4], 0);
  ASSERT_EQ(values[5], 0);
  ASSERT_EQ(values[6], 1);
}


TEST(RangeStrideSegmentTest, sizes_reverse_no_roundoff)
{
  RAJA::RangeStrideSegment segment1(19, -1, -1);
  ASSERT_EQ(segment1.size(), 20);

  RAJA::RangeStrideSegment segment2(19, -1, -2);
  ASSERT_EQ(segment2.size(), 10);

  RAJA::RangeStrideSegment segment3(19, -1, -4);
  ASSERT_EQ(segment3.size(), 5);

  RAJA::RangeStrideSegment segment4(19, -1, -5);
  ASSERT_EQ(segment4.size(), 4);

  RAJA::RangeStrideSegment segment5(19, -1, -10);
  ASSERT_EQ(segment5.size(), 2);

  RAJA::RangeStrideSegment segment6(19, -1, -20);
  ASSERT_EQ(segment6.size(), 1);
}


TEST(RangeStrideSegmentTest, sizes_reverse_roundoff1)
{
  RAJA::RangeStrideSegment segment2(20, -1, -2);
  ASSERT_EQ(segment2.size(), 11);

  RAJA::RangeStrideSegment segment3(20, -1, -4);
  ASSERT_EQ(segment3.size(), 6);

  RAJA::RangeStrideSegment segment4(20, -1, -5);
  ASSERT_EQ(segment4.size(), 5);

  RAJA::RangeStrideSegment segment5(20, -1, -10);
  ASSERT_EQ(segment5.size(), 3);

  RAJA::RangeStrideSegment segment6(20, -1, -20);
  ASSERT_EQ(segment6.size(), 2);
}


TEST(RangeStrideSegmentTest, values_forward_stride1)
{
  RAJA::Index_type expected[] = {0,1,2,3,4,5};
  RAJA::RangeStrideSegment segment(0,6,1);

  ASSERT_EQ(segment.size(), 6);

  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  }

  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  }
}

TEST(RangeStrideSegmentTest, values_forward_stride3)
{
  RAJA::Index_type expected[] = {0,3,6,9,12};
  RAJA::RangeStrideSegment segment(0,14,3);

  ASSERT_EQ(segment.size(), 5);

  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  }

  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  }
}

TEST(RangeStrideSegmentTest, values_reverse_stride1)
{
  RAJA::Index_type expected[] = {5,4,3,2,1,0};
  RAJA::RangeStrideSegment segment(5,-1,-1);

  ASSERT_EQ(segment.size(), 6);

  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  }

  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  }
}


TEST(RangeStrideSegmentTest, values_reverse_stride1_negative)
{
  RAJA::Index_type expected[] = {-10,-11,-12,-13};
  RAJA::RangeStrideSegment segment(-10,-14,-1);

  ASSERT_EQ(segment.size(), 4);

  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  }

  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  }
}


TEST(RangeStrideSegmentTest, zero_size)
{
  RAJA::RangeStrideSegment segment(3,2,1);

  ASSERT_EQ(segment.size(), 0);

}

TEST(RangeStrideSegmentTest, zero_size_reverse)
{
  RAJA::RangeStrideSegment segment(-3, 3,-1);

  ASSERT_EQ(segment.size(), 0);

}



TEST(RangeStrideSegmentTest, forall_values_forward_stride3)
{
  RAJA::Index_type expected[] = {0,3,6,9,12};
  RAJA::RangeStrideSegment segment(0,14,3);

  ASSERT_EQ(segment.size(), 5);

  for(RAJA::Index_type i = 0;i < segment.size();++ i)
  {
    ASSERT_EQ(segment.begin()[i], expected[i]);
  }

  size_t j = 0;


  for(auto i = segment.begin();i < segment.end();++i)
  {
    ASSERT_EQ(*i, expected[j++]);
  }

  ASSERT_EQ((RAJA::Index_type)j, segment.size());


  j = 0;

  RAJA::forall<RAJA::seq_exec>(segment, [&](RAJA::Index_type i)
  {
    ASSERT_EQ(i, expected[j++]);
  });


  ASSERT_EQ((RAJA::Index_type)j, segment.size());
}


TEST(RangeStrideSegmentTest, forall_values_reverse_stride5)
{
  RAJA::Index_type expected[] = {7,2,-3,-8};
  RAJA::RangeStrideSegment segment(7,-11,-5);

  ASSERT_EQ(segment.size(), 4);

  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  }

  size_t j = 0;

  for(auto i = segment.begin();i < segment.end();++i)
  {
    ASSERT_EQ(*i, expected[j++]);
  }

  ASSERT_EQ((RAJA::Index_type)j, segment.size());

  j = 0;

  RAJA::forall<RAJA::seq_exec>(segment, [&](RAJA::Index_type i)
  {
    ASSERT_EQ(i, expected[j++]);
  });

  ASSERT_EQ((RAJA::Index_type)j, segment.size());
}


TEST(RangeStrideSegmentTest, iterator_begin_end)
{
  RAJA::RangeStrideSegment segment(7,-11,-5);

  auto begin1 = segment.begin();
  auto begin2 = std::begin(segment);
  ASSERT_EQ(begin1, begin2);

  auto end1 = segment.end();
  auto end2 = std::end(segment);
  ASSERT_EQ(end1, end2);

}


TEST(RangeStrideSegmentTest, iterator_distance)
{
  {
    RAJA::RangeStrideSegment segment1(0,10,1);
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 10);
  }

  {
    RAJA::RangeStrideSegment segment1(10,20,1);
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 10);
  }

  {
    RAJA::RangeStrideSegment segment1(0,5,2);
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 3);
  }

  {
    RAJA::RangeStrideSegment segment1(10,20,2);
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 5);
  }

  {
    RAJA::RangeStrideSegment segment1(20,10,-2);
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 5);
  }

  {
    RAJA::RangeStrideSegment segment1(-10,10,3);
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 7);
  }


  {
    RAJA::RangeStrideSegment segment1(10,-10,-7);
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 3);
  }
}

TEST(SegmentTest, constructors)
{
  {
    RAJA::RangeStrideSegment first(0, 10, 2);
    RAJA::RangeStrideSegment copied(first);
    ASSERT_EQ(first, copied);
    RAJA::RangeStrideSegment moved(std::move(first));
    ASSERT_EQ(moved, copied);
  }

  {
    RAJA::RangeSegment first(0, 10);
    RAJA::RangeSegment copied(first);
    ASSERT_EQ(first, copied);
    RAJA::RangeSegment moved(std::move(first));
    ASSERT_EQ(moved, copied);
  }

  {
    RAJA::ListSegment first(RAJA::make_range(0,10));
    ASSERT_EQ(RAJA::Owned, first.getIndexOwnership());

    RAJA::ListSegment copied(first);
    ASSERT_EQ(RAJA::Owned, copied.getIndexOwnership());

    ASSERT_EQ(first, copied);
    RAJA::ListSegment moved(std::move(first));
    ASSERT_EQ(moved, copied);

    RAJA::ListSegment empty(nullptr, 100);
    RAJA::ListSegment empty2(first.begin(), -5);
    ASSERT_EQ(empty, empty2);
  }
}

TEST(SegmentTest, assignments)
{
  {
    auto r =  RAJA::make_range(RAJA::Index_type(), 5);
    RAJA::RangeSegment seg1 = r;
    ASSERT_EQ(r, seg1);
    RAJA::RangeSegment seg2 = std::move(r);
    ASSERT_EQ(seg2, seg1);
  }
  {
    auto r = RAJA::make_strided_range(RAJA::Index_type(), 5, 3);
    RAJA::RangeStrideSegment seg1 = r;
    ASSERT_EQ(r, seg1);
    RAJA::RangeStrideSegment seg2 = std::move(r);
    ASSERT_EQ(seg2, seg1);
  }
  {
    RAJA::Index_type vals[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    RAJA::ListSegment r(vals, 5, RAJA::Unowned);
    ASSERT_EQ(RAJA::Unowned, r.getIndexOwnership());

    RAJA::ListSegment seg1 = r;
    ASSERT_EQ(r, seg1);
    RAJA::ListSegment seg2 = std::move(r);
    ASSERT_EQ(seg2, seg1);
  }
}

TEST(SegmentTest, swaps)
{
  {
    RAJA::RangeSegment r1(0, 5);
    RAJA::RangeSegment r2(1, 6);
    RAJA::RangeSegment r3(r1);
    RAJA::RangeSegment r4(r2);
    std::swap(r1, r2);
    ASSERT_EQ(r1, r4);
    ASSERT_EQ(r2, r3);
  }
  {
    RAJA::RangeStrideSegment r1(0, 5, 2);
    RAJA::RangeStrideSegment r2(1, 6, 1);
    RAJA::RangeStrideSegment r3(r1);
    RAJA::RangeStrideSegment r4(r2);
    std::swap(r1, r2);
    ASSERT_EQ(r1, r4);
    ASSERT_EQ(r2, r3);
  }
  {
    RAJA::Index_type vals[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    RAJA::ListSegment r1(vals, 5, RAJA::Unowned);
    RAJA::ListSegment r2(vals + 5, 5, RAJA::Unowned);
    ASSERT_NE(r1, r2);
    RAJA::ListSegment r3(r1);
    RAJA::ListSegment r4(r2);
    std::swap(r1, r2);
    ASSERT_EQ(r1, r4);
    ASSERT_EQ(r2, r3);
  }

}

TEST(SegmentTest, iterators)
{
  {
    RAJA::RangeSegment r1(0, 100);
    ASSERT_EQ(0, *r1.begin());
    ASSERT_EQ(99, *(--r1.end()));
    ASSERT_EQ(100, r1.end() - r1.begin());
    ASSERT_EQ(100, std::distance(r1.begin(), r1.end()));
    ASSERT_EQ(100, r1.size());
  }
  {
    RAJA::RangeStrideSegment r1(0, 100, 4);
    ASSERT_EQ(0, *r1.begin());
    ASSERT_EQ(96, *(--r1.end()));
    ASSERT_EQ(25, r1.end() - r1.begin());
    ASSERT_EQ(25, std::distance(r1.begin(), r1.end()));
    ASSERT_EQ(25, r1.size());
  }
  {
    RAJA::Index_type data[5] = { 1, 3, 5, 7, 9 };
    RAJA::ListSegment r1(data, 5);
    ASSERT_EQ(1, *r1.begin());
    ASSERT_EQ(9, *(r1.end() - 1));
    ASSERT_EQ(5, r1.end() - r1.begin());
    ASSERT_EQ(5, std::distance(r1.begin(), r1.end()));
    ASSERT_EQ(5, r1.size());
    ASSERT_FALSE(r1.indicesEqual(nullptr, 10));
    ASSERT_FALSE(r1.indicesEqual(&(*r1.begin()) + 1, r1.size()));
  }
}
