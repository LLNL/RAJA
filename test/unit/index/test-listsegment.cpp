//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for ListSegment
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"
#include <vector>

template<typename T>
class ListSegmentUnitTest : public ::testing::Test {};

using MyTypes = ::testing::Types<RAJA::Index_type,
                                 int,
#if defined(RAJA_TEST_EXHAUSTIVE)
                                 unsigned int,
                                 char,
                                 unsigned char,
                                 short,
                                 unsigned short,
                                 long,
                                 unsigned long,
                                 long int,
                                 unsigned long int,
#endif
                                 unsigned long long>;

TYPED_TEST_SUITE(ListSegmentUnitTest, MyTypes);


TYPED_TEST(ListSegmentUnitTest, Constructors)
{
  std::vector<TypeParam> idx;
  for (TypeParam i = 0; i < 5; ++i){
    idx.push_back(i);
  }

  RAJA::TypedListSegment<TypeParam> list1( &idx[0], idx.size());
  RAJA::TypedListSegment<TypeParam> copied(list1);

  ASSERT_EQ(list1, copied);

  RAJA::TypedListSegment<TypeParam> moved(std::move(list1));

  ASSERT_EQ(moved, copied);

  RAJA::TypedListSegment<TypeParam> container(idx);

  ASSERT_EQ(list1, container); 
}

TYPED_TEST(ListSegmentUnitTest, Swaps)
{
  std::vector<TypeParam> idx1;
  std::vector<TypeParam> idx2;
  for (TypeParam i = 0; i < 5; ++i){
    idx1.push_back(i);
    idx2.push_back(i+5);
  }

  RAJA::TypedListSegment<TypeParam> list1( idx1 );
  RAJA::TypedListSegment<TypeParam> list2( idx2 );
  auto list3 = RAJA::TypedListSegment<TypeParam>(list1);
  auto list4 = RAJA::TypedListSegment<TypeParam>(list2);

  list1.swap(list2);

  ASSERT_EQ(list2, list3);
  ASSERT_EQ(list1, list4);

  std::swap(list1, list2);

  ASSERT_EQ(list1, list3);
  ASSERT_EQ(list2, list4);
}

TYPED_TEST(ListSegmentUnitTest, Equality)
{
  std::vector<TypeParam> idx1{5,3,1,2};
  RAJA::TypedListSegment<TypeParam> list( idx1 );

  std::vector<TypeParam> idx2{2,1,3,5};
  
  ASSERT_EQ(list.indicesEqual( &idx2.begin()[0], idx2.size() ), false);

  std::reverse( idx2.begin(), idx2.end() );

  ASSERT_EQ(list.indicesEqual( &idx2.begin()[0], idx2.size() ), true);
}

TYPED_TEST(ListSegmentUnitTest, Iterators)
{
  std::vector<TypeParam> idx1{5,3,1,2};
  RAJA::TypedListSegment<TypeParam> list( idx1 );

  ASSERT_EQ(TypeParam(5), *list.begin());
  ASSERT_EQ(TypeParam(2), *(list.end()-1));

  ASSERT_EQ(4, list.size());
}

