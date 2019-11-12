//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for Span
///

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/Span.hpp"
#include "gtest/gtest.h"

template<typename T>
class SpanTest : public ::testing::Test {};

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

TYPED_TEST_CASE(SpanTest, MyTypes);

constexpr int DATA_SIZE = 4;
TYPED_TEST(SpanTest, simple)
{

  TypeParam data[DATA_SIZE];
  for (int i = 0; i < DATA_SIZE; i++)
    data[i] = static_cast<TypeParam>(i);

  auto span = RAJA::impl::make_span(data, DATA_SIZE);

  ASSERT_EQ(0, *span.begin());
  ASSERT_EQ(0, *span.data());
  ASSERT_EQ(3, *(span.data() + 3));
  ASSERT_EQ(3, *(span.end() - 1));

  ASSERT_EQ(0, *span.cbegin());
  ASSERT_EQ(0, *span.data());
  ASSERT_EQ(3, *(span.data() + 3));
  ASSERT_EQ(3, *(span.cend() - 1));

  auto const cspan = span;
  ASSERT_EQ(0, *cspan.begin());
  ASSERT_EQ(3, *(cspan.end() - 1));

  ASSERT_FALSE(cspan.empty());
  ASSERT_EQ(4, cspan.size());
  ASSERT_EQ(4, cspan.max_size());

  auto const empty = RAJA::impl::make_span((int*)nullptr, 0);
  ASSERT_TRUE(empty.empty());
  ASSERT_EQ(0, empty.size());

}

TYPED_TEST(SpanTest, slice)
{
  TypeParam data[DATA_SIZE];
  for (int i = 0; i < DATA_SIZE; i++)
    data[i] = static_cast<TypeParam>(i);

  auto span = RAJA::impl::make_span(data, DATA_SIZE);

  for (int i = 0; i < DATA_SIZE; i++)
  {
    for (int j = 0; j < DATA_SIZE - i; j++)
    {
      auto slice = span.slice(i, j);

      if(j == 0){
        ASSERT_TRUE(slice.empty());
        continue;
      }
      ASSERT_FALSE(slice.empty());

      ASSERT_EQ(i, *slice.begin());
      ASSERT_EQ(i, *slice.data());
      if(j >= 1){
        ASSERT_EQ(i+1, *(slice.data() + 1));
        ASSERT_EQ(i+j-1, *(slice.end() - 1));
      }

      ASSERT_EQ(i, *slice.cbegin());
      ASSERT_EQ(i, *slice.data());
      if(j >= 1){
        ASSERT_EQ(i+1, *(slice.data() + 1));
        ASSERT_EQ(i+j-1, *(slice.cend() - 1));
      }

      auto const cslice = slice;
      ASSERT_EQ(i, *cslice.begin());
      if(j >= 1){
        ASSERT_EQ(i+j-1, *(cslice.end() - 1));
      }

      ASSERT_FALSE(cslice.empty());
      ASSERT_EQ(j, cslice.size());
      ASSERT_EQ(j, cslice.max_size());
    }
  }
}
