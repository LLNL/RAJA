//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for internal RAJA Iterators
///

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

TEST(NumericIterator, simple)
{
  RAJA::Iterators::numeric_iterator<> i;
  ASSERT_EQ(0, *i);
  ++i;
  ASSERT_EQ(1, *i);
  --i;
  ASSERT_EQ(0, *i);
  ASSERT_EQ(0, *i++);
  ASSERT_EQ(1, *i);
  ASSERT_EQ(1, *i--);
  ASSERT_EQ(0, *i);
  i += 2;
  ASSERT_EQ(2, *i);
  i -= 1;
  ASSERT_EQ(1, *i);
  RAJA::Iterators::numeric_iterator<> five(5);
  i += five;
  ASSERT_EQ(6, *i);
  i -= five;
  ASSERT_EQ(1, *i);
  RAJA::Iterators::numeric_iterator<> three(3);
  ASSERT_LE(three, three);
  ASSERT_LE(three, five);
  ASSERT_LT(three, five);
  ASSERT_GE(five, three);
  ASSERT_GT(five, three);
  ASSERT_NE(five, three);
  ASSERT_EQ(three + 2, five);
  ASSERT_EQ(2 + three, five);
  ASSERT_EQ(five - 2, three);
  ASSERT_EQ(8 - five, three);
}

TEST(StridedNumericIterator, simple)
{
  RAJA::Iterators::strided_numeric_iterator<> i(0, 2);
  ASSERT_EQ(0, *i);
  ++i;
  ASSERT_EQ(2, *i);
  --i;
  ASSERT_EQ(0, *i);
  i += 2;
  ASSERT_EQ(4, *i);
  i -= 1;
  ASSERT_EQ(2, *i);
  RAJA::Iterators::strided_numeric_iterator<> three(3, 2);
  RAJA::Iterators::strided_numeric_iterator<> five(5, 2);
  ASSERT_LE(three, three);
  ASSERT_LE(three, five);
  ASSERT_LT(three, five);
  ASSERT_GE(five, three);
  ASSERT_GT(five, three);
  ASSERT_NE(five, three);
  ASSERT_EQ(three + 1, five);
  ASSERT_EQ(five - 1, three);
}
