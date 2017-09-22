//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for internal RAJA Iterators
///

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

TEST(BaseIterator, simple)
{
  RAJA::Iterators::base_iterator<int> a;
  RAJA::Iterators::base_iterator<int> two(2);
  ASSERT_LT(a, two);
  ASSERT_LE(a, two);
  ASSERT_LE(a, a);
  ASSERT_EQ(a, a);
  ASSERT_GE(two, a);
  ASSERT_GT(two, a);
  ASSERT_NE(two, a);
  RAJA::Iterators::base_iterator<int> b(a);
  ASSERT_EQ(a, b);
}

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
