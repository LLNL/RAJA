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
/// Source file containing tests for RAJAVec
///

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

TEST(RAJAVec, basic_test)
{
  RAJA::RAJAVec<int> a;
  ASSERT_TRUE(a.empty());
  ASSERT_EQ(0lu, a.size());
  a.push_back(5);
  ASSERT_FALSE(a.empty());
  ASSERT_EQ(5, *a.begin());
  ASSERT_EQ(5, *(a.end() - 1));
  a.push_front(10);
  ASSERT_EQ(10, *a.begin());
  ASSERT_EQ(5, *(a.end() - 1));
  a.resize(5, 20);
  ASSERT_EQ(20, a[2]);
  ASSERT_EQ(20, a[3]);
  ASSERT_EQ(20, a[4]);
  ASSERT_EQ(5lu, a.size());
  a.resize(1);
  ASSERT_EQ(1lu, a.size());
  a.resize(0);
  for (int i = 0; i < 100; ++i)
    a.push_back(i);
  ASSERT_EQ(100lu, a.size());
  auto b = a;
  b.resize(0);
  ASSERT_EQ(0lu, b.size());
  ASSERT_EQ(100lu, a.size());
  a.swap(b);
  ASSERT_EQ(0lu, a.size());
  ASSERT_EQ(100lu, b.size());
  RAJA::RAJAVec<int> c;
  for (int i = 0; i < 100; ++i)
    c.push_front(i);
  for (int i = 0; i < 100; ++i)
    ASSERT_EQ(c[i], b[99 - i]);
  ASSERT_EQ(c.data() + c.size(), c.end());
  ASSERT_EQ(c.data(), c.begin());
}
