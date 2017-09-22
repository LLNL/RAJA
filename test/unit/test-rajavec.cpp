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
