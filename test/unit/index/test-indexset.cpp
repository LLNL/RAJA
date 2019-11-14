//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for IndexSet class.
///

#include "gtest/gtest.h"

#include "RAJA/RAJA.hpp"

TEST(IndexSetUnitTest, Empty)
{
  RAJA::TypedIndexSet<> is;
  ASSERT_EQ(0, is.size());
  ASSERT_EQ(is.begin(), is.end());

  RAJA::TypedIndexSet<> is2;
  ASSERT_EQ(is2.size(), is.size());
  is.swap(is2);
  ASSERT_EQ(is2.size(), is.size());
}

TEST(IndexSetUnitTest, ConstructAndCompare)
{
  using RangeSegType = RAJA::TypedRangeSegment<int>;
  using RIndexSetType = RAJA::TypedIndexSet<RangeSegType>;
  RIndexSetType isr;
  ASSERT_EQ(1, isr.getNumTypes());
  isr.push_back(RangeSegType(1, 3));
  isr.push_front(RangeSegType(0, 1));
  ASSERT_EQ(2, isr.size()); 
  ASSERT_EQ(3, isr.getLength()); 
  const RangeSegType& rs0 = isr.getSegment<const RangeSegType>(0);
  const RangeSegType& rs1 = isr.getSegment<const RangeSegType>(1);
  ASSERT_EQ(1, rs0.size());
  ASSERT_EQ(2, rs1.size());
  ASSERT_TRUE(isr.compareSegmentById(0, isr));
  ASSERT_TRUE(isr.compareSegmentById(1, isr));

  RIndexSetType isr2;
  isr2.push_back(RangeSegType(0, 3));
  ASSERT_TRUE(isr != isr2);
  ASSERT_FALSE(isr == isr2);
  ASSERT_NE(isr.size(), isr2.size());
  ASSERT_EQ(isr.getLength(), isr2.getLength());

  using ListSegType = RAJA::TypedListSegment<int>; 
  using RLIndexSetType = RAJA::TypedIndexSet<RangeSegType, ListSegType>;
  RLIndexSetType isrl;
  ASSERT_EQ(2, isrl.getNumTypes());
  int idx[ ] = {0, 2, 4, 5};
  ListSegType lseg(idx, 4); 
  isrl.push_back(lseg);
  isrl.push_back(RangeSegType(6, 8));
  ASSERT_EQ(2, isrl.size()); 
  ASSERT_EQ(6, isrl.getLength()); 
  const ListSegType ls0 = isrl.getSegment<const ListSegType>(0);
  const RangeSegType rs11 = isrl.getSegment<const RangeSegType>(1);
  ASSERT_EQ(4, ls0.size());
  ASSERT_EQ(2, rs11.size());

  ASSERT_FALSE(isrl.compareSegmentById(0, isr));
  ASSERT_FALSE(isr.compareSegmentById(1, isrl));

  RIndexSetType isr3(isr);
  RLIndexSetType isrl3 = isrl;
  ASSERT_TRUE(isr == isr3);
  ASSERT_FALSE(isrl != isrl3);
  ASSERT_FALSE(isr3 == isrl3);
  ASSERT_TRUE(isr3 != isrl3);
}

TEST(IndexSetUnitTest, Swap)
{
  using RangeSegType = RAJA::TypedRangeSegment<int>;
  using RIndexSetType = RAJA::TypedIndexSet<RangeSegType>;
  RIndexSetType iset1;
  RangeSegType range(0, 10);
  iset1.push_back(range);
  iset1.push_back_nocopy(&range);
  iset1.push_front(range);
  iset1.push_front_nocopy(&range);
  RIndexSetType iset2;

  ASSERT_EQ(4, iset1.size());
  ASSERT_EQ(40, iset1.getLength());
  ASSERT_EQ(0, iset2.size());
  ASSERT_EQ(0, iset2.getLength());

  iset1.swap(iset2);

  ASSERT_EQ(4, iset2.size());
  ASSERT_EQ(40, iset2.getLength());
  ASSERT_EQ(0, iset1.size());
  ASSERT_EQ(0, iset1.getLength());
}

#if 0
TEST(IndexSetUnitTest, CompareSegments)
{
}

TEST(IndexSetUnitTest, Slice)
{
}

TEST(IndexSetUnitTest, CUDAAccess)
{
}

#if !defined(RAJA_COMPILER_XLC12)
TEST_F(IndexSetUnitTest, conditionalOperation_even_indices)
{

  RAJA::RAJAVec<RAJA::Index_type> even_indices;
  getIndicesConditional(even_indices, index_sets_[0], [](RAJA::Index_type idx) {
    return !(idx % 2);
  });

  RAJA::RAJAVec<RAJA::Index_type> ref_even_indices;
  for (size_t i = 0; i < is_indices.size(); ++i) {
    RAJA::Index_type idx = is_indices[i];
    if (idx % 2 == 0) {
      ref_even_indices.push_back(idx);
    }
  }

  EXPECT_EQ(even_indices.size(), ref_even_indices.size());
  for (size_t i = 0; i < ref_even_indices.size(); ++i) {
    EXPECT_EQ(even_indices[i], ref_even_indices[i]);
  }
}

TEST_F(IndexSetUnitTest, conditionalOperation_lt300_indices)
{
  RAJA::RAJAVec<RAJA::Index_type> lt300_indices;
  getIndicesConditional(lt300_indices,
                        index_sets_[0],
                        [](RAJA::Index_type idx) { return (idx < 300); });

  RAJA::RAJAVec<RAJA::Index_type> ref_lt300_indices;
  for (size_t i = 0; i < is_indices.size(); ++i) {
    RAJA::Index_type idx = is_indices[i];
    if (idx < 300) {
      ref_lt300_indices.push_back(idx);
    }
  }

  EXPECT_EQ(lt300_indices.size(), ref_lt300_indices.size());
  for (size_t i = 0; i < ref_lt300_indices.size(); ++i) {
    EXPECT_EQ(lt300_indices[i], ref_lt300_indices[i]);
  }
}
#endif  // !defined(RAJA_COMPILER_XLC12)
#endif
