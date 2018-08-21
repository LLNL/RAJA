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

#include "buildIndexSet.hpp"

#include "RAJA/RAJA.hpp"

class IndexSetTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      buildIndexSet(index_sets_, static_cast<IndexSetBuildMethod>(ibuild));
    }

    getIndices(is_indices, index_sets_[0]);
  }

  RAJA::RAJAVec<RAJA::Index_type> is_indices;
  UnitIndexSet index_sets_[NumBuildMethods];
};

TEST_F(IndexSetTest, IndexSetEquality)
{
  for (unsigned ibuild = 1; ibuild < NumBuildMethods; ++ibuild) {
    EXPECT_EQ(index_sets_[ibuild], index_sets_[1]);
  }
}

#if !defined(RAJA_COMPILER_XLC12)
TEST_F(IndexSetTest, conditionalOperation_even_indices)
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

TEST_F(IndexSetTest, conditionalOperation_lt300_indices)
{
  RAJA::RAJAVec<RAJA::Index_type> lt300_indices;
  getIndicesConditional(lt300_indices, index_sets_[0], [](RAJA::Index_type idx) {
      return (idx < 300);
  });

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
#endif // !defined(RAJA_COMPILER_XLC12)

TEST(IndexSet, empty)
{
  RAJA::TypedIndexSet<> is;
  ASSERT_EQ(0, is.size());
  ASSERT_EQ(is.begin(), is.end());
  RAJA::TypedIndexSet<> is2;
  ASSERT_EQ(is2.size(), is.size());
  is.swap(is2);
  ASSERT_EQ(is2.size(), is.size());
}

TEST(IndexSet, compare)
{
  using RangeIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment>;
  RangeIndexSet is1, is2;
  is1.push_back(RAJA::RangeSegment(0, 10));
  is2.push_back(RAJA::RangeSegment(0, 5));
  is2.push_back(RAJA::RangeSegment(5, 10));
  ASSERT_TRUE(is1 != is2);
  ASSERT_FALSE(is1 == is2);
  ASSERT_NE(is1.size(), is2.size());
  ASSERT_EQ(is1.getLength(), is2.getLength());
}

TEST(IndexSet, swap)
{
  UnitIndexSet iset1;
  RAJA::RangeSegment range(0, 10);
  iset1.push_back(range);
  iset1.push_back_nocopy(&range);
  iset1.push_front(range);
  iset1.push_front_nocopy(&range);
  UnitIndexSet iset2;

  ASSERT_EQ(4l, iset1.size());
  ASSERT_EQ(40lu, iset1.getLength());
  ASSERT_EQ(0l, iset2.size());
  ASSERT_EQ(0lu, iset2.getLength());

  iset1.swap(iset2);

  ASSERT_EQ(4l, iset2.size());
  ASSERT_EQ(40lu, iset2.getLength());
  ASSERT_EQ(0l, iset1.size());
  ASSERT_EQ(0lu, iset1.getLength());
}
