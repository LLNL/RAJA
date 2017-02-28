#include "gtest/gtest.h"

#include "buildIndexSet.hxx"

#include "RAJA/RAJA.hxx"

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
  RAJA::IndexSet index_sets_[NumBuildMethods];
};

TEST_F(IndexSetTest, IndexSetEquality)
{
  for (unsigned ibuild = 1; ibuild < NumBuildMethods; ++ibuild) {
    EXPECT_EQ(index_sets_[ibuild], index_sets_[1]);
  }
}

// TODO: tests for adding other invalid types
TEST_F(IndexSetTest, InvalidSegments)
{
  RAJA::RangeStrideSegment rs_segment(0, 4, 2);

  EXPECT_NE(true, index_sets_[0].isValidSegmentType(&rs_segment));
  EXPECT_NE(true, index_sets_[0].push_back(rs_segment));
  EXPECT_NE(true, index_sets_[0].push_back_nocopy(&rs_segment));
}

#if !defined(RAJA_COMPILER_XLC12) && 1
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
#endif // !defined(RAJA_COMPILER_XLC12) && 1
