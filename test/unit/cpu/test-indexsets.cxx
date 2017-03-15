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
  }

  RAJA::BasicIndexSet<RAJA::RangeSegment, RAJA::ListSegment, RAJA::RangeStrideSegment> index_sets_[NumBuildMethods];
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

  // FIX ME push back will work here because RangeStrideSegment is a valid type in the BasicIndexSet - need to come up with a new type to test with
  //EXPECT_NE(true, index_sets_[0].isValidSegmentType(rs_segment));
  //EXPECT_NE(true, index_sets_[0].push_back(rs_segment));
  //EXPECT_NE(true, index_sets_[0].push_back_nocopy(&rs_segment));
}
