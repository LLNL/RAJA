//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/README.
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
#endif  // !defined(RAJA_COMPILER_XLC12) && 1
