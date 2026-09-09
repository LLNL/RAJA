//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for RAJA::expt::ValLoc.
///

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

RAJA_INDEX_VALUE(ValLocStrongIndex, "ValLocStrongIndex")

TEST(ValLocUnitTest, DefaultLocIsUnsetForPlainIndex)
{
  RAJA::expt::ValLoc<double> vl(1.0);

  ASSERT_EQ(vl.getVal(), 1.0);
  ASSERT_EQ(vl.getLoc(), RAJA::Index_type(-1));
}

TEST(ValLocUnitTest, DefaultLocIsUnsetForStrongIndex)
{
  RAJA::expt::ValLoc<double, ValLocStrongIndex> vl(1.0);

  ASSERT_EQ(vl.getVal(), 1.0);
  ASSERT_EQ(*vl.getLoc(), -1);
}

TEST(ValLocUnitTest, StrongIndexCanBeSetAndRead)
{
  RAJA::expt::ValLoc<double, ValLocStrongIndex> vl(2.5, ValLocStrongIndex(3));

  ASSERT_EQ(vl.getVal(), 2.5);
  ASSERT_EQ(*vl.getLoc(), 3);

  vl.set(4.5, ValLocStrongIndex(7));

  ASSERT_EQ(vl.getVal(), 4.5);
  ASSERT_EQ(*vl.getLoc(), 7);
}

TEST(ValLocUnitTest, StrongIndexWorksInAForallReduction)
{
  constexpr int N = 8;
  double a[N] = {5.0, 3.0, 9.0, 1.0, 7.0, 2.0, 8.0, 4.0};

  using VL = RAJA::expt::ValLoc<double, ValLocStrongIndex>;
  using VLOp =
      RAJA::expt::ValLocOp<double, ValLocStrongIndex, RAJA::operators::minimum>;

  VL result(RAJA::operators::limits<double>::max(), ValLocStrongIndex(0));

  RAJA::forall<RAJA::seq_exec>(
      RAJA::TypedRangeSegment<ValLocStrongIndex>(0, N),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&result),
      [=](ValLocStrongIndex i, VLOp& r) { r.minloc(a[*i], i); });

  ASSERT_EQ(result.getVal(), 1.0);
  ASSERT_EQ(*result.getLoc(), 3);
}
