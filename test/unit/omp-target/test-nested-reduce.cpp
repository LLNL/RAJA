//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing test for nested reductions...
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

static RAJA::Index_type const begin = 0;
static RAJA::Index_type const xExtent = 64;
static RAJA::Index_type const yExtent = 64;
static RAJA::Index_type const area = xExtent * yExtent;

// TEST(NestedReduceTargetOMP, outer)
// {
//   // This can't work
//   RAJA::Index_type l_begin = begin;
//   RAJA::Index_type l_xExtent = xExtent;
//   RAJA::ReduceSum<RAJA::omp_target_reduce, double> sumA(0.0);
//   RAJA::ReduceMin<RAJA::omp_target_reduce, double> minA(10000.0);
//   RAJA::ReduceMax<RAJA::omp_target_reduce, double> maxA(0.0);
//   RAJA::RangeSegment xrange(begin, xExtent);
//   RAJA::RangeSegment yrange(begin, yExtent);
//
//   RAJA::forall<RAJA::omp_target_parallel_for_exec<64>>(yrange, [=](int y) {
//     RAJA::forall<RAJA::seq_exec>(xrange, [=](int x) {
//       sumA += double(y * l_xExtent + x + 1);
//       minA.min(double(y * l_xExtent + x + 1));
//       maxA.max(double(y * l_xExtent + x + 1));
//     });
//   });
//
//   ASSERT_FLOAT_EQ((area * (area + 1) / 2.0), sumA.get());
//   ASSERT_FLOAT_EQ(1.0, minA.get());
//   ASSERT_FLOAT_EQ(area, maxA.get());
// }

TEST(NestedReduceTargetOMP, inner)
{
  RAJA::ReduceSum<RAJA::omp_target_reduce, double> sumB(0.0);
  RAJA::ReduceMin<RAJA::omp_target_reduce, double> minB(10000.0);
  RAJA::ReduceMax<RAJA::omp_target_reduce, double> maxB(0.0);
  RAJA::RangeSegment xrange(begin, xExtent);
  RAJA::RangeSegment yrange(begin, yExtent);

  RAJA::forall<RAJA::seq_exec>(yrange, [=](int y) {
    RAJA::forall<RAJA::omp_target_parallel_for_exec<64>>(xrange, [=](int x) {
      sumB += double(y * xExtent + x + 1);
      minB.min(double(y * xExtent + x + 1));
      maxB.max(double(y * xExtent + x + 1));
    });
  });

  ASSERT_FLOAT_EQ((area * (area + 1) / 2.0), sumB.get());
  ASSERT_FLOAT_EQ(1.0, minB.get());
  ASSERT_FLOAT_EQ(area, maxB.get());
}
