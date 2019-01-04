//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
/// Source file containing test for nested reductions...
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

static RAJA::Index_type const begin = 0;
static RAJA::Index_type const xExtent = 64;
static RAJA::Index_type const yExtent = 64;
static RAJA::Index_type const area = xExtent * yExtent;

TEST(NestedReduceTargetOMP, outer)
{
  RAJA::Index_type l_begin = begin;
  RAJA::Index_type l_xExtent = xExtent;
  RAJA::ReduceSum<RAJA::omp_target_reduce<64>, double> sumA(0.0);
  RAJA::ReduceMin<RAJA::omp_target_reduce<64>, double> minA(10000.0);
  RAJA::ReduceMax<RAJA::omp_target_reduce<64>, double> maxA(0.0);
  RAJA::RangeSegment xrange(begin, xExtent);
  RAJA::RangeSegment yrange(begin, yExtent);

  RAJA::forall<RAJA::omp_target_parallel_for_exec<64>>(yrange, [=](int y) {
    RAJA::forall<RAJA::seq_exec>(xrange, [=](int x) {
      sumA += double(y * l_xExtent + x + 1);
      minA.min(double(y * l_xExtent + x + 1));
      maxA.max(double(y * l_xExtent + x + 1));
    });
  });

  ASSERT_FLOAT_EQ((area * (area + 1) / 2.0), sumA.get());
  ASSERT_FLOAT_EQ(1.0, minA.get());
  ASSERT_FLOAT_EQ(area, maxA.get());
}

TEST(NestedReduceTargetOMP, inner)
{
  RAJA::ReduceSum<RAJA::omp_target_reduce<64>, double> sumB(0.0);
  RAJA::ReduceMin<RAJA::omp_target_reduce<64>, double> minB(10000.0);
  RAJA::ReduceMax<RAJA::omp_target_reduce<64>, double> maxB(0.0);
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
