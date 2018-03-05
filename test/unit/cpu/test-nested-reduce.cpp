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
/// Source file containing test for nested reductions...
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

template <typename Outer, typename Inner, typename Reduce>
void test(int inner, int outer)
{
  using limits = RAJA::operators::limits<double>;
  RAJA::ReduceSum<Reduce, double> sum(0.0);
  RAJA::ReduceMin<Reduce, double> min(limits::max());
  RAJA::ReduceMax<Reduce, double> max(limits::min());

  RAJA::forall<Outer>(RAJA::make_range(0, outer), [=](int y) {
    RAJA::forall<Inner>(RAJA::make_range(0, inner), [=](int x) {
      double val = y * inner + x + 1;
      sum += val;
      min.min(val);
      max.max(val);
    });
  });

  double area = inner * outer;
  ASSERT_EQ((area * (area + 1)) / 2, sum.get());
  ASSERT_EQ(1.0, min.get());
  ASSERT_EQ(area, max.get());
}

TEST(NestedReduce, seq_seq)
{
  test<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_reduce>(10, 20);
  test<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_reduce>(37, 73);
}

#if defined(RAJA_ENABLE_OPENMP)
TEST(NestedReduce, omp_seq)
{
  test<RAJA::omp_parallel_for_exec, RAJA::seq_exec, RAJA::omp_reduce>(10, 20);
  test<RAJA::omp_parallel_for_exec, RAJA::seq_exec, RAJA::omp_reduce>(37, 73);
}

TEST(NestedReduce, seq_omp)
{
  test<RAJA::seq_exec, RAJA::omp_parallel_for_exec, RAJA::omp_reduce>(10, 20);
  test<RAJA::seq_exec, RAJA::omp_parallel_for_exec, RAJA::omp_reduce>(37, 73);
}
#endif
#if defined(RAJA_ENABLE_TBB)
TEST(NestedReduce, tbb_seq)
{
  test<RAJA::tbb_for_exec, RAJA::seq_exec, RAJA::tbb_reduce>(10, 20);
  test<RAJA::tbb_for_exec, RAJA::seq_exec, RAJA::tbb_reduce>(37, 73);
}

TEST(NestedReduce, seq_tbb)
{
  test<RAJA::seq_exec, RAJA::tbb_for_exec, RAJA::tbb_reduce>(10, 20);
  test<RAJA::seq_exec, RAJA::tbb_for_exec, RAJA::tbb_reduce>(37, 73);
}
#endif
