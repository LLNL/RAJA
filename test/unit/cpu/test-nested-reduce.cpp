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
  test<RAJA::tbb_exec, RAJA::seq_exec, RAJA::tbb_reduce>(10, 20);
  test<RAJA::tbb_exec, RAJA::seq_exec, RAJA::tbb_reduce>(37, 73);
}

TEST(NestedReduce, seq_tbb)
{
  test<RAJA::seq_exec, RAJA::tbb_exec, RAJA::tbb_reduce>(10, 20);
  test<RAJA::seq_exec, RAJA::tbb_exec, RAJA::tbb_reduce>(37, 73);
}
#endif
