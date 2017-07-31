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

#include "gtest/gtest.h"
#include "RAJA/RAJA.hpp"

RAJA::Index_type const begin = 0;
RAJA::Index_type const xExtent = 64;
RAJA::Index_type const yExtent = 64;
RAJA::Index_type const area = xExtent * yExtent;

TEST(NestedReduce,outer)
{
  RAJA::ReduceSum<RAJA::omp_target_reduce<64>, double> sumA(0.0);
  RAJA::ReduceMin<RAJA::omp_target_reduce<64>, double> minA(10000.0);
  RAJA::ReduceMax<RAJA::omp_target_reduce<64>, double> maxA(0.0);

  RAJA::forall<RAJA::omp_target_parallel_for_exec<64>>(begin, yExtent, [=](int y) {
    RAJA::forall<RAJA::seq_exec>(begin, xExtent, [=](int x) {
      sumA += double(y * xExtent + x + 1);
      minA.min(double(y * xExtent + x + 1));
      maxA.max(double(y * xExtent + x + 1));
    });
  });

  ASSERT_FLOAT_EQ((area * (area + 1) / 2.0), sumA.get());
  ASSERT_FLOAT_EQ(1.0, minA.get());
  ASSERT_FLOAT_EQ(area, maxA.get());
}

TEST(NestedReduce,inner)
{
  RAJA::ReduceSum<RAJA::omp_target_reduce<64>, double> sumB(0.0);
  RAJA::ReduceMin<RAJA::omp_target_reduce<64>, double> minB(10000.0);
  RAJA::ReduceMax<RAJA::omp_target_reduce<64>, double> maxB(0.0);

  RAJA::forall<RAJA::seq_exec>(begin, yExtent, [=](int y) {
    RAJA::forall<RAJA::omp_target_parallel_for_exec<64>>(begin, xExtent, [=](int x) {
      sumB += double(y * xExtent + x + 1);
      minB.min(double(y * xExtent + x + 1));
      maxB.max(double(y * xExtent + x + 1));
    });
  });

  ASSERT_FLOAT_EQ((area * (area + 1) / 2.0), sumA.get());
  ASSERT_FLOAT_EQ(1.0, minA.get());
  ASSERT_FLOAT_EQ(area, maxA.get());
}
