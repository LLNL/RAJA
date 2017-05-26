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

#include <stdio.h>
#include "RAJA/RAJA.hpp"

#include "RAJA/util/defines.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  int run = 0;
  int passed = 0;

  RAJA::Index_type const begin = 0;

  RAJA::Index_type const xExtent = 10;
  RAJA::Index_type const yExtent = 10;
  RAJA::Index_type const area = xExtent * yExtent;

  RAJA::ReduceSum<RAJA::omp_reduce, double> sumA(0.0);
  RAJA::ReduceMin<RAJA::omp_reduce, double> minA(10000.0);
  RAJA::ReduceMax<RAJA::omp_reduce, double> maxA(0.0);

  RAJA::forall<RAJA::omp_parallel_for_exec>(begin, yExtent, [=](int y) {
    RAJA::forall<RAJA::seq_exec>(begin, xExtent, [=](int x) {
      sumA += double(y * xExtent + x + 1);
      minA.min(double(y * xExtent + x + 1));
      maxA.max(double(y * xExtent + x + 1));
    });
  });

  printf("sum(%6.1f) = %6.1f, min(1.0) = %3.1f, max(%5.1f) = %5.1f\n",
         (area * (area + 1) / 2.0),
         double(sumA),
         double(minA),
         double(area),
         double(maxA));

  run += 3;
  if (double(sumA) == (area * (area + 1) / 2.0)) {
    ++passed;
  }
  if (double(minA) == 1.0) {
    ++passed;
  }
  if (double(maxA) == double(area)) {
    ++passed;
  }

  RAJA::ReduceSum<RAJA::omp_reduce, double> sumB(0.0);
  RAJA::ReduceMin<RAJA::omp_reduce, double> minB(10000.0);
  RAJA::ReduceMax<RAJA::omp_reduce, double> maxB(0.0);

  RAJA::forall<RAJA::seq_exec>(begin, yExtent, [=](int y) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(begin, xExtent, [=](int x) {
      sumB += double(y * xExtent + x + 1);
      minB.min(double(y * xExtent + x + 1));
      maxB.max(double(y * xExtent + x + 1));
    });
  });

  printf("sum(%6.1f) = %6.1f, min(1.0) = %3.1f, max(%5.1f) = %5.1f\n",
         (area * (area + 1) / 2.0),
         double(sumB),
         double(minB),
         double(area),
         double(maxB));

  run += 3;
  if (double(sumB) == (area * (area + 1) / 2.0)) {
    ++passed;
  }
  if (double(minB) == 1.0) {
    ++passed;
  }
  if (double(maxB) == double(area)) {
    ++passed;
  }

  printf("\n All Tests : # passed / # run = %d / %d\n\n DONE!!!\n",
         passed,
         run);

  if (passed == run) {
    return 0;
  } else {
    return 1;
  }
}
