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
/// Source file containing tests for RAJA forallN strided tests.
///

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include <cstdlib>

#include <gtest/gtest.h>
#include <RAJA/RAJA.hpp>

const int x = 500, y = 500, z = 50;

using namespace RAJA;
void stride_test(int stride)
{
  double *arr = NULL;
  cudaErrchk(cudaMallocManaged(&arr, sizeof(*arr) * x * y * z));
  cudaMemset(arr, 0, sizeof(*arr) * x * y * z);

  forallN<NestedPolicy<ExecList<seq_exec,
                                cuda_block_x_exec,
                                cuda_thread_y_exec>,
                       Permute<PERM_IJK>>>(RangeStrideSegment(0, z, stride),
                                           RangeStrideSegment(0, y, stride),
                                           RangeStrideSegment(0, x, stride),
                                           [=] RAJA_DEVICE(int i,
                                                           int j,
                                                           int k) {
                                             int val = z * y * i + y * j + k;
                                             arr[val] = val;
                                           });
  cudaDeviceSynchronize();

  int prev_val = 0;
  for (int i = 0; i < z; i += stride) {
    for (int j = 0; j < y; j += stride) {
      for (int k = 0; k < x; k += stride) {
        int val = z * y * i + y * j + k;
        ASSERT_EQ(arr[val], val);
        for (int inner = prev_val + 1; inner < val; ++inner) {
          ASSERT_EQ(arr[inner], 0);
        }
        prev_val = val;
      }
    }
  }
  cudaFree(arr);
}

TEST(forallN, rangeStrides)
{

  stride_test(1);
  stride_test(2);
  stride_test(3);
  stride_test(4);
}
