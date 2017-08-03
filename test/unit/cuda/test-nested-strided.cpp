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

static const int x = 500, y = 300, z = 70;

using namespace RAJA;

static void stride_test(int stride, bool reverse = false)
{
  int *arr = nullptr;
  cudaErrchk(cudaMallocManaged(&arr, sizeof(*arr) * x * y * z));
  cudaMemset(arr, 0, sizeof(*arr) * x * y * z);

  RangeStrideSegment seg_x( reverse ? x-1     : 0,
                            reverse ? -1      : x,
                            reverse ? -stride : stride);
                            
  RangeStrideSegment seg_y( reverse ? y-1     : 0,
                            reverse ? -1      : y,
                            reverse ? -stride : stride);
                            
  RangeStrideSegment seg_z( reverse ? z-1     : 0,
                            reverse ? -1      : z,
                            reverse ? -stride : stride);                            

  forallN<NestedPolicy<ExecList<seq_exec,
                                cuda_block_x_exec,
                                cuda_thread_y_exec>,
                       Permute<PERM_IJK>>>(seg_x, seg_y, seg_z,
                                           [=] RAJA_DEVICE(Index_type i,
                                                           Index_type j,
                                                           Index_type k) {
                                             Index_type val = (i*y*z) + (j*z) + k;
                                             arr[val] = val;
                                           });
  cudaDeviceSynchronize();
  
  
  for (Index_type i : RangeSegment(0,x)) {
    for (Index_type j : RangeSegment(0,y)) {
      for (Index_type k : RangeSegment(0,z)) {
      
        Index_type val = (i*y*z) + (j*z) + k;
        
        // Determine if this i,j,k was in the iteration space
        bool inclusive;
        if(reverse){
          inclusive = ((x-i-1)%stride==0) && ((y-j-1)%stride==0) && ((z-k-1)%stride==0);
        }
        else{
          inclusive = (i%stride==0) && (j%stride==0) && (k%stride==0);
        }
        
        // Determine expected value
        int expected_value = inclusive ? val : 0;        
        
        ASSERT_EQ(expected_value, arr[val]);
      }
    }
  }
  cudaFree(arr);
}


TEST(forallN, rangeStrides1)
{
  stride_test(1, false);
}

TEST(forallN, rangeStrides2)
{
  stride_test(2, false);
}

TEST(forallN, rangeStrides3)
{
  stride_test(3, false);
}

TEST(forallN, rangeStrides4)
{
  stride_test(4, false);
}


TEST(forallN, rangeStrides1_reverse)
{
  stride_test(1, true);
}

TEST(forallN, rangeStrides2_reverse)
{
  stride_test(2, true);
}

TEST(forallN, rangeStrides3_reverse)
{
  stride_test(3, true);
}

TEST(forallN, rangeStrides4_reverse)
{
  stride_test(4, true);
}
