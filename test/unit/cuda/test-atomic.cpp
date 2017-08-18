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
/// Source file containing tests for atomic operations
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

template<typename T, RAJA::Index_type N>
void testAtomicBasicAdd(){
  RAJA::RangeSegment seg(0, N);

  // initialize an array
  T *vec_double = nullptr;
  cudaMallocManaged((void **)&vec_double, sizeof(T) * N);

  T *sum_ptr = nullptr;
  cudaMallocManaged((void **)&sum_ptr, sizeof(T));

  cudaDeviceSynchronize();

  T expected = 0.0;
  T *expected_ptr = &expected;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (T)1;
    *expected_ptr += (T)1;
  });

  // use atomic add to reduce the array
  sum_ptr[0] = 0.0;

  RAJA::forall<RAJA::cuda_exec<256>>(seg,
    [=] RAJA_DEVICE (RAJA::Index_type i){
      RAJA::atomicAdd<RAJA::cuda_atomic>(sum_ptr, (T)1);
    }
  );

  cudaDeviceSynchronize();

  EXPECT_EQ(expected, sum_ptr[0]);

  cudaFree(vec_double);
  cudaFree(sum_ptr);
}



CUDA_TEST(Atomic, basic_add_cuda_int)
{
  testAtomicBasicAdd<int, 100000>();
}

CUDA_TEST(Atomic, basic_add_cuda_unsigned)
{
  testAtomicBasicAdd<unsigned, 100000>();
}

CUDA_TEST(Atomic, basic_add_cuda_float)
{
  testAtomicBasicAdd<float, 100000>();
}

CUDA_TEST(Atomic, basic_add_cuda_double)
{
  testAtomicBasicAdd<double, 100000>();
}

