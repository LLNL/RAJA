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
  cudaMallocManaged((void **)&sum_ptr, sizeof(T)*2);

  cudaDeviceSynchronize();

  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (T)1;
  });

  // use atomic add to reduce the array
  sum_ptr[0] = 0.0;
  sum_ptr[1] = N;

  RAJA::forall<RAJA::cuda_exec<256>>(seg,
    [=] RAJA_DEVICE (RAJA::Index_type i){
      RAJA::atomicAdd<RAJA::cuda_atomic>(sum_ptr, (T)1);
      RAJA::atomicSub<RAJA::cuda_atomic>(sum_ptr+1, (T)1);
    }
  );

  cudaDeviceSynchronize();

  EXPECT_EQ((T)N, sum_ptr[0]);
  EXPECT_EQ((T)0, sum_ptr[1]);

  cudaFree(vec_double);
  cudaFree(sum_ptr);
}




template<typename T, RAJA::Index_type N>
void testAtomicRefBasic(){
  RAJA::RangeSegment seg(0, N);

  // initialize an array
  T *vec_double = nullptr;
  cudaMallocManaged((void **)&vec_double, sizeof(T) * N);

  T *sum_ptr = nullptr;
  cudaMallocManaged((void **)&sum_ptr, sizeof(T)*4);

  cudaDeviceSynchronize();

  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (T)1;
  });

  // use atomic add to reduce the array
  sum_ptr[0] = 0.0;
  RAJA::AtomicRef<T, RAJA::cuda_atomic> sum0(sum_ptr);

  sum_ptr[1] = 0.0;
  RAJA::AtomicRef<T, RAJA::cuda_atomic> sum1(sum_ptr+1);

  sum_ptr[2] = N;
  RAJA::AtomicRef<T, RAJA::cuda_atomic> sum2(sum_ptr+2);

  sum_ptr[3] = N;
  RAJA::AtomicRef<T, RAJA::cuda_atomic> sum3(sum_ptr+3);
  RAJA::forall<RAJA::cuda_exec<256>>(seg,
    [=] RAJA_DEVICE (RAJA::Index_type i){
      sum0 ++;

      ++ sum1;

      sum2 --;

      -- sum3;
    }
  );

  cudaDeviceSynchronize();

  EXPECT_EQ((T)N, sum_ptr[0]);
  EXPECT_EQ((T)N, sum_ptr[1]);
  EXPECT_EQ((T)0, sum_ptr[2]);
  EXPECT_EQ((T)0, sum_ptr[3]);

  cudaFree(vec_double);
  cudaFree(sum_ptr);
}




template<typename T, RAJA::Index_type N>
void testAtomicViewBasic(){
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N/2);

  // initialize an array
  T *vec_double = nullptr;
  cudaMallocManaged((void **)&vec_double, sizeof(T) * N);

  T *sum_ptr = nullptr;
  cudaMallocManaged((void **)&sum_ptr, sizeof(T) * N);

  cudaDeviceSynchronize();

  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (T)1;
  });

  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    sum_ptr[i] = (T)0;
  });

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(vec_double, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(sum_ptr, N);
  auto sum_atomic_view = RAJA::make_atomic_view(sum_view);

  RAJA::forall<RAJA::cuda_exec<256>>(seg,
    [=] RAJA_DEVICE (RAJA::Index_type i){
      sum_atomic_view(i/2) += vec_view(i);
    }
  );

  cudaDeviceSynchronize();

  for(RAJA::Index_type i = 0;i < N/2;++ i){
    EXPECT_EQ((T)2, sum_ptr[i]);
  }


  cudaFree(vec_double);
  cudaFree(sum_ptr);
}



CUDA_TEST(Atomic, basic_fcn_cuda_int)
{
  testAtomicBasicAdd<int, 100000>();
}

CUDA_TEST(Atomic, basic_fcn_cuda_unsigned)
{
  testAtomicBasicAdd<unsigned, 100000>();
}

CUDA_TEST(Atomic, basic_fcn_cuda_float)
{
  testAtomicBasicAdd<float, 100000>();
}

CUDA_TEST(Atomic, basic_fcn_cuda_double)
{
  testAtomicBasicAdd<double, 100000>();
}





CUDA_TEST(Atomic, basic_atomicref_int)
{
  testAtomicRefBasic<int, 100000>();
}

CUDA_TEST(Atomic, basic_atomicref_unsigned)
{
  testAtomicRefBasic<unsigned, 100000>();
}

CUDA_TEST(Atomic, basic_atomicref_float)
{
  testAtomicRefBasic<float, 100000>();
}

CUDA_TEST(Atomic, basic_atomicref_double)
{
  testAtomicRefBasic<double, 100000>();
}




CUDA_TEST(Atomic, basic_AtomicView_int)
{
  testAtomicViewBasic<int, 100000>();
}

CUDA_TEST(Atomic, basic_AtomicView_unsigned)
{
  testAtomicViewBasic<unsigned, 100000>();
}

CUDA_TEST(Atomic, basic_AtomicView_float)
{
  testAtomicViewBasic<float, 100000>();
}

CUDA_TEST(Atomic, basic_AtomicView_double)
{
  testAtomicViewBasic<int, 100000>();
}
