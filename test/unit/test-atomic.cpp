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



template<typename ExecPolicy, typename AtomicPolicy, typename T, RAJA::Index_type N>
void testAtomicFunctionBasic(){
  RAJA::RangeSegment seg(0, N);

  // initialize an array
  #ifdef RAJA_ENABLE_CUDA
      T *dest = nullptr;
      cudaMallocManaged((void **)&dest, sizeof(T) * 2);

      cudaDeviceSynchronize();

  #else
    T *dest = new T[4];
  #endif


  // use atomic add to reduce the array
  dest[0] = (T)0;
  dest[1] = (T)N;

  RAJA::forall<ExecPolicy>(seg,
    [=] RAJA_HOST_DEVICE (RAJA::Index_type){
      RAJA::atomicAdd<AtomicPolicy>(dest+0, (T)1);
      RAJA::atomicSub<AtomicPolicy>(dest+1, (T)1);
    }
  );

#ifdef RAJA_ENABLE_CUDA
  cudaDeviceSynchronize();
#endif

  EXPECT_EQ((T)N, dest[0]);
  EXPECT_EQ((T)0, dest[1]);

#ifdef RAJA_ENABLE_CUDA
  cudaFree(dest);
#else
  delete[] dest;
#endif
}


template<typename ExecPolicy, typename AtomicPolicy>
void testAtomicFunctionPol(){
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, float, 100000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, double, 100000>();
}



template<typename ExecPolicy, typename AtomicPolicy, typename T, RAJA::Index_type N>
void testAtomicRefBasic(){
  RAJA::RangeSegment seg(0, N);

  // initialize an array
  #ifdef RAJA_ENABLE_CUDA
      T *dest = nullptr;
      cudaMallocManaged((void **)&dest, sizeof(T) * 4);

      cudaDeviceSynchronize();

  #else
    T *dest = new T[4];
  #endif


  // use atomic add to reduce the array
  dest[0] = 0.0;
  RAJA::AtomicRef<T, AtomicPolicy> sum0(dest);

  dest[1] = 0.0;
  RAJA::AtomicRef<T, AtomicPolicy> sum1(dest+1);

  dest[2] = N;
  RAJA::AtomicRef<T, AtomicPolicy> sum2(dest+2);

  dest[3] = N;
  RAJA::AtomicRef<T, AtomicPolicy> sum3(dest+3);
  RAJA::forall<ExecPolicy>(seg,
    [=] RAJA_HOST_DEVICE (RAJA::Index_type){
      sum0 ++;

      ++ sum1;

      sum2 --;

      -- sum3;
    }
  );

#ifdef RAJA_ENABLE_CUDA
  cudaDeviceSynchronize();
#endif

  EXPECT_EQ((T)N, dest[0]);
  EXPECT_EQ((T)N, dest[1]);
  EXPECT_EQ((T)0, dest[2]);
  EXPECT_EQ((T)0, dest[3]);

#ifdef RAJA_ENABLE_CUDA
  cudaFree(dest);
#else
  delete[] dest;
#endif
}


template<typename ExecPolicy, typename AtomicPolicy>
void testAtomicRefPol(){
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, float, 100000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, double, 100000>();
}


template<typename ExecPolicy, typename AtomicPolicy, typename T, RAJA::Index_type N>
void testAtomicViewBasic(){
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N/2);

  // initialize an array
#ifdef RAJA_ENABLE_CUDA
  T *source = nullptr;
    cudaMallocManaged((void **)&source, sizeof(T) * N);

    T *dest = nullptr;
    cudaMallocManaged((void **)&dest, sizeof(T) * N/2);

    cudaDeviceSynchronize();
#else
  T *source = new T[N];
  T *dest = new T[N/2];
#endif

  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    source[i] = (T)1;
  });

  RAJA::forall<RAJA::seq_exec>(seg_half, [=](RAJA::Index_type i){
    dest[i] = (T)0;
  });

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(source, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);

  RAJA::forall<ExecPolicy>(seg,
    [=] RAJA_HOST_DEVICE (RAJA::Index_type i){
      sum_atomic_view(i/2) += vec_view(i);
    }
  );

#ifdef RAJA_ENABLE_CUDA
  cudaDeviceSynchronize();
#endif

  for(RAJA::Index_type i = 0;i < N/2;++ i){
    EXPECT_EQ((T)2, dest[i]);
  }

#ifdef RAJA_ENABLE_CUDA
  cudaFree(source);
  cudaFree(dest);
#else
  delete[] source;
  delete[] dest;
#endif
}




template<typename ExecPolicy, typename AtomicPolicy>
void testAtomicViewPol(){
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, float, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, double, 100000>();
}

#ifdef RAJA_ENABLE_OPENMP

TEST(Atomic, basic_OpenMP_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
}


TEST(Atomic, basic_OpenMP_AtomicRef)
{
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
}


TEST(Atomic, basic_OpenMP_AtomicView)
{
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
}
#endif

#ifdef RAJA_ENABLE_CUDA

CUDA_TEST(Atomic, basic_CUDA_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}

CUDA_TEST(Atomic, basic_CUDA_AtomicRef)
{
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}

CUDA_TEST(Atomic, basic_CUDA_AtomicView)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}
#endif

TEST(Atomic, basic_seq_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::auto_atomic>();
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::seq_atomic>();
}

TEST(Atomic, basic_seq_AtomicRef)
{
  testAtomicRefPol<RAJA::seq_exec, RAJA::auto_atomic>();
  testAtomicRefPol<RAJA::seq_exec, RAJA::seq_atomic>();
}

TEST(Atomic, basic_seq_AtomicView)
{
  testAtomicViewPol<RAJA::seq_exec, RAJA::auto_atomic>();
  testAtomicViewPol<RAJA::seq_exec, RAJA::seq_atomic>();
}
