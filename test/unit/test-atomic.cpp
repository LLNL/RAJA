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
/// Source file containing tests for atomic operations
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicFunctionBasic()
{
  RAJA::RangeSegment seg(0, N);

// initialize an array
#ifdef RAJA_ENABLE_CUDA
  T *dest = nullptr;
  cudaMallocManaged((void **)&dest, sizeof(T) * 8);

  cudaDeviceSynchronize();

#else
  T *dest = new T[8];
#endif


  // use atomic add to reduce the array
  dest[0] = (T)0;
  dest[1] = (T)N;
  dest[2] = (T)N;
  dest[3] = (T)0;
  dest[4] = (T)0;
  dest[5] = (T)0;
  dest[6] = (T)N + 1;
  dest[7] = (T)0;


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::atomic::atomicAdd<AtomicPolicy>(dest + 0, (T)1);
    RAJA::atomic::atomicSub<AtomicPolicy>(dest + 1, (T)1);

    RAJA::atomic::atomicMin<AtomicPolicy>(dest + 2, (T)i);
    RAJA::atomic::atomicMax<AtomicPolicy>(dest + 3, (T)i);
    RAJA::atomic::atomicInc<AtomicPolicy>(dest + 4);
    RAJA::atomic::atomicInc<AtomicPolicy>(dest + 5, (T)16);
    RAJA::atomic::atomicDec<AtomicPolicy>(dest + 6);
    RAJA::atomic::atomicDec<AtomicPolicy>(dest + 7, (T)16);
  });

#ifdef RAJA_ENABLE_CUDA
  cudaDeviceSynchronize();
#endif

  EXPECT_EQ((T)N, dest[0]);
  EXPECT_EQ((T)0, dest[1]);
  EXPECT_EQ((T)0, dest[2]);
  EXPECT_EQ((T)N - 1, dest[3]);
  EXPECT_EQ((T)N, dest[4]);
  EXPECT_EQ((T)4, dest[5]);
  EXPECT_EQ((T)1, dest[6]);
  EXPECT_EQ((T)13, dest[7]);


#ifdef RAJA_ENABLE_CUDA
  cudaFree(dest);
#else
  delete[] dest;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicFunctionPol()
{
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, int, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, unsigned, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, long long, 10000>();
  testAtomicFunctionBasic<ExecPolicy,
                          AtomicPolicy,
                          unsigned long long,
                          10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, float, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, double, 10000>();
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicRefBasic()
{
  RAJA::RangeSegment seg(0, N);

// initialize an array
#ifdef RAJA_ENABLE_CUDA
  T *dest = nullptr;
  cudaMallocManaged((void **)&dest, sizeof(T) * 6);

  cudaDeviceSynchronize();

#else
  T *dest = new T[6];
#endif


  // use atomic add to reduce the array
  dest[0] = (T)1;
  RAJA::atomic::AtomicRef<T, AtomicPolicy> sum0(dest);

  dest[1] = (T)1;
  RAJA::atomic::AtomicRef<T, AtomicPolicy> sum1(dest + 1);

  dest[2] = (T)1;
  RAJA::atomic::AtomicRef<T, AtomicPolicy> sum2(dest + 2);

  dest[3] = (T)(N + 1);
  RAJA::atomic::AtomicRef<T, AtomicPolicy> sum3(dest + 3);

  dest[4] = (T)(N + 1);
  RAJA::atomic::AtomicRef<T, AtomicPolicy> sum4(dest + 4);

  dest[5] = (T)(N + 1);
  RAJA::atomic::AtomicRef<T, AtomicPolicy> sum5(dest + 5);


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type) {
    sum0++;

    ++sum1;

    sum2 += 1;

    sum3--;

    --sum4;

    sum5 -= 1;
  });

#ifdef RAJA_ENABLE_CUDA
  cudaDeviceSynchronize();
#endif

  EXPECT_EQ((T)N + 1, dest[0]);
  EXPECT_EQ((T)N + 1, dest[1]);
  EXPECT_EQ((T)N + 1, dest[2]);
  EXPECT_EQ((T)1, dest[3]);
  EXPECT_EQ((T)1, dest[4]);
  EXPECT_EQ((T)1, dest[5]);

#ifdef RAJA_ENABLE_CUDA
  cudaFree(dest);
#else
  delete[] dest;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicRefPol()
{
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, int, 10000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, unsigned, 10000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, long long, 10000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, unsigned long long, 10000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, float, 10000>();
  testAtomicRefBasic<ExecPolicy, AtomicPolicy, double, 10000>();
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicViewBasic()
{
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N / 2);

// initialize an array
#ifdef RAJA_ENABLE_CUDA
  T *source = nullptr;
  cudaMallocManaged((void **)&source, sizeof(T) * N);

  T *dest = nullptr;
  cudaMallocManaged((void **)&dest, sizeof(T) * N / 2);

  cudaDeviceSynchronize();
#else
  T *source = new T[N];
  T *dest = new T[N / 2];
#endif

  RAJA::forall<RAJA::seq_exec>(seg,
                               [=](RAJA::Index_type i) { source[i] = (T)1; });

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(source, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);


  // Zero out dest using atomic view
  RAJA::forall<ExecPolicy>(seg_half, [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
    sum_atomic_view(i) = (T)0;
  });

  // Assign values to dest using atomic view
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i / 2) += vec_view(i);
  });

#ifdef RAJA_ENABLE_CUDA
  cudaDeviceSynchronize();
#endif

  for (RAJA::Index_type i = 0; i < N / 2; ++i) {
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


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicViewPol()
{
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, float, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, double, 100000>();
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicLogical()
{
  RAJA::RangeSegment seg(0, N * 8);
  RAJA::RangeSegment seg_bytes(0, N);

// initialize an array
#ifdef RAJA_ENABLE_CUDA
  T *dest_and = nullptr;
  cudaMallocManaged((void **)&dest_and, sizeof(T) * N);

  T *dest_or = nullptr;
  cudaMallocManaged((void **)&dest_or, sizeof(T) * N);

  T *dest_xor = nullptr;
  cudaMallocManaged((void **)&dest_xor, sizeof(T) * N);

  cudaDeviceSynchronize();
#else
  T *dest_and = new T[N];
  T *dest_or = new T[N];
  T *dest_xor = new T[N];
#endif

  RAJA::forall<RAJA::seq_exec>(seg_bytes, [=](RAJA::Index_type i) {
    dest_and[i] = (T)0;
    dest_or[i] = (T)0;
    dest_xor[i] = (T)0;
  });


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::Index_type offset = i / 8;
    RAJA::Index_type bit = i % 8;
    RAJA::atomic::atomicAnd<AtomicPolicy>(dest_and + offset,
                                          (T)(0xFF ^ (1 << bit)));
    RAJA::atomic::atomicOr<AtomicPolicy>(dest_or + offset, (T)(1 << bit));
    RAJA::atomic::atomicXor<AtomicPolicy>(dest_xor + offset, (T)(1 << bit));
  });

#ifdef RAJA_ENABLE_CUDA
  cudaDeviceSynchronize();
#endif

  for (RAJA::Index_type i = 0; i < N; ++i) {
    EXPECT_EQ((T)0x00, dest_and[i]);
    EXPECT_EQ((T)0xFF, dest_or[i]);
    EXPECT_EQ((T)0xFF, dest_xor[i]);
  }

#ifdef RAJA_ENABLE_CUDA
  cudaFree(dest_and);
  cudaFree(dest_or);
  cudaFree(dest_xor);
#else
  delete[] dest_and;
  delete[] dest_or;
  delete[] dest_xor;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicLogicalPol()
{
  testAtomicLogical<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
}


#ifdef RAJA_ENABLE_OPENMP

TEST(Atomic, basic_OpenMP_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_AtomicRef)
{
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_AtomicView)
{
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_Logical)
{
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}

#endif

#ifdef RAJA_ENABLE_CUDA

CUDA_TEST(Atomic, basic_CUDA_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}

CUDA_TEST(Atomic, basic_CUDA_AtomicRef)
{
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}

CUDA_TEST(Atomic, basic_CUDA_AtomicView)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}


CUDA_TEST(Atomic, basic_CUDA_Logical)
{
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}


#endif

TEST(Atomic, basic_seq_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}

TEST(Atomic, basic_seq_AtomicRef)
{
  testAtomicRefPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicRefPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicRefPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}

TEST(Atomic, basic_seq_AtomicView)
{
  testAtomicViewPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicViewPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicViewPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_seq_Logical)
{
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}
