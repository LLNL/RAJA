//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall and views.
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
#if defined(RAJA_ENABLE_CUDA)
  T *dest = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest, sizeof(T) * 8));

  cudaErrchk(cudaDeviceSynchronize());

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
    RAJA::atomicAdd<AtomicPolicy>(dest + 0, (T)1);
    RAJA::atomicSub<AtomicPolicy>(dest + 1, (T)1);

    RAJA::atomicMin<AtomicPolicy>(dest + 2, (T)i);
    RAJA::atomicMax<AtomicPolicy>(dest + 3, (T)i);
    RAJA::atomicInc<AtomicPolicy>(dest + 4);
    RAJA::atomicInc<AtomicPolicy>(dest + 5, (T)16);
    RAJA::atomicDec<AtomicPolicy>(dest + 6);
    RAJA::atomicDec<AtomicPolicy>(dest + 7, (T)16);
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

  EXPECT_EQ((T)N, dest[0]);
  EXPECT_EQ((T)0, dest[1]);
  EXPECT_EQ((T)0, dest[2]);
  EXPECT_EQ((T)N - 1, dest[3]);
  EXPECT_EQ((T)N, dest[4]);
  EXPECT_EQ((T)4, dest[5]);
  EXPECT_EQ((T)1, dest[6]);
  EXPECT_EQ((T)13, dest[7]);


#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(dest));
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
void testAtomicViewBasic()
{
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N / 2);

// initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *source = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&source, sizeof(T) * N));

  T *dest = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest, sizeof(T) * N / 2));

  cudaErrchk(cudaDeviceSynchronize());
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
  RAJA::forall<ExecPolicy>(seg_half, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i) = (T)0;
  });

  // Assign values to dest using atomic view
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i / 2) += vec_view(i);
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

  for (RAJA::Index_type i = 0; i < N / 2; ++i) {
    EXPECT_EQ((T)2, dest[i]);
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(source));
  cudaErrchk(cudaFree(dest));
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
#if defined(RAJA_ENABLE_CUDA)
  T *dest_and = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest_and, sizeof(T) * N));

  T *dest_or = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest_or, sizeof(T) * N));

  T *dest_xor = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest_xor, sizeof(T) * N));

  cudaErrchk(cudaDeviceSynchronize());
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
    RAJA::atomicAnd<AtomicPolicy>(dest_and + offset,
                                          (T)(0xFF ^ (1 << bit)));
    RAJA::atomicOr<AtomicPolicy>(dest_or + offset, (T)(1 << bit));
    RAJA::atomicXor<AtomicPolicy>(dest_xor + offset, (T)(1 << bit));
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

  for (RAJA::Index_type i = 0; i < N; ++i) {
    EXPECT_EQ((T)0x00, dest_and[i]);
    EXPECT_EQ((T)0xFF, dest_or[i]);
    EXPECT_EQ((T)0xFF, dest_xor[i]);
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(dest_and));
  cudaErrchk(cudaFree(dest_or));
  cudaErrchk(cudaFree(dest_xor));
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

