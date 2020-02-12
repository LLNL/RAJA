//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall.
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"
#include "RAJA_value_params.hpp"


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicFunctionBasic()
{
  RAJA::RangeSegment seg(0, N);

// initialize an array
  const int len = 10;
#if defined(RAJA_ENABLE_CUDA)
  T *dest = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest, sizeof(T) * len));

  cudaErrchk(cudaDeviceSynchronize());

#else
  T *dest = new T[len];
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
  dest[8] = (T)N;
  dest[9] = (T)0;


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::atomicAdd<AtomicPolicy>(dest + 0, (T)1);
    RAJA::atomicSub<AtomicPolicy>(dest + 1, (T)1);
    RAJA::atomicMin<AtomicPolicy>(dest + 2, (T)i);
    RAJA::atomicMax<AtomicPolicy>(dest + 3, (T)i);
    RAJA::atomicInc<AtomicPolicy>(dest + 4);
    RAJA::atomicInc<AtomicPolicy>(dest + 5, (T)16);
    RAJA::atomicDec<AtomicPolicy>(dest + 6);
    RAJA::atomicDec<AtomicPolicy>(dest + 7, (T)16);
    RAJA::atomicExchange<AtomicPolicy>(dest + 8, (T)i);
    RAJA::atomicCAS<AtomicPolicy>(dest + 9, (T)i, (T)(i+1));
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
  EXPECT_LE((T)0, dest[8]);
  EXPECT_GT((T)N, dest[8]);
  EXPECT_LT((T)0, dest[9]);
  EXPECT_GE((T)N, dest[9]);


#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(dest));
#else
  delete[] dest;
#endif
}


// Type parameterized test for experimentation/discussion.
template <typename T>
struct AtomicFuncBasicFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(AtomicFuncBasicFunctionalTest);

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T>
void testAtomicFunctionBasicV2( RAJA::Index_type seglimit )
{
  RAJA::RangeSegment seg(0, seglimit);

// initialize an array
  const int len = 10;
#if defined(RAJA_ENABLE_CUDA)
  T *dest = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest, sizeof(T) * len));

  cudaErrchk(cudaDeviceSynchronize());

#else
  T *dest = new T[len];
#endif


  // use atomic add to reduce the array
  dest[0] = (T)0;
  dest[1] = (T)seglimit;
  dest[2] = (T)seglimit;
  dest[3] = (T)0;
  dest[4] = (T)0;
  dest[5] = (T)0;
  dest[6] = (T)seglimit + 1;
  dest[7] = (T)0;
  dest[8] = (T)seglimit;
  dest[9] = (T)0;


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::atomicAdd<AtomicPolicy>(dest + 0, (T)1);
    RAJA::atomicSub<AtomicPolicy>(dest + 1, (T)1);
    RAJA::atomicMin<AtomicPolicy>(dest + 2, (T)i);
    RAJA::atomicMax<AtomicPolicy>(dest + 3, (T)i);
    RAJA::atomicInc<AtomicPolicy>(dest + 4);
    RAJA::atomicInc<AtomicPolicy>(dest + 5, (T)16);
    RAJA::atomicDec<AtomicPolicy>(dest + 6);
    RAJA::atomicDec<AtomicPolicy>(dest + 7, (T)16);
    RAJA::atomicExchange<AtomicPolicy>(dest + 8, (T)i);
    RAJA::atomicCAS<AtomicPolicy>(dest + 9, (T)i, (T)(i+1));
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

  EXPECT_EQ((T)seglimit, dest[0]);
  EXPECT_EQ((T)0, dest[1]);
  EXPECT_EQ((T)0, dest[2]);
  EXPECT_EQ((T)seglimit - 1, dest[3]);
  EXPECT_EQ((T)seglimit, dest[4]);
  EXPECT_EQ((T)4, dest[5]);
  EXPECT_EQ((T)1, dest[6]);
  EXPECT_EQ((T)13, dest[7]);
  EXPECT_LE((T)0, dest[8]);
  EXPECT_GT((T)seglimit, dest[8]);
  EXPECT_LT((T)0, dest[9]);
  EXPECT_GE((T)seglimit, dest[9]);


#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(dest));
#else
  delete[] dest;
#endif
}
// END Type parameterized test for experimentation/discussion.


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


