//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall.
///

#ifndef __TEST_FORALL_ATOMIC_BASIC_HPP__
#define __TEST_FORALL_ATOMIC_BASIC_HPP__

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include "../../test-forall-atomic-utils.hpp"

TYPED_TEST_SUITE_P(ForallAtomicBasicFunctionalTest);

template <typename T>
class ForallAtomicBasicFunctionalTest : public ::testing::Test
{
};

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WORKINGRES,
          typename SegmentType,
          typename T>
void testAtomicFunctionBasic( RAJA::Index_type seglimit )
{
  SegmentType seg = RSMultiplexer<RAJA::Index_type, SegmentType>().makeseg(seglimit);

  // initialize an array
  const int len = 10;

  camp::resources::Resource work_res{WORKINGRES()};

  T * dest = work_res.allocate<T>(len);

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
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

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
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

  work_res.deallocate(dest);
}

TYPED_TEST_P(ForallAtomicBasicFunctionalTest, AtomicBasicFunctionalForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using SType   = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;
  testAtomicFunctionBasic<AExec, APol, ResType, SType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P( ForallAtomicBasicFunctionalTest,
                             AtomicBasicFunctionalForall
                           );

#endif  //__TEST_FORALL_ATOMIC_BASIC_HPP__
