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
#include "../../test-forall-utils.hpp"
#include <numeric>

// range segment multiplexer
template< typename Index, typename SegType >
struct RSMultiplexer {};

template< typename Index >
struct RSMultiplexer < Index, RAJA::TypedRangeSegment<Index> >
{
  RAJA::TypedRangeSegment<Index>
  makeseg( Index N, camp::resources::Resource RAJA_UNUSED_ARG(work_res) )
  {
    return RAJA::TypedRangeSegment<Index>( 0, N );
  }
};

template< typename Index >
struct RSMultiplexer < Index, RAJA::TypedRangeStrideSegment<Index> >
{
  RAJA::TypedRangeStrideSegment<Index>
  makeseg( Index N, camp::resources::Resource RAJA_UNUSED_ARG(work_res) )
  {
    return RAJA::TypedRangeStrideSegment<Index>( 0, N, 1 );
  }
};

template< typename Index >
struct RSMultiplexer < Index, RAJA::TypedListSegment<Index> >
{
  RAJA::TypedListSegment<Index>
  makeseg( Index N, camp::resources::Resource work_res )
  {
    std::vector<Index> temp(N);
    std::iota( std::begin(temp), std::end(temp), 0 );
    return RAJA::TypedListSegment<Index>( &temp[0], static_cast<size_t>(temp.size()), work_res );
  }
};
// end range segment multiplexer


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
  // initialize an array
  const int len = 10;

  camp::resources::Resource work_res{WORKINGRES()};

  SegmentType seg = RSMultiplexer<RAJA::Index_type, SegmentType>().makeseg(seglimit, work_res);

  T * work_array;
  T * test_array;
  T * check_array;

  allocateForallTestData<T>(  len,
                              work_res,
                              &work_array,
                              &check_array,
                              &test_array );

  work_res.memcpy( work_array, test_array, sizeof(T) * len );

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  // use atomic add to reduce the array
  test_array[0] = (T)0;
  test_array[1] = (T)seglimit;
  test_array[2] = (T)seglimit;
  test_array[3] = (T)0;
  test_array[4] = (T)0;
  test_array[5] = (T)0;
  test_array[6] = (T)seglimit + 1;
  test_array[7] = (T)0;
  test_array[8] = (T)seglimit;
  test_array[9] = (T)0;

  work_res.memcpy( work_array, test_array, sizeof(T) * len );

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::atomicAdd<AtomicPolicy>(work_array + 0, (T)1);
    RAJA::atomicSub<AtomicPolicy>(work_array + 1, (T)1);
    RAJA::atomicMin<AtomicPolicy>(work_array + 2, (T)i);
    RAJA::atomicMax<AtomicPolicy>(work_array + 3, (T)i);
    RAJA::atomicInc<AtomicPolicy>(work_array + 4);
    RAJA::atomicInc<AtomicPolicy>(work_array + 5, (T)16);
    RAJA::atomicDec<AtomicPolicy>(work_array + 6);
    RAJA::atomicDec<AtomicPolicy>(work_array + 7, (T)16);
    RAJA::atomicExchange<AtomicPolicy>(work_array + 8, (T)i);
    RAJA::atomicCAS<AtomicPolicy>(work_array + 9, (T)i, (T)(i+1));
  });

  work_res.memcpy( check_array, work_array, sizeof(T) * len );

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  EXPECT_EQ((T)seglimit, check_array[0]);
  EXPECT_EQ((T)0, check_array[1]);
  EXPECT_EQ((T)0, check_array[2]);
  EXPECT_EQ((T)seglimit - 1, check_array[3]);
  EXPECT_EQ((T)seglimit, check_array[4]);
  EXPECT_EQ((T)4, check_array[5]);
  EXPECT_EQ((T)1, check_array[6]);
  EXPECT_EQ((T)13, check_array[7]);
  EXPECT_LE((T)0, check_array[8]);
  EXPECT_GT((T)seglimit, check_array[8]);
  EXPECT_LT((T)0, check_array[9]);
  EXPECT_GE((T)seglimit, check_array[9]);

  deallocateForallTestData<T>(  work_res,
                                work_array,
                                check_array,
                                test_array );
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
