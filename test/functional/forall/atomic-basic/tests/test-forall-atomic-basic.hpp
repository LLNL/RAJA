//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall.
///

#ifndef __TEST_FORALL_ATOMIC_BASIC_HPP__
#define __TEST_FORALL_ATOMIC_BASIC_HPP__

#include <numeric>

// segment multiplexer
template< typename IdxType, typename SegType >
struct RSMultiplexer {};

template< typename IdxType >
struct RSMultiplexer < IdxType, RAJA::TypedRangeSegment<IdxType> >
{
  RAJA::TypedRangeSegment<IdxType>
  makeseg( IdxType N, camp::resources::Resource RAJA_UNUSED_ARG(work_res) )
  {
    return RAJA::TypedRangeSegment<IdxType>( 0, N );
  }
};

template< typename IdxType >
struct RSMultiplexer < IdxType, RAJA::TypedRangeStrideSegment<IdxType> >
{
  RAJA::TypedRangeStrideSegment<IdxType>
  makeseg( IdxType N, camp::resources::Resource RAJA_UNUSED_ARG(work_res) )
  {
    return RAJA::TypedRangeStrideSegment<IdxType>( 0, N, 1 );
  }
};

template< typename IdxType >
struct RSMultiplexer < IdxType, RAJA::TypedListSegment<IdxType> >
{
  RAJA::TypedListSegment<IdxType>
  makeseg( IdxType N, camp::resources::Resource work_res )
  {
    std::vector<IdxType> temp(N);
    std::iota( std::begin(temp), std::end(temp), 0 );
    return RAJA::TypedListSegment<IdxType>( &temp[0], static_cast<size_t>(temp.size()), work_res );
  }
};
// end segment multiplexer


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WORKINGRES,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicBasicTestImpl( IdxType seglimit )
{
  // initialize an array
  const int len = 10;

  camp::resources::Resource work_res{WORKINGRES()};

  SegmentType seg = 
    RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);

  T * work_array;
  T * test_array;
  T * check_array;

  allocateForallTestData<T>(  len,
                              work_res,
                              &work_array,
                              &check_array,
                              &test_array );

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
  test_array[5] = (T)seglimit + 1;
  test_array[6] = (T)seglimit;
  test_array[7] = (T)0;
  test_array[8] = (T)0;
  test_array[9] = (T)0;

  work_res.memcpy( work_array, test_array, sizeof(T) * len );

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    RAJA::atomicAdd<AtomicPolicy>(work_array + 0, (T)1);
    RAJA::atomicSub<AtomicPolicy>(work_array + 1, (T)1);
    RAJA::atomicMin<AtomicPolicy>(work_array + 2, (T)i);
    RAJA::atomicMax<AtomicPolicy>(work_array + 3, (T)i);
    RAJA::atomicInc<AtomicPolicy>(work_array + 4);
    RAJA::atomicDec<AtomicPolicy>(work_array + 5);
    RAJA::atomicExchange<AtomicPolicy>(work_array + 6, (T)i);
    RAJA::atomicCAS<AtomicPolicy>(work_array + 7, (T)i, (T)(i+1));
    RAJA::atomicLoad<AtomicPolicy>(work_array + 8);
    RAJA::atomicStore<AtomicPolicy>(work_array + 9, (T)1);
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
  EXPECT_EQ((T)1, check_array[5]);
  EXPECT_LE((T)0, check_array[6]);
  EXPECT_GT((T)seglimit, check_array[6]);
  EXPECT_LT((T)0, check_array[7]);
  EXPECT_GE((T)seglimit, check_array[7]);
  EXPECT_EQ((T)0, check_array[8]);
  EXPECT_EQ((T)1, check_array[9]);

  deallocateForallTestData<T>(  work_res,
                                work_array,
                                check_array,
                                test_array );
}

TYPED_TEST_SUITE_P(ForallAtomicBasicTest);
template <typename T>
class ForallAtomicBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicBasicTest, AtomicBasicForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicBasicTestImpl<AExec, APol, ResType, 
                            IdxType, RAJA::TypedRangeSegment<IdxType>, 
                            DType>( 10000 );
  ForallAtomicBasicTestImpl<AExec, APol, ResType, 
                            IdxType, RAJA::TypedRangeStrideSegment<IdxType>, 
                            DType>( 10000 );
  ForallAtomicBasicTestImpl<AExec, APol, ResType, 
                            IdxType, RAJA::TypedListSegment<IdxType>, 
                            DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicBasicTest,
                            AtomicBasicForall);

#endif  //__TEST_FORALL_ATOMIC_BASIC_HPP__
