//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall.
///

#ifndef __TEST_FORALL_ATOMIC_BASIC_UNSIGNED_HPP__
#define __TEST_FORALL_ATOMIC_BASIC_UNSIGNED_HPP__

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
void ForallAtomicBasicUnsignedTestImpl( IdxType seglimit )
{
  // initialize an array
  const int len = 2;

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

  test_array[0] = (T)0;
  test_array[1] = (T)0;

  work_res.memcpy( work_array, test_array, sizeof(T) * len );

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType RAJA_UNUSED_ARG(i)) {
    RAJA::atomicInc<AtomicPolicy>(work_array + 0, (T)16);
    RAJA::atomicDec<AtomicPolicy>(work_array + 1, (T)16);
  });

  work_res.memcpy( check_array, work_array, sizeof(T) * len );
  work_res.wait();

  EXPECT_EQ((T)4, check_array[0]);
  EXPECT_EQ((T)13, check_array[1]);

  deallocateForallTestData<T>(  work_res,
                                work_array,
                                check_array,
                                test_array );
}

TYPED_TEST_SUITE_P(ForallAtomicBasicUnsignedTest);
template <typename T>
class ForallAtomicBasicUnsignedTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicBasicUnsignedTest, AtomicBasicUnsignedForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicBasicUnsignedTestImpl<AExec, APol, ResType,
                                    IdxType, RAJA::TypedRangeSegment<IdxType>,
                                    DType>( 10000 );
  ForallAtomicBasicUnsignedTestImpl<AExec, APol, ResType,
                                    IdxType, RAJA::TypedRangeStrideSegment<IdxType>,
                                    DType>( 10000 );
  ForallAtomicBasicUnsignedTestImpl<AExec, APol, ResType,
                                    IdxType, RAJA::TypedListSegment<IdxType>,
                                    DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicBasicUnsignedTest,
                            AtomicBasicUnsignedForall);

#endif  //__TEST_FORALL_ATOMIC_BASIC_UNSIGNED_HPP__
