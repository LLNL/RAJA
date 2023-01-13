//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA reducer constructors and initialization.
///

#ifndef __TEST_REDUCER_CONSTRUCTOR__
#define __TEST_REDUCER_CONSTRUCTOR__

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "../test-reducer.hpp"

template <typename T>
class ReducerBasicConstructorUnitTest : public ::testing::Test
{
};

template <typename T>
class ReducerInitConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ReducerBasicConstructorUnitTest);
TYPED_TEST_SUITE_P(ReducerInitConstructorUnitTest);

template <typename ReducePolicy,
          typename NumericType>
void testReducerConstructor()
{
  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum;
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min;
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max;
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc;

  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup;

  ASSERT_EQ((NumericType)reduce_sum.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_min.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_max.get(), NumericType());

  ASSERT_EQ((NumericType)reduce_minloc.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloc.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type());

  ASSERT_EQ((NumericType)reduce_minloctup.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
}

TYPED_TEST_P(ReducerBasicConstructorUnitTest, BasicReducerConstructor)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;

  testReducerConstructor< ReducePolicy, NumericType >();
}

template  < typename ReducePolicy,
            typename NumericType,
            typename ForOnePol >
typename  std::enable_if< // Host policy does nothing.
            std::is_base_of<RunOnHost, ForOnePol>::value
          >::type
exec_dispatcher( NumericType * RAJA_UNUSED_ARG(initVal) )
{
  // Do nothing for host policies.
}

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
template  < typename ReducePolicy,
            typename NumericType,
            typename ForOnePol >
typename  std::enable_if< // GPU policy fiddles with value.
            std::is_base_of<RunOnDevice, ForOnePol>::value
          >::type
exec_dispatcher( NumericType * initVal )
{
  forone<ForOnePol>( [=] __device__ () {
                        initVal[0] += 1;
                        initVal[0] -= 1;
                 });
}
#endif

template <typename ReducePolicy,
          typename NumericType,
          typename WORKING_RES,
          typename ForOnePol>
void testInitReducerConstructor()
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};
  camp::resources::Resource host_res{camp::resources::Host()};

  NumericType * theVal = nullptr;
  NumericType * workVal = nullptr;

  NumericType initVal = (NumericType)5;

  workVal = work_res.allocate<NumericType>(1);
  theVal = host_res.allocate<NumericType>(1);

  work_res.memcpy( workVal, &initVal, sizeof(initVal) );
  theVal[0] = (NumericType)10;

  #if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
  #endif

  #if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
  #endif

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(initVal);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(initVal);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(initVal);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(initVal, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(initVal, 1);

  RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(1, 1);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(initVal, LocTup);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(initVal, LocTup);

  // move a value onto device and fiddle
  exec_dispatcher < ReducePolicy,
                    NumericType,
                    ForOnePol
                  >
                  ( workVal );

  work_res.memcpy( &initVal, workVal, sizeof(initVal) );

  theVal[0] = initVal;

  ASSERT_EQ((NumericType)(theVal[0]), (NumericType)(initVal));

  ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(initVal));

  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(initVal));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)1);

  ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(initVal));
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);

  work_res.deallocate( workVal );
  host_res.deallocate( theVal );
}

TYPED_TEST_P(ReducerInitConstructorUnitTest, InitReducerConstructor)
{
  using ReduceType = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<2>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<3>>::type;

  testInitReducerConstructor< ReduceType, NumericType, ResourceType, ForOneType >();
}


REGISTER_TYPED_TEST_SUITE_P(ReducerBasicConstructorUnitTest,
                            BasicReducerConstructor);

REGISTER_TYPED_TEST_SUITE_P(ReducerInitConstructorUnitTest,
                            InitReducerConstructor);

#endif  //__TEST_REDUCER_CONSTRUCTOR__
