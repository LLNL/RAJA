//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCEMAX_HPP__
#define __TEST_FORALL_BASIC_REDUCEMAX_HPP__

#include <cstdlib>
#include <numeric>

template <typename IDX_TYPE, typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMaxBasicTestImpl(IDX_TYPE first, IDX_TYPE last)
{
  RAJA::TypedRangeSegment<IDX_TYPE> r1(first, last);

  camp::resources::Resource working_res{WORKING_RES()};
  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(last,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const int modval = 100;
  const DATA_TYPE max_init = -1;
  const DATA_TYPE big_max = modval + 1;

  for (IDX_TYPE i = 0; i < last; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }

  DATA_TYPE ref_max = max_init;
  for (IDX_TYPE i = first; i < last; ++i) {
    ref_max = RAJA_MAX(test_array[i], ref_max); 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> maxinit(big_max);
  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> max(max_init);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    maxinit.max( working_array[idx] );
    max.max( working_array[idx] );
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(maxinit.get()), big_max);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max);

  max.reset(max_init);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), max_init);

  DATA_TYPE factor = 2;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    max.max( working_array[idx] * factor);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
   
  factor = 3;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    max.max( working_array[idx] * factor);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceMaxBasicTest);
template <typename T>
class ForallReduceMaxBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceMaxBasicTest, ReduceMaxBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallReduceMaxBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceMaxBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceMaxBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceMaxBasicTest,
                            ReduceMaxBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCEMAX_HPP__
