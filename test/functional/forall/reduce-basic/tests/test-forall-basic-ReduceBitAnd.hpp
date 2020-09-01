//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCEBITAND_HPP__
#define __TEST_FORALL_BASIC_REDUCEBITAND_HPP__

#include <cstdlib>
#include <numeric>

template <typename IDX_TYPE, typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceBitAndBasicTestImpl(IDX_TYPE first, IDX_TYPE last)
{
  RAJA::TypedRangeSegment<IDX_TYPE> r1(first, last);

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(last,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  //
  // First a simple non-trivial test that is mildly interesting
  //
  for (IDX_TYPE i = 0; i < last; ++i) {
    test_array[i] = 13;
  }
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

  RAJA::ReduceBitAnd<REDUCE_POLICY, DATA_TYPE> simpand(21);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    simpand &= working_array[idx];
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(simpand.get()), 5);

  
  // 
  // And now a randomized test that pushes zeros around
  // 

  const int modval = 100;

  for (IDX_TYPE i = 0; i < last; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

  DATA_TYPE ref_and = 0;
  for (IDX_TYPE i = first; i < last; ++i) {
    ref_and &= test_array[i];
  }

  RAJA::ReduceBitAnd<REDUCE_POLICY, DATA_TYPE> redand(0);
  RAJA::ReduceBitAnd<REDUCE_POLICY, DATA_TYPE> redand2(2);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    redand  &= working_array[idx];
    redand2 &= working_array[idx];
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(redand.get()), ref_and);
  ASSERT_EQ(static_cast<DATA_TYPE>(redand2.get()), ref_and);

  redand.reset(0);

  const int nloops = 3;
  for (int j = 0; j < nloops; ++j) {
    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      redand &= working_array[idx];
    });
  }

  ASSERT_EQ(static_cast<DATA_TYPE>(redand.get()), ref_and);
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_SUITE_P(ForallReduceBitAndBasicTest);
template <typename T>
class ForallReduceBitAndBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceBitAndBasicTest, ReduceBitAndBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                  EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                  EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                  EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceBitAndBasicTest,
                            ReduceBitAndBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCEBITOR_HPP__
