//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCEBITAND_SANITY_HPP__
#define __TEST_FORALL_REDUCEBITAND_SANITY_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-reduce-sanity.hpp"

#include <cstdlib>
#include <numeric>

template <typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceBitAndSanityTest(RAJA::Index_type first, RAJA::Index_type last)
{
  RAJA::TypedRangeSegment<RAJA::Index_type> r1(first, last);

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

  for (RAJA::Index_type i = 0; i < last; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }

  DATA_TYPE ref_sum = 0;
  for (RAJA::Index_type i = first; i < last; ++i) {
    ref_sum &= test_array[i]; 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);


  RAJA::ReduceBitAnd<REDUCE_POLICY, DATA_TYPE> sum(0);
  RAJA::ReduceBitAnd<REDUCE_POLICY, DATA_TYPE> sum2(2);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    sum  &= working_array[idx];
    sum2 &= working_array[idx];
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(sum.get()), ref_sum);
  ASSERT_EQ(static_cast<DATA_TYPE>(sum2.get()), ref_sum);

  sum.reset(0);

  const int nloops = 3;

  for (int j = 0; j < nloops; ++j) {
    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
      sum &= working_array[idx];
    });
  }

  ASSERT_EQ(static_cast<DATA_TYPE>(sum.get()), ref_sum);
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_P(ForallReduceSanityTestBit, ReduceBitAndSanityForall)
{
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallReduceBitAndSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceBitAndSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceBitAndSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

#endif  // __TEST_FORALL_REDUCEBITOR_SANITY_HPP__
