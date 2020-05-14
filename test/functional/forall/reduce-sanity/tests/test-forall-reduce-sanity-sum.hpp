//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCESUM_SANITY_HPP__
#define __TEST_FORALL_REDUCESUM_SANITY_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-reduce-sanity.hpp"

#include <cstdlib>
#include <numeric>

template <typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceSumSanityTest(RAJA::Index_type first, RAJA::Index_type last)
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
    ref_sum += test_array[i]; 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);


  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum2(2);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    sum  += working_array[idx];
    sum2 += working_array[idx];
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(sum.get()), ref_sum);
  ASSERT_EQ(static_cast<DATA_TYPE>(sum2.get()), ref_sum + 2);

#if !defined(RAJA_ENABLE_TARGET_OPENMP)
  sum.reset(0);
#endif

  const int nloops = 2;

  for (int j = 0; j < nloops; ++j) {
    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
      sum += working_array[idx];
    });
  }

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  ASSERT_EQ(static_cast<DATA_TYPE>(sum.get()), nloops * ref_sum + ref_sum);
#else
  ASSERT_EQ(static_cast<DATA_TYPE>(sum.get()), nloops * ref_sum);
#endif
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_P(ForallReduceSanityTest, ReduceSumSanityForall)
{
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallReduceSumSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceSumSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceSumSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

#endif  // __TEST_FORALL_REDUCESUM_SANITY_HPP__
