//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCEMAX_SANITY_HPP__
#define __TEST_FORALL_REDUCEMAX_SANITY_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-reduce-sanity.hpp"

#include <cstdlib>
#include <numeric>

template <typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMaxSanityTest(RAJA::Index_type first, RAJA::Index_type last)
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
  const DATA_TYPE max_init = -1;
  const DATA_TYPE big_max = modval + 1;

  for (RAJA::Index_type i = 0; i < last; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }

  DATA_TYPE ref_max = max_init;
  for (RAJA::Index_type i = first; i < last; ++i) {
    ref_max = RAJA_MAX(test_array[i], ref_max); 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> maxinit(big_max);
  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> max(max_init);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    maxinit.max( working_array[idx] );
    max.max( working_array[idx] );
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(maxinit.get()), big_max);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max);

#if !defined(RAJA_ENABLE_TARGET_OPENMP)
  //
  // Note: RAJA OpenMP target reductions do not currently support reset
  //
  max.reset(max_init);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), max_init);
#endif

  DATA_TYPE factor = 2;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    max.max( working_array[idx] * factor);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
   
  factor = 3;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    max.max( working_array[idx] * factor);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_P(ForallReduceSanityTest, ReduceMaxSanityForall)
{
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallReduceMaxSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceMaxSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceMaxSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

#endif  // __TEST_FORALL_REDUCEMAX_SANITY_HPP__
