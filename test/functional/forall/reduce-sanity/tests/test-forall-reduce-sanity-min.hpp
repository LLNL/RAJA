//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCEMIN_SANITY_HPP__
#define __TEST_FORALL_REDUCEMIN_SANITY_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-reduce-sanity.hpp"

#include <cstdlib>
#include <numeric>

template <typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMinSanityTest(RAJA::Index_type first, RAJA::Index_type last)
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
  const DATA_TYPE min_init = modval+1;
  const DATA_TYPE small_min = -modval;

  for (RAJA::Index_type i = 0; i < last; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }

  DATA_TYPE ref_min = min_init;
  for (RAJA::Index_type i = first; i < last; ++i) {
    ref_min = RAJA_MIN(test_array[i], ref_min); 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);


  RAJA::ReduceMin<REDUCE_POLICY, DATA_TYPE> mininit(small_min);
  RAJA::ReduceMin<REDUCE_POLICY, DATA_TYPE> min(min_init);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    mininit.min( working_array[idx] );
    min.min( working_array[idx] );
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(mininit.get()), small_min);
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min);

#if !defined(RAJA_ENABLE_TARGET_OPENMP)
  //
  // Note: RAJA OpenMP target reductions do not currently support reset
  //
  min.reset(min_init);
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), min_init);
#endif

  DATA_TYPE factor = 3; 
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    min.min( working_array[idx] * factor);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min * factor);

  factor = 2;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) { 
    min.min( working_array[idx] * factor);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min * factor);
   

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_P(ForallReduceSanityTest, ReduceMinSanityForall)
{
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallReduceMinSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceMinSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceMinSanityTest<DATA_TYPE, WORKING_RES, 
                            EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

#endif  // __TEST_FORALL_REDUCEMIN_SANITY_HPP__
