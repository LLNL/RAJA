//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCEMAXLOC_SANITY_HPP__
#define __TEST_FORALL_REDUCEMAXLOC_SANITY_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-reduce-sanity.hpp"

#include <cstdlib>
#include <numeric>

#include <iostream>

template <typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMaxLocSanityTest(RAJA::Index_type first, RAJA::Index_type last)
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
  const DATA_TYPE max_init = -modval;
  const RAJA::Index_type maxloc_init = -1;
  const RAJA::Index_type maxloc_idx = (last - first) * 2/3 + first;
  const DATA_TYPE big_max = modval+1;
  const RAJA::Index_type big_maxloc = maxloc_init;

  for (RAJA::Index_type i = 0; i < last; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }
  test_array[maxloc_idx] = static_cast<DATA_TYPE>(big_max);

  DATA_TYPE ref_max = max_init;
  RAJA::Index_type ref_maxloc = maxloc_init;
  for (RAJA::Index_type i = first; i < last; ++i) {
    if ( test_array[i] > ref_max ) {
       ref_max = test_array[i];
       ref_maxloc = i;
    } 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);


  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, RAJA::Index_type> maxinit(big_max, maxloc_init);
  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, RAJA::Index_type> max(max_init, maxloc_init);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    maxinit.maxloc( working_array[idx], idx );
    max.maxloc( working_array[idx], idx );
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(maxinit.get()), big_max);
  ASSERT_EQ(static_cast<RAJA::Index_type>(maxinit.getLoc()), big_maxloc);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max);
  ASSERT_EQ(static_cast<RAJA::Index_type>(max.getLoc()), ref_maxloc);

#if !defined(RAJA_ENABLE_TARGET_OPENMP)
  //
  // Note: RAJA OpenMP target reductions do not currently support reset
  //
  max.reset(max_init, maxloc_init);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), max_init);
  ASSERT_EQ(static_cast<RAJA::Index_type>(max.getLoc()), maxloc_init);
#endif

  DATA_TYPE factor = 2;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
    max.maxloc( working_array[idx] * factor, idx);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
  ASSERT_EQ(static_cast<RAJA::Index_type>(max.getLoc()), ref_maxloc);
  
  factor = 3;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) { 
    max.maxloc( working_array[idx] * factor, idx);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
  ASSERT_EQ(static_cast<RAJA::Index_type>(max.getLoc()), ref_maxloc);
 

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_P(ForallReduceSanityTest, ReduceMaxLocSanityForall)
{
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallReduceMaxLocSanityTest<DATA_TYPE, WORKING_RES, 
                               EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceMaxLocSanityTest<DATA_TYPE, WORKING_RES, 
                               EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceMaxLocSanityTest<DATA_TYPE, WORKING_RES, 
                               EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

#endif  // __TEST_FORALL_REDUCEMAXLOC_SANITY_HPP__
