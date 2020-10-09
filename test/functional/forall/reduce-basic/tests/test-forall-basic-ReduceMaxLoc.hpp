//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCEMAXLOC_HPP__
#define __TEST_FORALL_BASIC_REDUCEMAXLOC_HPP__

#include <cstdlib>
#include <numeric>
#include <iostream>

template <typename IDX_TYPE, typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMaxLocBasicTestImpl(IDX_TYPE first, IDX_TYPE last)
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

  const int modval = 100;
  const DATA_TYPE max_init = -modval;
  const IDX_TYPE maxloc_init = -1;
  const IDX_TYPE maxloc_idx = (last - first) * 2/3 + first;
  const DATA_TYPE big_max = modval+1;
  const IDX_TYPE big_maxloc = maxloc_init;

  for (IDX_TYPE i = 0; i < last; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }
  test_array[maxloc_idx] = static_cast<DATA_TYPE>(big_max);

  DATA_TYPE ref_max = max_init;
  IDX_TYPE ref_maxloc = maxloc_init;
  for (IDX_TYPE i = first; i < last; ++i) {
    if ( test_array[i] > ref_max ) {
       ref_max = test_array[i];
       ref_maxloc = i;
    } 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);


  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> maxinit(big_max, maxloc_init);
  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> max(max_init, maxloc_init);

  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    maxinit.maxloc( working_array[idx], idx );
    max.maxloc( working_array[idx], idx );
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(maxinit.get()), big_max);
  ASSERT_EQ(static_cast<IDX_TYPE>(maxinit.getLoc()), big_maxloc);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max);
  ASSERT_EQ(static_cast<IDX_TYPE>(max.getLoc()), ref_maxloc);

  max.reset(max_init, maxloc_init);
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), max_init);
  ASSERT_EQ(static_cast<IDX_TYPE>(max.getLoc()), maxloc_init);

  DATA_TYPE factor = 2;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    max.maxloc( working_array[idx] * factor, idx);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
  ASSERT_EQ(static_cast<IDX_TYPE>(max.getLoc()), ref_maxloc);
  
  factor = 3;
  RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) { 
    max.maxloc( working_array[idx] * factor, idx);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
  ASSERT_EQ(static_cast<IDX_TYPE>(max.getLoc()), ref_maxloc);
 

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceMaxLocBasicTest);
template <typename T>
class ForallReduceMaxLocBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceMaxLocBasicTest, ReduceMaxLocBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                  EXEC_POLICY, REDUCE_POLICY>(0, 28);
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                  EXEC_POLICY, REDUCE_POLICY>(3, 642);
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                  EXEC_POLICY, REDUCE_POLICY>(0, 2057);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceMaxLocBasicTest,
                            ReduceMaxLocBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCEMAXLOC_HPP__
