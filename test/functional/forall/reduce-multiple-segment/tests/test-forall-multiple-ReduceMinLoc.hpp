//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_MULTIPLE_REDUCEMINLOC_HPP__
#define __TEST_FORALL_MULTIPLE_REDUCEMINLOC_HPP__

#include <cfloat>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <random>

template <typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMinLocMultipleTestImpl(RAJA::Index_type first, 
                                        RAJA::Index_type last)
{
  RAJA::TypedRangeSegment<RAJA::Index_type> r1(first, last);

  RAJA::Index_type index_len = last - first;

  camp::resources::Resource working_res{WORKING_RES()};
  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(last,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const DATA_TYPE default_val = static_cast<DATA_TYPE>(SHRT_MAX);
  const RAJA::Index_type default_loc = -1;
  const DATA_TYPE big_val = -500;

  for (RAJA::Index_type i = 0; i < last; ++i) {
    test_array[i] = default_val;
  }

  
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_real_distribution<double> dist(-100, 100);
  static std::uniform_real_distribution<double> dist2(0, index_len - 1);

  DATA_TYPE current_min    = default_val;
  RAJA::Index_type current_loc = default_loc;

  RAJA::ReduceMinLoc<REDUCE_POLICY, DATA_TYPE, RAJA::Index_type> min0(default_val, default_loc);;
  RAJA::ReduceMinLoc<REDUCE_POLICY, DATA_TYPE, RAJA::Index_type> min1(default_val, default_loc);
  RAJA::ReduceMinLoc<REDUCE_POLICY, DATA_TYPE, RAJA::Index_type> min2(big_val, default_loc);

  const int nloops = 8;
  for (int j = 0; j < nloops; ++j) {

    DATA_TYPE roll = static_cast<DATA_TYPE>( dist(mt) );
    RAJA::Index_type min_index = static_cast<RAJA::Index_type>(dist2(mt));

    if ( current_min > roll && test_array[min_index] > roll ) {
      test_array[min_index] = roll;
      current_min = RAJA_MIN( current_min, roll );
      current_loc = min_index;

      working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);
    }

    working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
      min0.minloc(working_array[idx], idx);
      min1.minloc(2 * working_array[idx], idx);
      min2.minloc(working_array[idx], idx);
    });

    ASSERT_EQ(current_min, static_cast<DATA_TYPE>(min0.get()));
    ASSERT_EQ(current_loc, static_cast<RAJA::Index_type>(min0.getLoc()));

    ASSERT_EQ(current_min * 2, static_cast<DATA_TYPE>(min1.get()));
    ASSERT_EQ(current_loc, static_cast<RAJA::Index_type>(min1.getLoc()));

    ASSERT_EQ(big_val, static_cast<DATA_TYPE>(min2.get()));
    ASSERT_EQ(default_loc, static_cast<RAJA::Index_type>(min2.getLoc()));

  }

  min0.reset(default_val, default_loc); 
  min1.reset(default_val, default_loc); 
  min2.reset(big_val, default_loc); 

  const int nloops_b = 4;
  for (int j = 0; j < nloops_b; ++j) {

    DATA_TYPE roll = static_cast<DATA_TYPE>( dist(mt) );
    RAJA::Index_type min_index = static_cast<RAJA::Index_type>(dist2(mt));

    if ( current_min > roll && test_array[min_index] > roll ) {
      test_array[min_index] = roll;
      current_min = RAJA_MIN( current_min, roll );
      current_loc = min_index;

      working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);
    }

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(RAJA::Index_type idx) {
      min0.minloc(working_array[idx], idx);
      min1.minloc(2 * working_array[idx], idx);
      min2.minloc(working_array[idx], idx);    
    });

    ASSERT_EQ(current_min, static_cast<DATA_TYPE>(min0.get()));
    ASSERT_EQ(current_loc, static_cast<RAJA::Index_type>(min0.getLoc()));

    ASSERT_EQ(current_min * 2, static_cast<DATA_TYPE>(min1.get()));
    ASSERT_EQ(current_loc, static_cast<RAJA::Index_type>(min1.getLoc()));

    ASSERT_EQ(big_val, static_cast<DATA_TYPE>(min2.get()));
    ASSERT_EQ(default_loc, static_cast<RAJA::Index_type>(min2.getLoc()));

  }

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceMinLocMultipleTest);
template <typename T>
class ForallReduceMinLocMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceMinLocMultipleTest, ReduceMinLocMultipleForall)
{
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<2>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  ForallReduceMinLocMultipleTestImpl<DATA_TYPE, WORKING_RES, 
                                     EXEC_POLICY, REDUCE_POLICY>(0, 2115);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceMinLocMultipleTest,
                            ReduceMinLocMultipleForall);

#endif  // __TEST_FORALL_MULTIPLE_REDUCEMINLOC_HPP__
