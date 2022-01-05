//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_MULTIPLE_REDUCEMAXLOC_HPP__
#define __TEST_FORALL_MULTIPLE_REDUCEMAXLOC_HPP__

#include <cfloat>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <random>

template <typename IDX_TYPE, 
          typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMaxLocMultipleTestImpl(IDX_TYPE first, 
                                        IDX_TYPE last)
{
  RAJA::TypedRangeSegment<IDX_TYPE> r1(first, last);

  IDX_TYPE index_len = last - first;

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(last,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const DATA_TYPE default_val = static_cast<DATA_TYPE>(-SHRT_MAX);
  const IDX_TYPE default_loc = -1;
  const DATA_TYPE big_val = 500;

  for (IDX_TYPE i = 0; i < last; ++i) {
    test_array[i] = default_val;
  }

  
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_real_distribution<double> dist(-100, 100);
  static std::uniform_real_distribution<double> dist2(0, index_len - 1);

  DATA_TYPE current_max        = default_val;
  IDX_TYPE current_loc = default_loc;

  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> max0(default_val, default_loc);;
  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> max1(default_val, default_loc);
  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> max2(big_val, default_loc);

  const int nloops = 8;
  for (int j = 0; j < nloops; ++j) {

    DATA_TYPE roll = static_cast<DATA_TYPE>( dist(mt) );
    IDX_TYPE max_index = static_cast<IDX_TYPE>(dist2(mt));

    if ( current_max < roll && test_array[max_index] < roll ) {
      test_array[max_index] = roll;
      current_max = RAJA_MAX( current_max, roll );
      current_loc = max_index;

      working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);
    }

    working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      max0.maxloc(working_array[idx], idx);
      max1.maxloc(2 * working_array[idx], idx);
      max2.maxloc(working_array[idx], idx);
    });

    ASSERT_EQ(current_max, static_cast<DATA_TYPE>(max0.get()));
    ASSERT_EQ(current_loc, static_cast<IDX_TYPE>(max0.getLoc()));

    ASSERT_EQ(current_max * 2, static_cast<DATA_TYPE>(max1.get()));
    ASSERT_EQ(current_loc, static_cast<IDX_TYPE>(max1.getLoc()));

    ASSERT_EQ(big_val, static_cast<DATA_TYPE>(max2.get()));
    ASSERT_EQ(default_loc, static_cast<IDX_TYPE>(max2.getLoc()));

  }

  max0.reset(default_val, default_loc); 
  max1.reset(default_val, default_loc); 
  max2.reset(big_val, default_loc); 

  const int nloops_b = 4;
  for (int j = 0; j < nloops_b; ++j) {

    DATA_TYPE roll = static_cast<DATA_TYPE>( dist(mt) );
    IDX_TYPE max_index = static_cast<IDX_TYPE>(dist2(mt));

    if ( current_max < roll && test_array[max_index] < roll ) {
      test_array[max_index] = roll;
      current_max = RAJA_MAX( current_max, roll );
      current_loc = max_index;

      working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);
    }

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      max0.maxloc(working_array[idx], idx);
      max1.maxloc(2 * working_array[idx], idx);
      max2.maxloc(working_array[idx], idx);    
    });

    ASSERT_EQ(current_max, static_cast<DATA_TYPE>(max0.get()));
    ASSERT_EQ(current_loc, static_cast<IDX_TYPE>(max0.getLoc()));

    ASSERT_EQ(current_max * 2, static_cast<DATA_TYPE>(max1.get()));
    ASSERT_EQ(current_loc, static_cast<IDX_TYPE>(max1.getLoc()));

    ASSERT_EQ(big_val, static_cast<DATA_TYPE>(max2.get()));
    ASSERT_EQ(default_loc, static_cast<IDX_TYPE>(max2.getLoc()));

  }

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceMaxLocMultipleTest);
template <typename T>
class ForallReduceMaxLocMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceMaxLocMultipleTest, ReduceMaxLocMultipleForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallReduceMaxLocMultipleTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES,
                                     EXEC_POLICY, REDUCE_POLICY>(0, 2115);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceMaxLocMultipleTest,
                            ReduceMaxLocMultipleForall);

#endif  // __TEST_FORALL_MULTIPLE_REDUCEMAXLOC_HPP__
