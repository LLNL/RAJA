//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_STATIC_MEM_HPP__
#define __TEST_LAUNCH_STATIC_MEM_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename TEAM_POLICY, typename THREAD_POLICY>
void LaunchStaticMemTestImpl(INDEX_TYPE block_range, INDEX_TYPE thread_range)
{

  RAJA::TypedRangeSegment<INDEX_TYPE> outer_range(RAJA::stripIndexType(INDEX_TYPE(0)), RAJA::stripIndexType(block_range));
  RAJA::TypedRangeSegment<INDEX_TYPE> inner_range(RAJA::stripIndexType(INDEX_TYPE(0)), RAJA::stripIndexType(thread_range));

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;
  
  size_t data_len = RAJA::stripIndexType(block_range)*RAJA::stripIndexType(thread_range);
  
  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);
  
  
  for(int b=0; b<RAJA::stripIndexType(block_range); ++b) {
    for(int c=0; c<RAJA::stripIndexType(thread_range); ++c) {
      int idx = c + RAJA::stripIndexType(thread_range)*b;
      test_array[idx] = INDEX_TYPE(b);
    }
  }

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(RAJA::stripIndexType(block_range)),
                        RAJA::Threads(RAJA::stripIndexType(thread_range))),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

      RAJA::loop<TEAM_POLICY>(ctx, outer_range, [&](INDEX_TYPE bid) {

          RAJA_TEAM_SHARED INDEX_TYPE Tile[1];

          RAJA::loop<THREAD_POLICY>(ctx, RAJA::TypedRangeSegment<INDEX_TYPE>(0,1), [&](INDEX_TYPE ) {
              Tile[0] = bid;
          });

          ctx.teamSync();

          RAJA::loop<THREAD_POLICY>(ctx, inner_range, [&](INDEX_TYPE tid) {
              INDEX_TYPE idx = tid + thread_range * bid;
              working_array[RAJA::stripIndexType(idx)] = Tile[0];
          });
          
          ctx.releaseSharedMemory();
        });

    });
  
  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);

  for (INDEX_TYPE i = INDEX_TYPE(0); i < data_len; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }
  
  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(LaunchStaticMemTest);
template <typename T>
class LaunchStaticMemTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchStaticMemTest, StaticMemLaunch)
{
  
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;
  using TEAM_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<2>>::type;
  

  LaunchStaticMemTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY>
    (INDEX_TYPE(4), INDEX_TYPE(2));

  LaunchStaticMemTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY>
    (INDEX_TYPE(5), INDEX_TYPE(32));
  
}

REGISTER_TYPED_TEST_SUITE_P(LaunchStaticMemTest,
                            StaticMemLaunch);

#endif  // __TEST_DYNAMIC_MEM_HPP__
