//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_NESTED_DIRECT_HPP__
#define __TEST_LAUNCH_NESTED_DIRECT_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY,
          typename THREAD_X_POLICY, typename THREAD_Y_POLICY, typename THREAD_Z_POLICY,
          typename TEAM_X_POLICY, typename TEAM_Y_POLICY, typename TEAM_Z_POLICY>
void LaunchNestedDirectTestImpl(INDEX_TYPE M)
{

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(0, 2*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r2(0, 3*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r3(0, 4*M);

  RAJA::TypedRangeSegment<INDEX_TYPE> r4(0, 4*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r5(0, 5*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r6(0, 6*M);

  INDEX_TYPE N1 = static_cast<INDEX_TYPE>(r1.end() - r1.begin());
  INDEX_TYPE N2 = static_cast<INDEX_TYPE>(r2.end() - r2.begin());
  INDEX_TYPE N3 = static_cast<INDEX_TYPE>(r3.end() - r3.begin());

  INDEX_TYPE N4 = static_cast<INDEX_TYPE>(r4.end() - r4.begin());
  INDEX_TYPE N5 = static_cast<INDEX_TYPE>(r5.end() - r5.begin());
  INDEX_TYPE N6 = static_cast<INDEX_TYPE>(r6.end() - r6.begin());

  INDEX_TYPE N = static_cast<INDEX_TYPE>(N1 * N2 *                                          
                                         N3 * N4 *
                                         N5 * N6);                                         

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  size_t data_len = RAJA::stripIndexType(N);
  if ( data_len == 0 ) {
    data_len = 1;
  }

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);
  //6 threads total
  constexpr int threads_x = 2;
  constexpr int threads_y = 3;
  constexpr int threads_z = 4;

  constexpr int blocks_x = 4;
  constexpr int blocks_y = 5;
  constexpr int blocks_z = 6;

  if ( RAJA::stripIndexType(N) > 0 ) {

    std::iota(test_array, test_array + RAJA::stripIndexType(N), 0);

    constexpr int DIM = 6;
    using layout_t = RAJA::Layout<DIM, INDEX_TYPE,DIM-1>;
    RAJA::View<INDEX_TYPE, layout_t> Aview(working_array, N6, N5, N4, N3, N2, N1);

    RAJA::launch<LAUNCH_POLICY>
      (RAJA::LaunchParams(RAJA::Teams(blocks_x, blocks_y, blocks_z), RAJA::Threads(threads_x, threads_y,threads_z)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<TEAM_Z_POLICY>(ctx, r6, [&](INDEX_TYPE bz) {
            RAJA::loop<TEAM_Y_POLICY>(ctx, r5, [&](INDEX_TYPE by) {
                RAJA::loop<TEAM_X_POLICY>(ctx, r4, [&](INDEX_TYPE bx) {

                    RAJA::loop<THREAD_Z_POLICY>(ctx, r3, [&](INDEX_TYPE tz) {
                        RAJA::loop<THREAD_Y_POLICY>(ctx, r2, [&](INDEX_TYPE ty) {
                            RAJA::loop<THREAD_X_POLICY>(ctx, r1, [&](INDEX_TYPE tx) {

                                auto idx = tx + N1 * (ty + N2 * (tz + N3 * (bx + N4 * (by + N5 * bz))));


                                Aview(bz, by, bx, tz, ty, tx) = static_cast<INDEX_TYPE>(idx);
                              });
                          });
                      });

                  });
              });
          });
    });
  } else { // zero-length segment

    memset(static_cast<void*>(test_array), 0, sizeof(INDEX_TYPE) * data_len);

    working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);

    RAJA::launch<LAUNCH_POLICY>
      (RAJA::LaunchParams(RAJA::Teams(blocks_x, blocks_y, blocks_z), RAJA::Threads(blocks_x, blocks_y ,blocks_z)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<TEAM_Z_POLICY>(ctx, r3, [&](INDEX_TYPE RAJA_UNUSED_ARG(bz)) {
            RAJA::loop<TEAM_Y_POLICY>(ctx, r2, [&](INDEX_TYPE RAJA_UNUSED_ARG(by)) {
                RAJA::loop<TEAM_X_POLICY>(ctx, r1, [&](INDEX_TYPE RAJA_UNUSED_ARG(bx)) {

                    RAJA::loop<THREAD_Z_POLICY>(ctx, r3, [&](INDEX_TYPE RAJA_UNUSED_ARG(tz)) {
                        RAJA::loop<THREAD_Y_POLICY>(ctx, r2, [&](INDEX_TYPE RAJA_UNUSED_ARG(ty)) {
                            RAJA::loop<THREAD_X_POLICY>(ctx, r1, [&](INDEX_TYPE RAJA_UNUSED_ARG(tx)) {

                                working_array[0]++;
                                
                              });
                          });
                      });

                  });
              });
          });
      });
  }

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);

  if (RAJA::stripIndexType(N) > 0) {
    
    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
      ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
    }
    
  } else {
    
    ASSERT_EQ(test_array[0], check_array[0]);
    
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(LaunchNestedDirectTest);
template <typename T>
class LaunchNestedDirectTest : public ::testing::Test
{
};


TYPED_TEST_P(LaunchNestedDirectTest, RangeSegmentTeams)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;

  using TEAM_Z_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using TEAM_Y_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<2>>::type;
  using TEAM_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<3>>::type;

  using THREAD_Z_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<4>>::type;
  using THREAD_Y_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<5>>::type;
  using THREAD_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<6>>::type;



  // test zero-length range segment
  LaunchNestedDirectTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                           THREAD_X_POLICY, THREAD_Y_POLICY, THREAD_Z_POLICY,
                           TEAM_X_POLICY, TEAM_Y_POLICY, TEAM_Z_POLICY>
    (INDEX_TYPE(0));

  //Keep at one since we are doing a direct thread test
  LaunchNestedDirectTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                           THREAD_X_POLICY, THREAD_Y_POLICY, THREAD_Z_POLICY,
                           TEAM_X_POLICY, TEAM_Y_POLICY, TEAM_Z_POLICY>
    (INDEX_TYPE(1));


}

REGISTER_TYPED_TEST_SUITE_P(LaunchNestedDirectTest,
                            RangeSegmentTeams);

#endif  // __TEST_LAUNCH_NESTED_DIRECT_HPP__
