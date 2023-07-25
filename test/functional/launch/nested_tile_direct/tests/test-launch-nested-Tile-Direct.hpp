//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_NESTED_TILE_DIRECT_HPP__
#define __TEST_LAUNCH_NESTED_TILE_DIRECT_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY,
          typename THREAD_X_POLICY, typename THREAD_Y_POLICY, typename THREAD_Z_POLICY,
          typename TEAM_X_POLICY, typename TEAM_Y_POLICY, typename TEAM_Z_POLICY>
void LaunchNestedTileDirectTestImpl(INDEX_TYPE M)
{

  constexpr int tile_size_x = 2;
  constexpr int tile_size_y = 3;
  constexpr int tile_size_z = 4;

  constexpr int threads_x = 2*tile_size_x;
  constexpr int threads_y = 3*tile_size_y;
  constexpr int threads_z = 4*tile_size_z;

  constexpr int blocks_x = 4;
  constexpr int blocks_y = 5;
  constexpr int blocks_z = 6;

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(0, tile_size_x*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r2(0, tile_size_y*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r3(0, tile_size_z*M);

  INDEX_TYPE N1 = static_cast<INDEX_TYPE>(r1.end() - r1.begin());
  INDEX_TYPE N2 = static_cast<INDEX_TYPE>(r2.end() - r2.begin());
  INDEX_TYPE N3 = static_cast<INDEX_TYPE>(r3.end() - r3.begin());

  INDEX_TYPE N = static_cast<INDEX_TYPE>(N1 *
                                         N2 *
                                         N3);

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

  if ( RAJA::stripIndexType(N) > 0 ) {

    std::iota(test_array, test_array + RAJA::stripIndexType(N), 0);

    constexpr int DIM = 3;
    using layout_t = RAJA::Layout<DIM, INDEX_TYPE,DIM-1>;
    RAJA::View<INDEX_TYPE, layout_t> Aview(working_array, N3, N2, N1);

    RAJA::launch<LAUNCH_POLICY>
      (RAJA::LaunchParams(RAJA::Teams(blocks_x, blocks_y, blocks_z), RAJA::Threads(threads_x, threads_y,threads_z)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::tile<TEAM_Z_POLICY>(ctx, tile_size_z, r3, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &z_tile) {
            RAJA::tile<TEAM_Y_POLICY>(ctx, tile_size_y, r2, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &y_tile) {
                RAJA::tile<TEAM_X_POLICY>(ctx, tile_size_x, r1, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &x_tile) {

                    RAJA::loop<THREAD_Z_POLICY>(ctx, z_tile, [&](INDEX_TYPE tz) {
                        RAJA::loop<THREAD_Y_POLICY>(ctx, y_tile, [&](INDEX_TYPE ty) {
                            RAJA::loop<THREAD_X_POLICY>(ctx, x_tile, [&](INDEX_TYPE tx) {

                                auto idx = tx + N1 * (ty + N2 * tz);

                                Aview(tz, ty, tx) = static_cast<INDEX_TYPE>(idx);

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

        RAJA::tile<TEAM_Z_POLICY>(ctx, threads_z, r3, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &z_tile) {
            RAJA::tile<TEAM_Y_POLICY>(ctx, threads_y, r2, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &y_tile) {
                RAJA::tile<TEAM_X_POLICY>(ctx, threads_x, r1, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &x_tile) {

                    RAJA::loop<THREAD_Z_POLICY>(ctx, z_tile, [&](INDEX_TYPE RAJA_UNUSED_ARG(tz)) {
                        RAJA::loop<THREAD_Y_POLICY>(ctx, y_tile, [&](INDEX_TYPE RAJA_UNUSED_ARG(ty)) {
                            RAJA::loop<THREAD_X_POLICY>(ctx, x_tile, [&](INDEX_TYPE RAJA_UNUSED_ARG(tx)) {

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


TYPED_TEST_SUITE_P(LaunchNestedTileDirectTest);
template <typename T>
class LaunchNestedTileDirectTest : public ::testing::Test
{
};


TYPED_TEST_P(LaunchNestedTileDirectTest, RangeSegmentTeams)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;

  using THREAD_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using THREAD_Y_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<2>>::type;
  using THREAD_Z_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<3>>::type;

  using TEAM_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<4>>::type;
  using TEAM_Y_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<5>>::type;
  using TEAM_Z_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<6>>::type;


  // test zero-length range segment
  LaunchNestedTileDirectTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                           THREAD_X_POLICY, THREAD_Y_POLICY, THREAD_Z_POLICY,
                           TEAM_X_POLICY, TEAM_Y_POLICY, TEAM_Z_POLICY>
    (INDEX_TYPE(0));

  //Keep at one since we are doing a direct thread test
  LaunchNestedTileDirectTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                           THREAD_X_POLICY, THREAD_Y_POLICY, THREAD_Z_POLICY,
                           TEAM_X_POLICY, TEAM_Y_POLICY, TEAM_Z_POLICY>
    (INDEX_TYPE(1));


}

REGISTER_TYPED_TEST_SUITE_P(LaunchNestedTileDirectTest,
                            RangeSegmentTeams);

#endif  // __TEST_LAUNCH_NESTED_TILE_DIRECT_HPP__
