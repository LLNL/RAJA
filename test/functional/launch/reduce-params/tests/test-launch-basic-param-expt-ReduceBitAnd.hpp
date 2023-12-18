//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_PARAM_EXPT_BASIC_REDUCEBITAND_HPP__
#define __TEST_LAUNCH_PARAM_EXPT_BASIC_REDUCEBITAND_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename DATA_TYPE,
          typename SEG_TYPE,
          typename LAUNCH_POLICY,
          typename GLOBAL_THREAD_POLICY>

void LaunchParamExptReduceBitAndBasicTestImpl(const SEG_TYPE& seg,
                                     const std::vector<IDX_TYPE>& seg_idx,
                                     camp::resources::Resource working_res)
{
  IDX_TYPE data_len = seg_idx[seg_idx.size() - 1] + 1;
  IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );

  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  constexpr int threads = 256;
  int blocks = (seg.size() - 1)/threads + 1;

  allocateForallTestData<DATA_TYPE>(data_len,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  //
  // First a simple non-trivial test that is mildly interesting
  //
  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = 13;
  }
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);

  DATA_TYPE simpand(21);

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
     RAJA::expt::Reduce<RAJA::operators::bit_and>(&simpand),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx, DATA_TYPE &_simpand) {

      RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {

          _simpand &= working_array[idx];
     });

  });

  ASSERT_EQ(static_cast<DATA_TYPE>(simpand), 5);


  //
  // And now a randomized test that pushes zeros around
  //

  const int modval = 100;

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);

  DATA_TYPE ref_and = 0;
  for (IDX_TYPE i = 0; i < idx_len; ++i) {
    ref_and &= test_array[ seg_idx[i] ];
  }

  DATA_TYPE redand(0);
  DATA_TYPE redand2(2);

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
     RAJA::expt::Reduce<RAJA::operators::bit_and>(&redand),
     RAJA::expt::Reduce<RAJA::operators::bit_and>(&redand2),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx, DATA_TYPE &_redand, DATA_TYPE &_redand2) {
      RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {
        _redand  &= working_array[idx];
        _redand2 &= working_array[idx];
    });
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(redand), ref_and);
  ASSERT_EQ(static_cast<DATA_TYPE>(redand2), ref_and);

  redand = 0;

  const int nloops = 3;
  for (int j = 0; j < nloops; ++j) {
    RAJA::launch<LAUNCH_POLICY>
      (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
       RAJA::expt::Reduce<RAJA::operators::bit_and>(&redand),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx, DATA_TYPE _redand) {
        RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {
          _redand &= working_array[idx];
      });
    });
  }

  ASSERT_EQ(static_cast<DATA_TYPE>(redand), ref_and);


  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_SUITE_P(LaunchParamExptReduceBitAndBasicTest);
template <typename T>
class LaunchParamExptReduceBitAndBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchParamExptReduceBitAndBasicTest, ReduceBitAndBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<3>>::type, camp::num<0>>::type;
  using GLOBAL_THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<3>>::type, camp::num<1>>::type;

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
  RAJA::getIndices(seg_idx, r1);
  LaunchParamExptReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                    r1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  LaunchParamExptReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                    r2, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  LaunchParamExptReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                    r3, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  LaunchParamExptReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                  LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                    r4, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  LaunchParamExptReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                  LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                    r5, seg_idx, working_res);

  // List segment tests
  seg_idx.clear();
  IDX_TYPE last = 10567;
  srand( time(NULL) );
  for (IDX_TYPE i = 0; i < last; ++i) {
    IDX_TYPE randval = IDX_TYPE( rand() % RAJA::stripIndexType(last) );
    if ( i < randval ) {
      seg_idx.push_back(i);
    }
  }
  RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(),
                                       working_res );
  LaunchParamExptReduceBitAndBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedListSegment<IDX_TYPE>,
                                  LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                    l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(LaunchParamExptReduceBitAndBasicTest,
                            ReduceBitAndBasicForall);

#endif  // __TEST_LAUNCH_BASIC_REDUCEBITOR_HPP__
