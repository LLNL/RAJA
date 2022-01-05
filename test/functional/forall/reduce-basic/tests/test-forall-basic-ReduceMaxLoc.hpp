//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCEMAXLOC_HPP__
#define __TEST_FORALL_BASIC_REDUCEMAXLOC_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename DATA_TYPE,
          typename SEG_TYPE,
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMaxLocBasicTestImpl(const SEG_TYPE& seg,
                                     const std::vector<IDX_TYPE>& seg_idx,
                                     camp::resources::Resource working_res)
{
  IDX_TYPE data_len = seg_idx[seg_idx.size() - 1] + 1;
  IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );

  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(data_len,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const int modval = 100;
  const DATA_TYPE max_init = -modval;
  const IDX_TYPE maxloc_init = -1;
  const IDX_TYPE maxloc_idx = seg_idx[ idx_len * 2/3 ];
  const DATA_TYPE big_max = modval+1;
  const IDX_TYPE big_maxloc = maxloc_init;

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }
  test_array[maxloc_idx] = static_cast<DATA_TYPE>(big_max);

  DATA_TYPE ref_max = max_init;
  IDX_TYPE ref_maxloc = maxloc_init;
  for (IDX_TYPE i = 0; i < idx_len; ++i) {
    if ( test_array[ seg_idx[i] ] > ref_max ) {
       ref_max = test_array[ seg_idx[i] ];
       ref_maxloc = seg_idx[i];
    } 
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);


  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> maxinit(big_max, maxloc_init);
  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, IDX_TYPE> max(max_init, maxloc_init);

  RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
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
  RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
    max.maxloc( working_array[idx] * factor, idx);
  });
  ASSERT_EQ(static_cast<DATA_TYPE>(max.get()), ref_max * factor);
  ASSERT_EQ(static_cast<IDX_TYPE>(max.getLoc()), ref_maxloc);
  
  factor = 3;
  RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) { 
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

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
  RAJA::getIndices(seg_idx, r1);
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r2, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r3, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    r4, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
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
  ForallReduceMaxLocBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                  RAJA::TypedListSegment<IDX_TYPE>,
                                  EXEC_POLICY, REDUCE_POLICY>(
                                    l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceMaxLocBasicTest,
                            ReduceMaxLocBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCEMAXLOC_HPP__
