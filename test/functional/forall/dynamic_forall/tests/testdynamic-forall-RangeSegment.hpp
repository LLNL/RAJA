//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_DYNAMIC_FORALL_RANGESEGMENT_HPP__
#define __TEST_DYNAMIC_FORALL_RANGESEGMENT_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXE_POL>(INDEX_TYPE first, INDEX_TYPE last, int pol)
void DynamicForallRangeSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last, int pol)
{

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last));
  INDEX_TYPE N = static_cast<INDEX_TYPE>(r1.end() - r1.begin());

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

    const INDEX_TYPE rbegin = *r1.begin();

    std::iota(test_array, test_array + RAJA::stripIndexType(N), rbegin);

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      working_array[RAJA::stripIndexType(idx - rbegin)] = idx;
    });

  } else { // zero-length segment 

    memset(static_cast<void*>(test_array), 0, sizeof(INDEX_TYPE) * data_len);

    working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
      (void) idx;
      working_array[0]++;
    });

  }

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(DynamicForallRangeSegmentTest);
template <typename T>
class DynamicForallRangeSegmentTest : public ::testing::Test
{
};

TYPED_TEST_P(DynamicForallRangeSegmentTest, DynamicRangeSegmentForall)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  DynamicForallRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POL>(INDEX_TYPE(3), INDEX_TYPE(10));

}

REGISTER_TYPED_TEST_SUITE_P(DynamicForallRangeSegmentTest,
                            BasicSharedTeams);

#endif  // __TEST_BASIC_SHARED_HPP__
