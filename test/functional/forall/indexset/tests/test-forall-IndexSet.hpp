//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_HPP__
#define __TEST_FORALL_INDEXSET_HPP__

#include <cstdio>
#include <algorithm>
#include <vector>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallIndexSetTestImpl()
{

  using RangeSegType       = RAJA::TypedRangeSegment<INDEX_TYPE>;
  using RangeStrideSegType = RAJA::TypedRangeStrideSegment<INDEX_TYPE>;
  using ListSegType        = RAJA::TypedListSegment<INDEX_TYPE>;

  using IndexSetType = 
   RAJA::TypedIndexSet< RangeSegType, RangeStrideSegType, ListSegType >; 

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  IndexSetType iset; 
  std::vector<INDEX_TYPE> is_indices; 
  buildIndexSet<INDEX_TYPE, RangeSegType, RangeStrideSegType, ListSegType>(
    iset, is_indices, working_res);

  //
  // Working array length
  //
  const INDEX_TYPE N = is_indices[ is_indices.size() - 1 ] + 1;

  //
  // Allocate and initialize arrays used in testing
  //  
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  memset( test_array, 0, sizeof(INDEX_TYPE) * N );  

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * N);

  for (size_t i = 0; i < is_indices.size(); ++i) {
    test_array[ is_indices[i] ] = is_indices[i];
  }

  RAJA::forall(EXEC_POLICY(), iset, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    working_array[idx] = idx;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * N);

  // 
  for (INDEX_TYPE i = 0; i < N; i++) {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(ForallIndexSetTest);
template <typename T>
class ForallIndexSetTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallIndexSetTest, IndexSetForall)
{
  using INDEX_TYPE       = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<2>>::type;

  ForallIndexSetTestImpl<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetTest,
                            IndexSetForall);

#endif  // __TEST_FORALL_INDEXSET_HPP__
