//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_LISTSEGMENT_HPP__
#define __TEST_FORALL_LISTSEGMENT_HPP__

#include "test-forall-segment.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>

using namespace camp::resources;
using namespace camp;

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallListSegmentTest(INDEX_TYPE N)
{

  // Create and initialize indices in idx_array used to create list segment
  std::vector<INDEX_TYPE> idx_array;

  srand ( time(NULL) );

  for (INDEX_TYPE i = 0; i < N; ++i) {
    INDEX_TYPE randval = rand() % N;
    if ( i < randval ) {
      idx_array.push_back(i);
    }     
  }

  size_t idxlen = idx_array.size();

  Resource working_res{WORKING_RES()};

  // Create list segment for tests
  RAJA::NewTypedListSegment<INDEX_TYPE> lseg(&idx_array[0], idxlen,
                                             &working_res);
//RAJA::NewTypedListSegment<INDEX_TYPE> lseg(&idx_array[0], idxlen);
//RAJA::TypedListSegment<INDEX_TYPE> lseg(&idx_array[0], idxlen);

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

  std::for_each( std::begin(idx_array), std::end(idx_array), 
                 [=](INDEX_TYPE& idx ) { test_array[idx] = idx; }
               );

  RAJA::forall<EXEC_POLICY>(lseg, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
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


TYPED_TEST_P(ForallSegmentTest, ListSegmentForall)
{
  using INDEX_TYPE       = typename at<TypeParam, num<0>>::type;
  using WORKING_RESOURCE = typename at<TypeParam, num<1>>::type;
  using EXEC_POLICY      = typename at<TypeParam, num<2>>::type;

  ForallListSegmentTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(13);

  ForallListSegmentTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(2047);

  ForallListSegmentTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(32000);

}

#endif  // __TEST_FORALL_LISTSEGMENT_HPP__
