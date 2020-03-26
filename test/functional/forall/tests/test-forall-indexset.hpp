//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_HPP__
#define __TEST_FORALL_INDEXSET_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-utils.hpp"

TYPED_TEST_SUITE_P(ForallIndexSetTest);
template <typename T>
class ForallIndexSetTest : public ::testing::Test
{
};

#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>

using namespace camp::resources;
using namespace camp;

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallISetTest()
{

  using RangeSegType       = RAJA::TypedRangeSegment<INDEX_TYPE>;
  using RangeStrideSegType = RAJA::TypedRangeStrideSegment<INDEX_TYPE>;
  using ListSegType        = RAJA::TypedListSegment<INDEX_TYPE>;

  using IndexSetType = 
    RAJA::TypedIndexSet< RangeSegType, RangeStrideSegType, ListSegType >; 

  //
  //  Build vector of integers for creating List segments.
  //
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<INDEX_TYPE> lindices;
  INDEX_TYPE idx = 0;
  while (lindices.size() < 3000) {
    double dval = dist(gen);
    if (dval > 0.3) {
      lindices.push_back(idx);
    }
    idx++;
  }

  //
  // Construct index set with mix of Range, RangeStride, and List segments.
  //
  INDEX_TYPE rbeg;
  INDEX_TYPE rend;
  INDEX_TYPE stride;
  INDEX_TYPE last_idx;
  INDEX_TYPE lseg_len = static_cast<INDEX_TYPE>( lindices.size() );
  std::vector<INDEX_TYPE> lseg(lseg_len);
  std::vector<INDEX_TYPE> lseg_vec(lseg_len);

  IndexSetType iset;  

  // Create empty Range segment
  rbeg = 1;
  rend = 1;
  iset.push_back(RangeSegType(rbeg, rend));
  last_idx = rend;

  // Create Range segment
  rbeg = 1;
  rend = 1578;
  iset.push_back(RangeSegType(rbeg, rend));
  last_idx = rend;

  // Create List segment
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg[i] = lindices[i] + last_idx + 3;
  }
  iset.push_back(ListSegType(&lseg[0], lseg_len));
  last_idx = lseg[lseg_len - 1];

  // Create List segment using alternate ctor
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg_vec[i] = lindices[i] + last_idx + 3;
  }
  iset.push_back(ListSegType(lseg_vec));
  last_idx = lseg_vec[lseg_len - 1];

  // Create Range-stride segment
  rbeg = last_idx + 16;
  rend = rbeg + 2040;
  stride = 3;
  iset.push_back(RangeStrideSegType(rbeg, rend, stride));
  last_idx = rend;

  // Create Range segment
  rbeg = last_idx + 4;
  rend = rbeg + 2759;
  iset.push_back(RangeSegType(rbeg, rend));
  last_idx = rend;

  // Create List segment
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg[i] = lindices[i] + last_idx + 5;
  }
  iset.push_back(ListSegType(&lseg[0], lseg_len));
  last_idx = lseg[lseg_len - 1];

  // Create Range segment
  rbeg = last_idx + 1;
  rend = rbeg + 320;
  iset.push_back(RangeSegType(rbeg, rend));
  last_idx = rend;

  // Create List segment using alternate ctor
  for (INDEX_TYPE i = 0; i < lseg_len; ++i) {
    lseg_vec[i] = lindices[i] + last_idx + 7;
  }
  iset.push_back(ListSegType(lseg_vec));
  last_idx = lseg_vec[lseg_len - 1];

 
  //
  // Collect actual indices in index set for testing.
  //
  std::vector<INDEX_TYPE> is_indices; 
  getIndices(is_indices, iset);


  //
  // Working array length
  //
  const INDEX_TYPE N = last_idx + 1;

  //
  // Allocate and initialize arrays used in testing
  //  
  Resource working_res{WORKING_RES()};
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

  std::for_each( std::begin(is_indices), std::end(is_indices), 
                 [=](INDEX_TYPE& idx ) { test_array[idx] = idx; }
               );

  RAJA::forall<EXEC_POLICY>(iset, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
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


TYPED_TEST_P(ForallIndexSetTest, IndexSetForall)
{
  using INDEX_TYPE       = typename at<TypeParam, num<0>>::type;
  using WORKING_RESOURCE = typename at<TypeParam, num<1>>::type;
  using EXEC_POLICY      = typename at<TypeParam, num<2>>::type;

  ForallISetTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetTest,
                            IndexSetForall);

#endif  // __TEST_FORALL_INDEXSET_HPP__
