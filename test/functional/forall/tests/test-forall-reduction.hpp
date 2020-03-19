//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCTION_HPP__
#define __TEST_FORALL_REDUCTION_HPP__

#include "test-forall.hpp"

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCT_POLICY>
void ForallReductionFunctionalTest()
{
  int N = 128;
  RAJA::RangeSegment arange(0, N);

  Resource working_res{WORKING_RES()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);


  for (int i = 0; i < N; ++i) {
    if ( i % 2 == 0 ) {
      test_array[i] = 1;
    } else {
      test_array[i] = -1; 
    }
  }


  RAJA::ReduceSum<REDUCT_POLICY, INDEX_TYPE> red_sum(0);

  RAJA::forall<EXEC_POLICY>(arange, [=](INDEX_TYPE i) {
    red_sum += test_array[i];
  });

  std::cout << "Reduction Sum = " << red_sum.get() << std::endl;

}

TYPED_TEST_P(ForallFunctionalReductionTest, ReductionForall)
{
  using INDEX_TYPE    = typename at<TypeParam, num<0>>::type;
  using WORKING_RES   = typename at<TypeParam, num<1>>::type;
  using EXEC_POLICY   = typename at<TypeParam, num<2>>::type;
  using REDUCT_POLICY = typename at<TypeParam, num<3>>::type;

  ForallReductionFunctionalTest<INDEX_TYPE,WORKING_RES,EXEC_POLICY,REDUCT_POLICY>();

}

#endif  // __TEST_FORALL_REDUTCION_HPP__
