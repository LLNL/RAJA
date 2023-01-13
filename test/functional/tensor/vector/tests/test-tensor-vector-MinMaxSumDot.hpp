//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_VECTOR_MinMaxSumDot_HPP__
#define __TEST_TENSOR_VECTOR_MinMaxSumDot_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void MinMaxSumDotImpl()
{

  using vector_t = VECTOR_TYPE;
  using element_t = typename vector_t::element_type;


  element_t A[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load array A as vector
    vector_t vec;
    vec.load_packed_n(&A[0], N);

    // check min
    ASSERT_SCALAR_EQ(vec.min_n(N), (element_t)0);

    // check max
    ASSERT_SCALAR_EQ(vec.max_n(N), (element_t)(N-1));

    // compute expected values
    element_t ex_sum(0);
    element_t ex_dot(0);
    for(camp::idx_t i = 0;i < N;++ i){
      ex_sum += A[i];
      ex_dot += A[i]*A[i];
    }

    // check sum
    ASSERT_SCALAR_EQ(vec.sum(), ex_sum);

    // check dot
    ASSERT_SCALAR_EQ(vec.dot(vec), ex_dot);

  }
}



TYPED_TEST_P(TestTensorVector, MinMaxSumDot)
{
  MinMaxSumDotImpl<TypeParam>();
}


#endif
