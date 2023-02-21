//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_VECTOR_CtorGetSet_HPP__
#define __TEST_TESNOR_VECTOR_CtorGetSet_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void CtorGetSetImpl()
{

  using vector_t = VECTOR_TYPE;
  using element_t = typename vector_t::element_type;



  element_t A[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)(i*2);
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load array A as vector
    vector_t vec;
    vec.load_packed_n(&A[0], N);

    // check get operations
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(vec.get(i), (element_t)(i*2));
    }

    // check set operations
    for(camp::idx_t i = 0;i < N;++ i){
      vec.set((element_t)(i+1), i);
    }
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(vec.get(i), (element_t)(i+1));
    }

  }

}



TYPED_TEST_P(TestTensorVector, CtorGetSet)
{
  CtorGetSetImpl<TypeParam>();
}


#endif
