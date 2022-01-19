//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_VECTOR_FmaFms_HPP__
#define __TEST_TESNOR_VECTOR_FmaFms_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void FmaFmsImpl()
{

  using vector_t = VECTOR_TYPE;
  using element_t = typename vector_t::element_type;


  element_t A[vector_t::s_num_elem];
  element_t B[vector_t::s_num_elem];
  element_t C[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
    B[i] = (element_t)i*2;
    C[i] = (element_t)i*3;
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load arrays as vectors
    vector_t vec_A;
    vec_A.load_packed_n(&A[0], N);

    vector_t vec_B;
    vec_B.load_packed_n(&B[0], N);

    vector_t vec_C;
    vec_C.load_packed_n(&C[0], N);


    // check FMA (A*B+C)

    vector_t fma = vec_A.multiply_add(vec_B, vec_C);
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fma.get(i), A[i]*B[i]+C[i]);
    }

    // check FMS (A*B-C)
    vector_t fms = vec_A.multiply_subtract(vec_B, vec_C);
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fms.get(i), A[i]*B[i]-C[i]);
    }

  }

}



TYPED_TEST_P(TestTensorVector, FmaFms)
{
  FmaFmsImpl<TypeParam>();
}


#endif
