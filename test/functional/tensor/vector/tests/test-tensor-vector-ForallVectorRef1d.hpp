//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_VECTOR_ForallVectorRef1d_HPP__
#define __TEST_TESNOR_VECTOR_ForallVectorRef1d_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void ForallVectorRef1dImpl()
{

  using vector_t = VECTOR_TYPE;
  using element_t = typename vector_t::element_type;


  size_t N = 10*vector_t::s_num_elem+1;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
    //N += (size_t)(100*NO_OPT_RAND);


  element_t *A = new element_t[N];
  element_t *B = new element_t[N];
  element_t *C = new element_t[N];
  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<element_t, RAJA::Layout<1>> X(A, N);
  RAJA::View<element_t, RAJA::Layout<1>> Y(B, N);
  RAJA::View<element_t, RAJA::Layout<1>> Z(C, N);


  using idx_t = RAJA::VectorIndex<int, vector_t>;

  auto all = idx_t::all();

  Z[all] = 3 + (X[all]*(5/Y[all])) + 9;

//  for(size_t i = 0;i < N; ++ i){
//    printf("%lf ", (double)C[i]);
//  }
//  printf("\n\n");

  for(size_t i = 0;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }


  for(size_t i = 0;i < N; ++ i){
    C[i] = 0.0;
  }

  // evaluate on a subrange [N/2, N)
  auto some = idx_t::range(N/2, N);
  Z[some] = 3.+ (X[some]*(5/Y[some])) + 9;


  for(size_t i = 0;i < N/2;i ++){
    ASSERT_SCALAR_EQ(0, C[i]);
  }
  for(size_t i = N/2;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }




  // evaluate on a subrange [0, N/2) using a forall statement
  for(size_t i = 0;i < N; ++ i){
    C[i] = 0.0;
  }
  RAJA::forall<RAJA::vector_exec<vector_t>>(RAJA::TypedRangeSegment<int>(0,N/2),
      [=](int i){

     Z[i] = 3 + (X[i]*(5/Y[i])) + 9;
  });


  for(size_t i = 0;i < N/2;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }
  for(size_t i = N/2;i < N;i ++){
    ASSERT_SCALAR_EQ(0, C[i]);
  }





  delete[] A;
  delete[] B;
  delete[] C;
}



TYPED_TEST_P(TestTensorVector, ForallVectorRef1d)
{
  ForallVectorRef1dImpl<TypeParam>();
}


#endif
