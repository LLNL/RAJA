//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_VECTOR_ForallVectorRef2d_HPP__
#define __TEST_TESNOR_VECTOR_ForallVectorRef2d_HPP__

#include<RAJA/RAJA.hpp>

template <typename VECTOR_TYPE>
void ForallVectorRef2dImpl()
{

  using vector_t = VECTOR_TYPE;
  using element_t = typename vector_t::element_type;


  using index_t = ptrdiff_t;

  index_t N = 3*vector_t::s_num_elem+1;
  index_t M = 4*vector_t::s_num_elem+1;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
  N += (size_t)(10*NO_OPT_RAND);
  M += (size_t)(10*NO_OPT_RAND);

  element_t *A = new element_t[N*M];
  element_t *B = new element_t[N*M];
  element_t *C = new element_t[N*M];
  for(index_t i = 0;i < N*M; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<element_t, RAJA::Layout<2>> X(A, N, M);
  RAJA::View<element_t, RAJA::Layout<2>> Y(B, N, M);
  RAJA::View<element_t, RAJA::Layout<2>> Z(C, N, M);

  using idx_t = RAJA::VectorIndex<index_t, vector_t>;
  auto all = idx_t::all();




  //
  // Test with kernel, using sequential policies and ::all()
  //
  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }
  using policy1_t =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0>
        >
      >;





  // Test with kernel, using sequential policies and ::all()
  RAJA::kernel<policy1_t>(
      RAJA::make_tuple(RAJA::TypedRangeSegment<index_t>(0, N)),

      [=](index_t i)
  {
    Z(i,all) = 3+(X(i,all)*(5/Y(i,all)))+9;
  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }


#if 1

  //
  // Test with kernel, using tensor_exec policy
  //


  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }

  using policy2_t =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::For<1, RAJA::vector_exec<vector_t>,
            RAJA::statement::Lambda<0>
          >
        >
      >;

  RAJA::kernel<policy2_t>(
      RAJA::make_tuple(RAJA::TypedRangeSegment<index_t>(0, N),
                       RAJA::TypedRangeSegment<index_t>(0, M)),

      [=](index_t i, index_t j)
  {
    Z(i, j) = 3+(X(i, j)*(5/Y(i, j)))+9;
  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }




  //
  // Test with forall with vectors in i
  //
  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }
  RAJA::forall<RAJA::loop_exec>(RAJA::TypedRangeSegment<index_t>(0, M),
      [=](index_t j){


    Z(all,j) = 3+(X(all,j)*(5/Y(all,j)))+9;


  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }



  //
  // Test with forall with vectors in j
  //
  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }
  RAJA::forall<RAJA::loop_exec>(RAJA::TypedRangeSegment<index_t>(0, N),
      [=](index_t i){


    Z(i,all) = 3+(X(i,all)*(5/Y(i,all)))+9;


  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }


#endif


  delete[] A;
  delete[] B;
  delete[] C;
}



TYPED_TEST_P(TestTensorVector, ForallVectorRef2d)
{
  ForallVectorRef2dImpl<TypeParam>();
}


#endif
