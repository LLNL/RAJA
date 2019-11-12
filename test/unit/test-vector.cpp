//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic simd/simt vector operations
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"


using VectorTestTypes = ::testing::Types<

#ifdef __AVX__
    RAJA::FixedVector<RAJA::Register<RAJA::simd_avx_register, double,4>, 4>,
    RAJA::FixedVector<RAJA::Register<RAJA::simd_avx_register, double,4>, 8>,
    RAJA::StreamVector<RAJA::Register<RAJA::simd_avx_register, double,4>, 4>,
    RAJA::StreamVector<RAJA::Register<RAJA::simd_avx_register, double,4>, 8>,
#endif

    RAJA::FixedVector<RAJA::Register<RAJA::simd_scalar_register, double,1>, 3>,
    RAJA::FixedVector<RAJA::Register<RAJA::simd_scalar_register, double,1>, 5>,
    RAJA::StreamVector<RAJA::Register<RAJA::simd_scalar_register, double,1>, 1>,
    RAJA::StreamVector<RAJA::Register<RAJA::simd_scalar_register, double,1>, 3>>;

template <typename Policy>
class VectorTest : public ::testing::Test
{
protected:

  VectorTest() = default;
  virtual ~VectorTest() = default;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};
TYPED_TEST_CASE_P(VectorTest);



TYPED_TEST_P(VectorTest, ForallVectorRef1d)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;


  size_t N = 100*vector_t::s_num_elem;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
  if(!vector_t::s_is_fixed){
    N += (100*drand48());
  }

  element_t *A = new element_t[N];
  element_t *B = new element_t[N];
  element_t *C = new element_t[N];
  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<double, RAJA::Layout<1>> X(A, N);
  RAJA::View<double, RAJA::Layout<1>> Y(B, N);
  RAJA::View<double, RAJA::Layout<1>> Z(C, N);

  using policy_t = RAJA::simd_vector_exec<vector_t>;

  RAJA::forall<policy_t>(RAJA::TypedRangeSegment<size_t>(0, N),
      [=](RAJA::VectorIndex<size_t, vector_t> i)
  {
    Z[i] = 3+(X[i]*(5/Y[i]))+9;
  });


  for(size_t i = 0;i < N;i ++){
    ASSERT_DOUBLE_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }

  delete[] A;
  delete[] B;
  delete[] C;
}


TYPED_TEST_P(VectorTest, ForallVectorRef2d)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;


  size_t N = 128, M = 16*vector_t::s_num_elem;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
  if(!vector_t::s_is_fixed){
    N += (10*drand48());
    M += (10*drand48());
  }

  element_t *A = new element_t[N*M];
  element_t *B = new element_t[N*M];
  element_t *C = new element_t[N*M];
  for(size_t i = 0;i < N*M; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<double, RAJA::Layout<2>> X(A, N, M);
  RAJA::View<double, RAJA::Layout<2>> Y(B, N, M);
  RAJA::View<double, RAJA::Layout<2>> Z(C, N, M);

  using policy_t = RAJA::simd_vector_exec<vector_t>;

  ASSERT_EQ(A, X(0, RAJA::VectorIndex<size_t, vector_t>(0, 1)).get_pointer());
  ASSERT_EQ(A+M, X(1, RAJA::VectorIndex<size_t, vector_t>(0, 1)).get_pointer());
  ASSERT_EQ(A+1, X(0, RAJA::VectorIndex<size_t, vector_t>(1, 1)).get_pointer());


//      RAJA::KernelPolicy<
//        RAJA::statement::For<0, RAJA::loop_exec,
//          RAJA::statement::For<1, RAJA::simd_vector_exec<vector_t>,
//            RAJA::statement::Lambda<0>
//          >
//        >
//      >;


//  RAJA::kernel<policy_t>(
//      RAJA::make_tuple(RAJA::TypedRangeSegment<size_t>(0, N),
//                       RAJA::TypedRangeSegment<size_t>(0, M)),
//
//      [=](size_t i, RAJA::VectorIndex<size_t, vector_t> j)
//  {
//    Z(i,j) = 3+(X(i,j)*(5/Y(i,j)))+9;
//  });

  RAJA::forall<RAJA::loop_exec>(RAJA::TypedRangeSegment<size_t>(0, N),
      [=](size_t i){

    RAJA::forall<policy_t>(RAJA::TypedRangeSegment<size_t>(0, M),
        [=](RAJA::VectorIndex<size_t, vector_t> j){

      Z(i,j) = 3+(X(i,j)*(5/Y(i,j)))+9;
    });

  });

  for(size_t i = 0;i < N;i ++){
    ASSERT_DOUBLE_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }



  RAJA::forall<policy_t>(RAJA::TypedRangeSegment<size_t>(0, N),
      [=](RAJA::VectorIndex<size_t, vector_t> i){

    RAJA::forall<RAJA::loop_exec>(RAJA::TypedRangeSegment<size_t>(0, M),
        [=](size_t j){

      Z(i,j) = 3+(X(i,j)*(5/Y(i,j)))+9;
    });

  });

  for(size_t i = 0;i < N;i ++){
    ASSERT_DOUBLE_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }


  delete[] A;
  delete[] B;
  delete[] C;
}


REGISTER_TYPED_TEST_CASE_P(VectorTest, ForallVectorRef1d, ForallVectorRef2d);

INSTANTIATE_TYPED_TEST_CASE_P(SIMD, VectorTest, VectorTestTypes);
