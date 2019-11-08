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

#include "RAJA/pattern/register.hpp"
#include "RAJA/pattern/vector.hpp"


using VectorTestTypes = ::testing::Types<
    RAJA::FixedVector<RAJA::Register<RAJA::simd_register, double,4>, 4>,
    RAJA::FixedVector<RAJA::Register<RAJA::simd_register, double,4>, 8>,
    RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 4>,
    RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 8>>;


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


  size_t N = 8000;// + (100*drand48());

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


REGISTER_TYPED_TEST_CASE_P(VectorTest, ForallVectorRef1d);

INSTANTIATE_TYPED_TEST_CASE_P(SIMD, VectorTest, VectorTestTypes);
