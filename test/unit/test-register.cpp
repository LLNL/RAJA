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


    using RegisterTestTypes = ::testing::Types<
#ifdef __AVX__
       RAJA::Register<RAJA::vector_avx_register, double, 2>,
       RAJA::Register<RAJA::vector_avx_register, double, 3>,
       RAJA::Register<RAJA::vector_avx_register, double, 4>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,1>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,2>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,3>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,4>, 27>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx_register, double,4>, 4>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx_register, double,4>, 8>,
#endif

#ifdef __AVX2__
       RAJA::Register<RAJA::vector_avx2_register, double, 2>,
       RAJA::Register<RAJA::vector_avx2_register, double, 3>,
       RAJA::Register<RAJA::vector_avx2_register, double, 4>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,1>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,2>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,3>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,4>, 27>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,4>, 4>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,4>, 8>,
#endif
       RAJA::Register<RAJA::vector_scalar_register, int, 1>,
       RAJA::Register<RAJA::vector_scalar_register, float, 1>,
       RAJA::Register<RAJA::vector_scalar_register, double, 1>,

       // Test automatically wrapped types to make things easier for users
       RAJA::StreamVector<double>,
       RAJA::StreamVector<double, 2>,
       RAJA::FixedVector<double, 1>,
       RAJA::FixedVector<double, 2>,
       RAJA::FixedVector<double, 4>,
       RAJA::FixedVector<double, 8>,
       RAJA::FixedVector<double, 16>>;
template <typename NestedPolicy>
class RegisterTest : public ::testing::Test
{
protected:

  RegisterTest() = default;
  virtual ~RegisterTest() = default;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};
TYPED_TEST_CASE_P(RegisterTest);


/*
 * We are using drand48() for input values so the compiler cannot do fancy
 * things, like constexpr out all of the intrinsics.
 */

TYPED_TEST_P(RegisterTest, VectorRegisterSetGet)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    x.set(i, A[i]);
  }

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(x[i], A[i]);
  }

}


TYPED_TEST_P(RegisterTest, VectorRegisterLoad)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*2];
  for(size_t i = 0;i < num_elem*2; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
  }


  // load stride-1 from pointer
  register_t x;
  x.load(A);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(x[i], A[i]);
  }

  // load stride-2from pointer
  register_t y;
  y.load(A, 2);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(y[i], A[i*2]);
  }
}



TYPED_TEST_P(RegisterTest, VectorRegisterAdd)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x+y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z[i], A[i] + B[i]);
  }

  register_t z2 = x;
  z2 += y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z2[i], A[i] + B[i]);
  }

}

TYPED_TEST_P(RegisterTest, VectorRegisterSubtract)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x-y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z[i], A[i] - B[i]);
  }

  register_t z2 = x;
  z2 -= y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z2[i], A[i] - B[i]);
  }
}

TYPED_TEST_P(RegisterTest, VectorRegisterMultiply)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x*y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z[i], A[i] * B[i]);
  }

  register_t z2 = x;
  z2 *= y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z2[i], A[i] * B[i]);
  }
}

TYPED_TEST_P(RegisterTest, VectorRegisterDivide)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0)+1.0;
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x/y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z[i], A[i] / B[i]);
  }

  register_t z2 = x;
  z2 /= y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z2[i], A[i] / B[i]);
  }
}

TYPED_TEST_P(RegisterTest, VectorRegisterDotProduct)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  element_t expected = 0.0;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
    expected += A[i]*B[i];
  }

  ASSERT_DOUBLE_EQ(x.dot(y), expected);

}

TYPED_TEST_P(RegisterTest, VectorRegisterMax)
{
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    x.set(i, A[i]);
  }

  element_t expected = A[0];
  for(size_t i = 1;i < num_elem;++ i){
    expected = expected > A[i] ? expected : A[i];
  }

  ASSERT_DOUBLE_EQ(x.max(), expected);

}

TYPED_TEST_P(RegisterTest, VectorRegisterMin)
{
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    x.set(i, A[i]);
  }

  element_t expected = A[0];
  for(size_t i = 1;i < num_elem;++ i){
    expected = expected < A[i] ? expected : A[i];
  }

  ASSERT_DOUBLE_EQ(x.min(), expected);

}


REGISTER_TYPED_TEST_CASE_P(RegisterTest, VectorRegisterSetGet,
                                       VectorRegisterLoad,
                                       VectorRegisterAdd,
                                       VectorRegisterSubtract,
                                       VectorRegisterMultiply,
                                       VectorRegisterDivide,
                                       VectorRegisterDotProduct,
                                       VectorRegisterMax,
                                       VectorRegisterMin);

INSTANTIATE_TYPED_TEST_CASE_P(SIMD, RegisterTest, RegisterTestTypes);



