//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic vector operations
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "RAJA/pattern/vector.hpp"



using TestTypes = ::testing::Types<RAJA::SimdRegister<double, 1>,
                                   RAJA::SimdRegister<double, 4>>;



template <typename NestedPolicy>
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


/*
 * We are using drand48() for input values so the compiler cannot do fancy
 * things, like constexpr out all of the intrinsics.
 */

TYPED_TEST_P(VectorTest, SimdRegisterSetGet)
{

  using register_t = TypeParam;

  static constexpr size_t num_elem = register_t::s_num_elem;

  double A[num_elem];
  register_t x;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = drand48();
    x.set(i, A[i]);
  }

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_EQ(x[i], A[i]);
  }

}




TYPED_TEST_P(VectorTest, SimdRegisterAdd)
{

  using register_t = TypeParam;

  static constexpr size_t num_elem = register_t::s_num_elem;

  double A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = drand48();
    B[i] = drand48();
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

TYPED_TEST_P(VectorTest, SimdRegisterSubtract)
{

  using register_t = TypeParam;

  static constexpr size_t num_elem = register_t::s_num_elem;

  double A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = drand48();
    B[i] = drand48();
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

TYPED_TEST_P(VectorTest, SimdRegisterMultiply)
{

  using register_t = TypeParam;

  static constexpr size_t num_elem = register_t::s_num_elem;

  double A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = drand48();
    B[i] = drand48();
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

TYPED_TEST_P(VectorTest, SimdRegisterDivide)
{

  using register_t = TypeParam;

  static constexpr size_t num_elem = register_t::s_num_elem;

  double A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = drand48();
    B[i] = drand48()+1.0;
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

TYPED_TEST_P(VectorTest, SimdRegisterDotProduct)
{

  using register_t = TypeParam;

  static constexpr size_t num_elem = register_t::s_num_elem;

  double A[num_elem], B[num_elem];
  register_t x, y;

  double expected = 0.0;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = drand48();
    B[i] = drand48();
    x.set(i, A[i]);
    y.set(i, B[i]);
    expected += A[i]*B[i];
  }

  ASSERT_DOUBLE_EQ(x.dot(y), expected);

}


REGISTER_TYPED_TEST_CASE_P(VectorTest, SimdRegisterSetGet,
                                       SimdRegisterAdd,
                                       SimdRegisterSubtract,
                                       SimdRegisterMultiply,
                                       SimdRegisterDivide,
                                       SimdRegisterDotProduct);

INSTANTIATE_TYPED_TEST_CASE_P(SIMD, VectorTest, TestTypes);
