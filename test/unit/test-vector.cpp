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

#if 0

using RegisterTestTypes = ::testing::Types<
                                   RAJA::Register<RAJA::simd_register, int, 1>,
                                   RAJA::Register<RAJA::simd_register, float, 1>,
                                   RAJA::Register<RAJA::simd_register, double, 1>,
                                   RAJA::Register<RAJA::simd_register, double, 2>,
                                   RAJA::Register<RAJA::simd_register, double, 3>,
                                   RAJA::Register<RAJA::simd_register, double, 4>,
                                   RAJA::FixedVector<RAJA::Register<RAJA::simd_register, double,1>, 27>,
                                   RAJA::FixedVector<RAJA::Register<RAJA::simd_register, double,2>, 27>,
                                   RAJA::FixedVector<RAJA::Register<RAJA::simd_register, double,3>, 27>,
                                   RAJA::FixedVector<RAJA::Register<RAJA::simd_register, double,4>, 27>,
                                   RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 4>,
                                   RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 8>,
                                   RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 12>,
                                   RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 16>>;

//usingRegister TestTypes = ::testing::Types<RAJA::FixedVector<RAJA::SimdRegister<double,4>, 27>>;

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

TYPED_TEST_P(RegisterTest, SimdRegisterSetGet)
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


TYPED_TEST_P(RegisterTest, SimdRegisterLoad)
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



TYPED_TEST_P(RegisterTest, SimdRegisterAdd)
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

TYPED_TEST_P(RegisterTest, SimdRegisterSubtract)
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

TYPED_TEST_P(RegisterTest, SimdRegisterMultiply)
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

TYPED_TEST_P(RegisterTest, SimdRegisterDivide)
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

TYPED_TEST_P(RegisterTest, SimdRegisterDotProduct)
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

TYPED_TEST_P(RegisterTest, SimdRegisterMax)
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

TYPED_TEST_P(RegisterTest, SimdRegisterMin)
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


REGISTER_TYPED_TEST_CASE_P(RegisterTest, SimdRegisterSetGet,
                                       SimdRegisterLoad,
                                       SimdRegisterAdd,
                                       SimdRegisterSubtract,
                                       SimdRegisterMultiply,
                                       SimdRegisterDivide,
                                       SimdRegisterDotProduct,
                                       SimdRegisterMax,
                                       SimdRegisterMin);

INSTANTIATE_TYPED_TEST_CASE_P(SIMD, RegisterTest, RegisterTestTypes);




TEST(StreamVectorTest, Test1)
{
  using TypeParam = RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 8>;
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
  }


  for(size_t N = 1;N <= 8;++ N){
    for(size_t i = 0;i < 8;++ i){
      B[i] = 0;
    }

    register_t x, y;
    x.load_n(A, N);
    y = 3.0;
    x = x+y;
    x.store(B);

    for(size_t i = 0;i < 8;++ i){
      if(i < N){
        ASSERT_DOUBLE_EQ(B[i], A[i]+3.0);
      }
      else
      {
        ASSERT_DOUBLE_EQ(B[i], 0.0);
      }
    }
  }
}

TEST(StreamVectorTest, TestStreamLoop)
{
  using TypeParam = RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 32>;
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  // Use drand48 to change sizes of everything: this ensures that the compiler
  // cannot optimize out sizes (and do more optimization than we want)
  size_t N = 8000 + (100*drand48());

  element_t *A = new element_t[N];
  element_t *B = new element_t[N];
  element_t *C = new element_t[N];
  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
  }

  size_t Nsimd = N - (N%num_elem);
  size_t Nrem = N - Nsimd;
  for(size_t i = 0;i < Nsimd;i += num_elem){
    register_t x,y;
    x.load_n(&A[i],num_elem);
    y.load_n(&B[i],num_elem);

    register_t z = x*y;
    z.store(&C[i]);
  }
  if(Nrem > 0){
    register_t x,y;
    x.load_n(&A[Nsimd], Nrem);
    y.load_n(&B[Nsimd], Nrem);

    register_t z = x*y;
    z.store(&C[Nsimd]);
  }

  for(size_t i = 0;i < N;i ++){
    ASSERT_DOUBLE_EQ(A[i]*B[i], C[i]);
  }

  delete[] A;
  delete[] B;
  delete[] C;
}

TEST(StreamVectorTest, TestFixedForall)
{
  using TypeParam = RAJA::FixedVector<RAJA::Register<RAJA::simd_register, double,4>, 8>;
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;


  size_t N = 1024*num_elem;

  element_t *A = new element_t[N];
  element_t *B = new element_t[N];
  element_t *C = new element_t[N];
  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
  }

  using policy_t = RAJA::simd_fixed_exec<register_t>;

  RAJA::forall<policy_t>(RAJA::TypedRangeSegment<size_t>(0, N),
      [=](RAJA::FixedRegisterIndex<size_t, register_t> i)
  {
    register_t x,y;
    x.load(&A[*i]);
    y.load(&B[*i]);

    register_t z = x*y;
    z.store(&C[*i]);
  });


  for(size_t i = 0;i < N;i ++){
    ASSERT_DOUBLE_EQ(A[i]*B[i], C[i]);
  }

  delete[] A;
  delete[] B;
  delete[] C;
}

TEST(StreamVectorTest, TestStreamForall)
{
  using TypeParam = RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 8>;
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;


  size_t N = 8000 + (100*drand48());

  element_t *A = new element_t[N];
  element_t *B = new element_t[N];
  element_t *C = new element_t[N];
  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(drand48()*1000.0);
    B[i] = (element_t)(drand48()*1000.0);
  }

  using policy_t = RAJA::simd_stream_exec<register_t>;

  RAJA::forall<policy_t>(RAJA::TypedRangeSegment<size_t>(0, N),
      [=](RAJA::StreamRegisterIndex<size_t, register_t> i)
  {
    register_t x,y;
    x.load_n(&A[*i], i.size());
    y.load_n(&B[*i], i.size());

    register_t z = x*y;
    z.store(&C[*i]);
  });


  for(size_t i = 0;i < N;i ++){
    ASSERT_DOUBLE_EQ(A[i]*B[i], C[i]);
  }

  delete[] A;
  delete[] B;
  delete[] C;
}
#endif

TEST(StreamVectorTest, TestStreamForallRef)
{
  using TypeParam = RAJA::StreamVector<RAJA::Register<RAJA::simd_register, double,4>, 8>;
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;


  size_t N = 8000 + (100*drand48());

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
