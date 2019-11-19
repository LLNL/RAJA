//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for Operators.
///

#include "gtest/gtest.h"
#include "RAJA/RAJA.hpp"

template<typename T>
class OperatorsUnitTest : public ::testing::Test {};
template<typename T>
class OperatorsIntegralUnitTest : public ::testing::Test {};
template<typename T>
class OperatorsFloatingUnitTest : public ::testing::Test {};

using MyIntegralTypes = ::testing::Types<RAJA::Index_type,
                                         char,
                                         unsigned char,
                                         short,
                                         unsigned short,
                                         int,
                                         unsigned int,
                                         long,
                                         unsigned long,
                                         long int,
                                         unsigned long int,
                                         long long,
                                         unsigned long long>;

using MyFloatTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(OperatorsIntegralUnitTest, MyIntegralTypes);
TYPED_TEST_CASE(OperatorsFloatingUnitTest, MyFloatTypes);

template<typename T>
void plus_test()
{
  using Plus = RAJA::operators::plus<T>;
  auto ident = Plus::identity();
  ASSERT_EQ(ident, 0);

  Plus p;
  T i = static_cast<T>(1);
  T j = static_cast<T>(2);
  ASSERT_EQ(p(i,j), 3);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(p(i,j), -7);
  }
}

template<typename T>
void minus_test()
{
  using Minus = RAJA::operators::minus<T>;

  Minus m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), 3);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), -3);
  }
}

template<typename T>
void multiplies_test()
{
  using Mult = RAJA::operators::multiplies<T>;
  auto ident = Mult::identity();
  ASSERT_EQ(ident, 1);

  Mult m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), 10);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), 10);
  }
}

template<typename T>
void divides_test()
{
  using Div = RAJA::operators::divides<T>;

  Div d;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  if(std::is_floating_point<T>::value) 
    ASSERT_EQ(d(i,j), 2.5);
  else
    ASSERT_EQ(d(i,j), 2);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    if(std::is_floating_point<T>::value) 
      ASSERT_EQ(d(i,j), 2.5);
    else
      ASSERT_EQ(d(i,j), 2);
  }
}

template<typename T>
void modulus_test()
{
  using Mod = RAJA::operators::modulus<T>;

  Mod m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), 1);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), -1);
  }
}

template<typename T>
void logical_and_test()
{
  using And = RAJA::operators::logical_and<T>;

  And a;
  T i0 = static_cast<T>(0);
  T i1 = static_cast<T>(1);
  T i2 = static_cast<T>(2);
  T j0 = static_cast<T>(0);
  T j1 = static_cast<T>(1);
  T j2 = static_cast<T>(2);
  ASSERT_FALSE(a(i0,j0));
  ASSERT_FALSE(a(i0,j1));
  ASSERT_FALSE(a(i1,j0));
  ASSERT_TRUE(a(i1,j1));
  ASSERT_TRUE(a(i2,j2));
  if (std::is_signed<T>::value) {
    i1 = static_cast<T>(-1);
    j1 = static_cast<T>(-1);
    ASSERT_FALSE(a(i0,j1));
    ASSERT_FALSE(a(i1,j0));
    ASSERT_TRUE(a(i1,j1));
  }
}

template<typename T>
void logical_or_test()
{
  using Or = RAJA::operators::logical_or<T>;

  Or o;
  T i0 = static_cast<T>(0);
  T i1 = static_cast<T>(1);
  T i2 = static_cast<T>(2);
  T j0 = static_cast<T>(0);
  T j1 = static_cast<T>(1);
  T j2 = static_cast<T>(2);
  ASSERT_FALSE(o(i0,j0));
  ASSERT_TRUE(o(i0,j1));
  ASSERT_TRUE(o(i1,j0));
  ASSERT_TRUE(o(i1,j1));
  ASSERT_TRUE(o(i2,j2));
  if (std::is_signed<T>::value) {
    i1 = static_cast<T>(-1);
    j1 = static_cast<T>(-1);
    ASSERT_TRUE(o(i0,j1));
    ASSERT_TRUE(o(i1,j0));
    ASSERT_TRUE(o(i1,j1));
  }
}

template<typename T>
void logical_not_test()
{
  using Not = RAJA::operators::logical_not<T>;

  Not n;
  T i0 = static_cast<T>(0);
  T i1 = static_cast<T>(1);
  ASSERT_FALSE(n(i1));
  ASSERT_TRUE(n(i0));
  if (std::is_signed<T>::value) {
    i1 = static_cast<T>(-1);
    ASSERT_FALSE(n(i1));
  }
}

template<typename T>
void bit_or_test()
{
  using Or = RAJA::operators::bit_or<T>;

  Or o;
  T i = static_cast<T>(0010);
  T j = static_cast<T>(0001);
  T k = static_cast<T>(0111);
  ASSERT_EQ(o(i,j), 0011);
  ASSERT_EQ(o(i,k), 0111);
  ASSERT_EQ(o(j,k), 0111);
}

template<typename T>
void bit_and_test()
{
  using And = RAJA::operators::bit_and<T>;

  And a;
  T i = static_cast<T>(0010);
  T j = static_cast<T>(0001);
  T k = static_cast<T>(0111);
  ASSERT_EQ(a(i,j), 0000);
  ASSERT_EQ(a(i,k), 0010);
  ASSERT_EQ(a(j,k), 0001);
}

template<typename T>
void bit_xor_test()
{
  using Xor = RAJA::operators::bit_xor<T>;

  Xor x;
  T i = static_cast<T>(0010);
  T j = static_cast<T>(0001);
  T k = static_cast<T>(0111);
  ASSERT_EQ(x(i,j), 0011);
  ASSERT_EQ(x(i,k), 0101);
  ASSERT_EQ(x(j,k), 0110);
}

template<typename T>
void maximum_test()
{
  using Max = RAJA::operators::maximum<T>;

  Max m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), i);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), j);
  }
}


template<typename T>
void minimum_test()
{
  using Min = RAJA::operators::minimum<T>;

  Min m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), j);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), i);
  }
}


template<typename T>
void equal_test()
{
  using Eq = RAJA::operators::equal_to<T>;

  Eq eq;
  T i = static_cast<T>(5);
  T j = static_cast<T>(5);
  ASSERT_TRUE(eq(i,j));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-5);
    ASSERT_TRUE(eq(i,j));
  }
}

template<typename T>
void not_equal_test()
{
  using NEq = RAJA::operators::not_equal_to<T>;

  NEq neq;
  T i = static_cast<T>(5);
  T j = static_cast<T>(3);
  ASSERT_TRUE(neq(i,j));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-3);
    ASSERT_TRUE(neq(i,j));
  }
}

template<typename T>
void greater_test()
{
  using G = RAJA::operators::greater<T>;

  G g;
  T i = static_cast<T>(5);
  T j = static_cast<T>(4);
  ASSERT_TRUE(g(i,j));
  ASSERT_FALSE(g(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-4);
    j = static_cast<T>(-5);
    ASSERT_TRUE(g(i,j));
    ASSERT_FALSE(g(j,i));
  }
}

template<typename T>
void less_test()
{
  using L = RAJA::operators::less<T>;

  L l;
  T i = static_cast<T>(4);
  T j = static_cast<T>(5);
  ASSERT_TRUE(l(i,j));
  ASSERT_FALSE(l(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-4);
    ASSERT_TRUE(l(i,j));
    ASSERT_FALSE(l(j,i));
  }
}

template<typename T>
void greater_eq_test()
{
  using G = RAJA::operators::greater_equal<T>;

  G g;
  T i = static_cast<T>(5);
  T i2 = static_cast<T>(5);
  T j = static_cast<T>(4);
  ASSERT_TRUE(g(i,j));
  ASSERT_FALSE(g(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-4);
    i2 = static_cast<T>(-4);
    j = static_cast<T>(-5);
    ASSERT_TRUE(g(i,j));
    ASSERT_FALSE(g(j,i));
  }
}

template<typename T>
void less_eq_test()
{
  using L = RAJA::operators::less_equal<T>;

  L l;
  T i = static_cast<T>(4);
  T i2 = static_cast<T>(4);
  T j = static_cast<T>(5);
  ASSERT_TRUE(l(i,j));
  ASSERT_FALSE(l(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    i2 = static_cast<T>(-5);
    j = static_cast<T>(-4);
    ASSERT_TRUE(l(i,j));
    ASSERT_FALSE(l(j,i));
  }
}

TYPED_TEST(OperatorsIntegralUnitTest, plus) { plus_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, plus) { plus_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, minus) { minus_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, minus) { minus_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, multiplies) { multiplies_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, multiplies) { multiplies_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, divides) { divides_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, divides) { divides_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, modulus) { modulus_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, logical_and) { logical_and_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, logical_and) { logical_and_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, logical_or) { logical_or_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, logical_or) { logical_or_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, logical_not) { logical_not_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, logical_not) { logical_not_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, bit_or) { bit_or_test<TypeParam>(); }
TYPED_TEST(OperatorsIntegralUnitTest, bit_and) { bit_and_test<TypeParam>(); }
TYPED_TEST(OperatorsIntegralUnitTest, bit_xor) { bit_xor_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, minimum) { minimum_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, minimum) { minimum_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, maximum) { maximum_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, maximum) { maximum_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, equal_to) { equal_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, equal_to) { equal_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, not_equal_to) { not_equal_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, not_equal_to) { not_equal_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, greater) { greater_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, greater) { greater_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, less) { less_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, less) { less_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, greater_eq) { greater_eq_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, greater_eq) { greater_eq_test<TypeParam>(); }

TYPED_TEST(OperatorsIntegralUnitTest, less_eq) { less_eq_test<TypeParam>(); }
TYPED_TEST(OperatorsFloatingUnitTest, less_eq) { less_eq_test<TypeParam>(); }
