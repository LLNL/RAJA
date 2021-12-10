//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

template<typename T>
class TypedLayoutNoProjUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(TypedLayoutNoProjUnitTest, UnitIndexTypes);


TYPED_TEST(TypedLayoutNoProjUnitTest, TypedLayoutNoProjConstructors)
{

  const RAJA::TypedLayoutNoProj<TypeParam, RAJA::tuple<TypeParam, TypeParam>> l(10,5);

  ASSERT_EQ(TypeParam{0}, l(TypeParam{0}, TypeParam{0}));

  ASSERT_EQ(TypeParam{2}, l(TypeParam{0}, TypeParam{2}));

  ASSERT_EQ(TypeParam{10}, l(TypeParam{2}, TypeParam{0}));

  TypeParam x{5};
  TypeParam y{0};
  l.toIndices(TypeParam{10}, y, x);
  ASSERT_EQ(x, TypeParam{0});
  ASSERT_EQ(y, TypeParam{2});
}

TYPED_TEST(TypedLayoutNoProjUnitTest, 2D_accessor)
{
  using my_layout = RAJA::TypedLayoutNoProj<TypeParam, RAJA::tuple<TypeParam, TypeParam>>;

  /*
   * Construct a 2D layout:
   *
   * I is stride 5
   * J is stride 1
   *
   * Linear indices range from [0, 15)
   *
   */

  // Construct using variadic "sizes" ctor
  const my_layout layout_a(3, 5);

  // Construct using copy ctor
  const my_layout layout_b(layout_a);

  // Test default ctor and assignment operator
  my_layout layout;
  layout = layout_b;


  ASSERT_EQ(TypeParam(0), layout(0, 0));

  ASSERT_EQ(TypeParam(5), layout(1, 0));
  ASSERT_EQ(TypeParam(14), layout(2, 4));

  ASSERT_EQ(TypeParam(1), layout(0, 1));
  ASSERT_EQ(TypeParam(4), layout(0, 4));

  // Check that we get the identity
  for (int k = 0; k < 15; ++k) {

    // inverse map
    TypeParam i, j;
    layout.toIndices(k, i, j);

    // forward map
    TypeParam k2 = layout(i, j);

    // check ident
    ASSERT_EQ(k, k2);

    // check with a and b
    ASSERT_EQ(k2, layout_a(i, j));
    ASSERT_EQ(k2, layout_b(i, j));
  }

}

TYPED_TEST(TypedLayoutNoProjUnitTest, 2D_IJ_zero)
{
  using my_layout = RAJA::TypedLayoutNoProj<TypeParam, RAJA::tuple<TypeParam, TypeParam>>;

  // Zero for J size should correctly produce size 0 layout
  const my_layout layout70(7, 0);

  ASSERT_EQ(TypeParam(0), layout70.size());

  // Zero for I size should correctly produce size 0 layout
  const my_layout layout07(0, 7);

  ASSERT_EQ(TypeParam(0), layout07.size());

  // Zero for I and J sizes should correctly produce size 0 layout
  const my_layout layout00(0, 0);

  ASSERT_EQ(TypeParam(0), layout00.size());
}
