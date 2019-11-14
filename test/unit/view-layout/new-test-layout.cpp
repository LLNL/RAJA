//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

template<typename T>
class LayoutTest : public ::testing::Test {};

template<typename T>
class OffsetLayoutTest : public ::testing::Test {};

using MyTypes = ::testing::Types<RAJA::Index_type,
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


TYPED_TEST_CASE(LayoutTest, MyTypes);
TYPED_TEST_CASE(OffsetLayoutTest, MyTypes);

TYPED_TEST(LayoutTest, Constructors)
{

  const RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>> l(10,5);

  ASSERT_EQ(TypeParam{0}, l(TypeParam{0}, TypeParam{0}));

  ASSERT_EQ(TypeParam{2}, l(TypeParam{0}, TypeParam{2}));

  ASSERT_EQ(TypeParam{10}, l(TypeParam{2}, TypeParam{0}));

  TypeParam x{5};
  TypeParam y{0};
  l.toIndices(TypeParam{10}, y, x);
  ASSERT_EQ(x, TypeParam{0});
  ASSERT_EQ(y, TypeParam{2});
}

TYPED_TEST(OffsetLayoutTest, OffsetVsRegular)
{

  const RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>> layout =
    RAJA::make_permuted_layout({{6, 6}},
                               RAJA::as_array<RAJA::Perm<1, 0>>::get());
  //  const auto offset =
  const RAJA::TypedOffsetLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>> offset =
      RAJA::make_permuted_offset_layout({{0, 0}},
                                        {{5, 5}},
                                        RAJA::as_array<RAJA::PERM_JI>::get());
  /*
   * OffsetLayout with 0 offset should function like the regular Layout.
   */
  for (TypeParam j = 0; j < 6; ++j) {
    for (TypeParam i = 0; i < 6; ++i) {
      ASSERT_EQ(offset(i, j), layout(i, j))
          << layout.strides[0] << layout.strides[1];
    }
  }
}


TYPED_TEST(OffsetLayoutTest, 2D_IJ)
{
  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const RAJA::TypedOffsetLayout<TypeParam, RAJA::tuple<TypeParam,TypeParam>> layout =
    RAJA::make_offset_layout<2>({{-1, -2}}, {{1, 0}});

  /*
   * First element, (-1, -2), should have index 0.
   */
  ASSERT_EQ(0, layout(-1, -2));

  /*
   * (0, -2) should have index 3.
   */
  ASSERT_EQ(3, layout(0, -2));

  /*
   * Last element, (1, 0), should have index 8.
   */
  ASSERT_EQ(8, layout(1, 0));
}


TYPED_TEST(LayoutTest, 2D_IJ)
{
  using my_layout = RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>>;

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


  ASSERT_EQ(0, layout(0, 0));

  ASSERT_EQ(5, layout(1, 0));
  ASSERT_EQ(15, layout(3, 0));

  ASSERT_EQ(1, layout(0, 1));
  ASSERT_EQ(5, layout(0, 5));

  // Check that we get the identity (mod 15)
  TypeParam pK = 0;
  for (int k = 0; k < 20; ++k) {

    // inverse map
    TypeParam i, j;
    layout.toIndices(pK, i, j);

    // forward map
    TypeParam k2 = layout(i, j);

    // check ident
    ASSERT_EQ(pK % 15, k2);

    // check with a and b
    ASSERT_EQ(k2, layout_a(i, j));
    ASSERT_EQ(k2, layout_b(i, j));
    pK++;
  }

}


TYPED_TEST(LayoutTest, 2D_JI)
{
  using my_layout = RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>>;

  /*
   * Construct a 2D layout:
   *
   * I is stride 1
   * J is stride 3
   *
   * Linear indices range from [0, 15)
   *
   */
  const my_layout layout =
      RAJA::make_permuted_layout({{3, 5}},
                                 RAJA::as_array<RAJA::PERM_JI>::get());

  ASSERT_EQ(0, layout(0, 0));

  ASSERT_EQ(1, layout(1, 0));
  ASSERT_EQ(3, layout(3, 0));

  ASSERT_EQ(3, layout(0, 1));
  ASSERT_EQ(15, layout(0, 5));

  // Check that we get the identity (mod 15)
  TypeParam pK = 0;
  for (int k = 0; k < 20; ++k) {

    // inverse map
    TypeParam i, j;
    layout.toIndices(k, i, j);

    // forward map
    TypeParam k2 = layout(i, j);

    ASSERT_EQ(k % 15, k2);
    pK++;
  }
}

TYPED_TEST(LayoutTest, 2D_IJ_ProjJ)
{
  using my_layout = RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>>;

  /*
   * Construct a 2D projective layout:
   *
   * I is stride 1
   * J is stride 0  -  projected out
   *
   * Linear indices range from [0, 7)
   *
   * values of J should have no effect on linear index
   *
   * All linear indices should produce J=0
   *
   */

  // Construct using variadic "sizes" ctor
  // Zero for J size should correctly produce projective layout
  const my_layout layout(7, 0);

  ASSERT_EQ(0, layout(0, 0));

  ASSERT_EQ(1, layout(1, 0));
  ASSERT_EQ(3, layout(3, 0));

  // J should be projected out
  ASSERT_EQ(0, layout(0, 1));
  ASSERT_EQ(0, layout(0, 5));

  TypeParam pK = 0;
  // Check that we get the identity (mod 7)
  for (int k = 0; k < 20; ++k) {

    // inverse map
    TypeParam i, j;
    layout.toIndices(pK, i, j);

    // forward map
    TypeParam k2 = layout(i, j);

    // check ident
    ASSERT_EQ(pK % 7, k2);

    // check projection of j
    ASSERT_EQ(j, 0);
    pK++;
  }
}

TYPED_TEST(LayoutTest, 3D_KJI_ProjJ)
{
  //using my_layout = RAJA::Layout<3>;
  using my_layout = RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam, TypeParam>>;

  /*
   * Construct a 3D projective layout:
   *
   * I is stride 1
   * J is stride 0  -  projected out
   * K is stride 3
   *
   * Linear indices range from [0, 21)
   *
   * values of J should have no effect on linear index
   *
   * All linear indices should produce J=0
   *
   */

  // Construct using variadic "sizes" ctor
  // Zero for J size should correctly produce projective layout
  const my_layout layout =
      RAJA::make_permuted_layout({{3, 0, 7}},
                                 RAJA::as_array<RAJA::PERM_KJI>::get());

  ASSERT_EQ(0, layout(0, 0, 0));

  ASSERT_EQ(1, layout(1, 0, 0));
  ASSERT_EQ(3, layout(3, 0, 0));

  // J should be projected out
  ASSERT_EQ(0, layout(0, 1, 0));
  ASSERT_EQ(0, layout(0, 5, 0));

  ASSERT_EQ(6, layout(0, 0, 2));
  ASSERT_EQ(12, layout(0, 0, 4));

  // Check that we get the identity (mod 21)
  for (TypeParam x = 0; x < 40; ++x) {

    // inverse map
    TypeParam i, j, k;
    layout.toIndices(x, i, j, k);

    // forward map
    TypeParam x2 = layout(i, j, k);

    // check ident
    ASSERT_EQ(x % 21, x2);

    // check projection of j
    ASSERT_EQ(j, 0);
  }
}

TEST(LayoutTest, 2D_StrideOne)
{
  using my_layout = RAJA::Layout<2>;
  using my_layout_s1 = RAJA::Layout<2, ptrdiff_t, 0>; // first index is stride-1

  /*
   * Construct a 2D layout:
   *
   * I is stride 1
   * J is stride 3
   *
   * Linear indices range from [0, 15)
   *
   */
  const my_layout layout =
      RAJA::make_permuted_layout({{3, 5}},
                                 RAJA::as_array<RAJA::PERM_JI>::get());


  /*
   * Construct another 2D layout that forces J to be stride-1
   */
  const my_layout_s1 layout_s1 = layout;


  // Check that we get the same layout
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 15; ++j) {

      ASSERT_EQ(layout(i, j), layout_s1(i, j));
    }
  }
}


TEST(LayoutTest, 2D_StaticLayout)
{
  RAJA::Layout<2> dynamic_layout(7, 5);
  using static_layout = RAJA::TypedStaticLayout<RAJA::PERM_IJ,RAJA::tuple<int,int>,7,5>;
  
  //Breaks here...
  static_layout::s_oper(0,0);
#if 0
  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 5; ++j) {

      ASSERT_EQ(dynamic_layout(i, j), static_layout::s_oper(i,j));
    }
  }
#endif
}

TEST(LayoutTest, 2D_PermutedStaticLayout)
{
  auto dynamic_layout =
    RAJA::make_permuted_layout({{7, 5}},
                               RAJA::as_array<RAJA::PERM_JI>::get());
  using static_layout = RAJA::StaticLayout<RAJA::PERM_JI, 7,5>;

  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(dynamic_layout(i, j), static_layout::s_oper(i,j));
    }
  }
}

TEST(LayoutTest, 3D_PermutedStaticLayout)
{
  auto dynamic_layout =
    RAJA::make_permuted_layout({{7, 13, 5}},
                               RAJA::as_array<RAJA::PERM_JKI>::get());
  using static_layout = RAJA::StaticLayout<RAJA::PERM_JKI, 7,13,5>;

  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 13; ++j) {
      for (int k = 0; k < 5; ++k) {
        ASSERT_EQ(dynamic_layout(i, j, k), static_layout::s_oper(i,j,k));
      }
    }
  }
}


TEST(LayoutTest, 4D_PermutedStaticLayout)
{
  auto dynamic_layout =
    RAJA::make_permuted_layout({{7, 13, 5, 17}},
                               RAJA::as_array<RAJA::PERM_LJKI>::get());
  using static_layout = RAJA::StaticLayout<RAJA::PERM_LJKI, 7,13,5,17>;

  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 13; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 5; ++l) {
          ASSERT_EQ(dynamic_layout(i, j, k, l), static_layout::s_oper(i,j,k,l));
        }
      }
    }
  }
}
