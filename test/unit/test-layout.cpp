//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic layout operations
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

RAJA_INDEX_VALUE(TestIndex1D, "TestIndex1D");

RAJA_INDEX_VALUE(TIX, "TIX");
RAJA_INDEX_VALUE(TIY, "TIY");
RAJA_INDEX_VALUE(TIL, "TIL");

TEST(OffsetLayoutTest, 1D)
{
  using layout = RAJA::OffsetLayout<1>;

  /*
   * Construct a 1D view with  with the following indices:
   *
   * 10, 11, 12, 13, 14
   */
  const layout l({{10}}, {{14}});

  /*
   * First element, 10, should have index 0.
   */
  ASSERT_EQ(0, l(10));

  ASSERT_EQ(2, l(12));

  /*
   * Last element, 14, should have index 5.
   */
  ASSERT_EQ(4, l(14));
}

TEST(TypedLayoutTest, 1D)
{
  /*
   * Construct a 2D view, 10x5
   */
  const RAJA::TypedLayout<TIL, RAJA::tuple<TIX, TIY>> l(10, 5);

  ASSERT_EQ(TIL{0}, l(TIX{0}, TIY{0}));

  ASSERT_EQ(TIL{2}, l(TIX{0}, TIY{2}));

  ASSERT_EQ(TIL{10}, l(TIX{2}, TIY{0}));

  TIX x{5};
  TIY y{0};
  l.toIndices(TIL{10}, y, x);
  ASSERT_EQ(x, TIX{0});
  ASSERT_EQ(y, TIY{2});
}


TEST(LayoutTest, OffsetVsRegular)
{
  const auto layout =
      RAJA::make_permuted_layout({{6, 6}},
                                 RAJA::as_array<RAJA::Perm<1, 0>>::get());
  const auto offset =
      RAJA::make_permuted_offset_layout({{0, 0}},
                                        {{5, 5}},
                                        RAJA::as_array<RAJA::PERM_JI>::get());

  /*
   * OffsetLayout with 0 offset should function like the regular Layout.
   */
  for (int j = 0; j < 6; ++j) {
    for (int i = 0; i < 6; ++i) {
      ASSERT_EQ(offset(i, j), layout(i, j))
          << layout.strides[0] << layout.strides[1];
    }
  }
}

TEST(OffsetLayoutTest, 2D_IJ)
{
  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const auto layout = RAJA::make_offset_layout<2>({{-1, -2}}, {{1, 0}});

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

TEST(OffsetLayoutTest, 2D_JI)
{
  typedef RAJA::OffsetLayout<2> my_layout;

  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const my_layout layout =
      RAJA::make_permuted_offset_layout({{-1, -2}},
                                        {{1, 0}},
                                        RAJA::as_array<RAJA::PERM_JI>::get());

  /*
   * First element, (-1, -2), should have index 0.
   * things.
   */
  ASSERT_EQ(0, layout(-1, -2));

  ASSERT_EQ(1, layout(-0, -2));

  /*
   * Last element, (1, 0), should have index 8.
   */
  ASSERT_EQ(8, layout(1, 0));
}

TEST(OffsetLayoutTest, View)
{
  int* data = new int[10];

  using layout = RAJA::OffsetLayout<>;

  /*
   * View is constructed by passing in the layout.
   */
  std::array<RAJA::Index_type, 1> lower{{1}};
  std::array<RAJA::Index_type, 1> upper{{10}};
  RAJA::View<int, layout> view(data, RAJA::make_offset_layout<1>(lower, upper));

  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }

  ASSERT_EQ(data[0], view(1));
  ASSERT_EQ(data[9], view(10));
}


TEST(LayoutTest, 2D_IJ)
{
  typedef RAJA::Layout<2> my_layout;

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
  for (int k = 0; k < 20; ++k) {

    // inverse map
    int i, j;
    layout.toIndices(k, i, j);

    // forward map
    int k2 = layout(i, j);

    // check ident
    ASSERT_EQ(k % 15, k2);

    // check with a and b
    ASSERT_EQ(k2, layout_a(i, j));
    ASSERT_EQ(k2, layout_b(i, j));
  }
}

TEST(LayoutTest, 2D_JI)
{
  typedef RAJA::Layout<2> my_layout;

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
      RAJA::make_permuted_layout({{3, 5}}, RAJA::as_array<RAJA::PERM_JI>::get());

  ASSERT_EQ(0, layout(0, 0));

  ASSERT_EQ(1, layout(1, 0));
  ASSERT_EQ(3, layout(3, 0));

  ASSERT_EQ(3, layout(0, 1));
  ASSERT_EQ(15, layout(0, 5));

  // Check that we get the identity (mod 15)
  for (int k = 0; k < 20; ++k) {

    // inverse map
    int i, j;
    layout.toIndices(k, i, j);

    // forward map
    int k2 = layout(i, j);

    ASSERT_EQ(k % 15, k2);
  }
}


TEST(LayoutTest, 2D_IJ_ProjJ)
{
  typedef RAJA::Layout<2> my_layout;

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

  // Check that we get the identity (mod 7)
  for (int k = 0; k < 20; ++k) {

    // inverse map
    int i, j;
    layout.toIndices(k, i, j);

    // forward map
    int k2 = layout(i, j);

    // check ident
    ASSERT_EQ(k % 7, k2);

    // check projection of j
    ASSERT_EQ(j, 0);
  }
}


TEST(LayoutTest, 3D_KJI_ProjJ)
{
  typedef RAJA::Layout<3> my_layout;

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
  for (int x = 0; x < 40; ++x) {

    // inverse map
    int i, j, k;
    layout.toIndices(x, i, j, k);

    // forward map
    int x2 = layout(i, j, k);

    // check ident
    ASSERT_EQ(x % 21, x2);

    // check projection of j
    ASSERT_EQ(j, 0);
  }
}


TEST(LayoutTest, 2D_StrideOne)
{
  typedef RAJA::Layout<2> my_layout;
  typedef RAJA::Layout<2, ptrdiff_t, 0> my_layout_s1; // first index is stride-1

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
      RAJA::make_permuted_layout({{3, 5}}, RAJA::as_array<RAJA::PERM_JI>::get());


  /*
   * Construct another 2D layout that forces J to be stride-1
   */
  const my_layout_s1 layout_s1 = layout;



  // Check that we get the same layout
  for (int i = 0;i < 3;++ i){
    for (int j = 0; j < 15; ++j) {

      ASSERT_EQ(layout(i,j), layout_s1(i,j));
    }
  }
}
