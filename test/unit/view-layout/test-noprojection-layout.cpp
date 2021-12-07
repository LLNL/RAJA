//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"

/*
 * Basic layout test
 */

TEST(OffsetLayoutNoProjUnitTest, Constructors)
{
  using layout = RAJA::LayoutNoProj<1>;

  /*
   * Construct a 1D view with  with the following indices:
   *
   * 0, 1, 2, 3, 4
   */
  const layout l(5);

  /*
   * First element, 0, should have index 0.
   */
  ASSERT_EQ(0, l(0));

  ASSERT_EQ(2, l(2));

  /*
   * Last element, 4, should have index 4.
   */
  ASSERT_EQ(4, l(4));
}

TEST(LayoutNoProjUnitTest, 2D_IJ)
{
  using my_layout = RAJA::LayoutNoProj<2>;

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
  ASSERT_EQ(14, layout(2, 4));

  ASSERT_EQ(1, layout(0, 1));
  ASSERT_EQ(4, layout(0, 4));

  // Check that we get the identity
  for (int k = 0; k < 15; ++k) {

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

TEST(LayoutNoProjUnitTest, 2D_IJ_zero)
{
  using my_layout = RAJA::LayoutNoProj<2>;

  // Zero for J size should correctly produce size 0 layout
  const my_layout layout70(7, 0);

  ASSERT_EQ(0, layout70.size());

  // Zero for I size should correctly produce size 0 layout
  const my_layout layout07(0, 7);

  ASSERT_EQ(0, layout07.size());

  // Zero for I and J sizes should correctly produce size 0 layout
  const my_layout layout00(0, 0);

  ASSERT_EQ(0, layout00.size());
}

