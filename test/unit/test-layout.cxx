#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"

TEST(LayoutTest, 1D)
{
  using layout = RAJA::OffsetLayout<int, RAJA::PERM_I, int>;

  /*
   * Construct a 1D view with  with the following indices:
   *
   * 10, 11, 12, 13, 14
   */
  const layout l({10}, {14});

  /*
   * First element, 10, should have index 0.
   */
  ASSERT_EQ(0, l(10));

  /*
   * Last element, 14, should have index 5.
   */
  ASSERT_EQ(4, l(14));
}

TEST(LayoutTest, 2D)
{
  typedef RAJA::OffsetLayout<int, RAJA::PERM_IJ, int, int> my_layout;

  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const my_layout layout({-1,-2}, {1,0});

  /*
   * First element, (-1, -2), should have index 0.
   */
  ASSERT_EQ(0, layout(-1, -2));

  /*
   * Last element, (1, 0), should have index 8.
   */
  ASSERT_EQ(8, layout(1,0));
}
