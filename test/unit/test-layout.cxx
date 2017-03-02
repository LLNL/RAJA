#include "gtest/gtest.h"
#include "RAJA/RAJA.hxx"

RAJA_INDEX_VALUE(TestIndex1D, "TestIndex1D");

TEST(LayoutTest, 1D)
{
  using layout = RAJA::OffsetLayout<1>;

  /*
   * Construct a 1D view with  with the following indices:
   *
   * 10, 11, 12, 13, 14
   */
  const layout l({10}, std::array<int, 1>{14});

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

TEST(LayoutTest, OffsetVsRegular)
{
  const RAJA::Layout<2> layout = RAJA::make_permuted_layout<RAJA::PERM_JI>(6,6);
  const RAJA::OffsetLayout<2> offset({0,0}, layout);

  /*
   * OffsetLayout with 0 offset should function like the regular Layout.
   */
  for (int j = 0; j < 6; ++j) {
    for (int i = 0; i < 6; ++i) {
      ASSERT_EQ(offset(i,j), layout(i,j));
    }
  }
}

TEST(LayoutTest, 2D_IJ)
{
  typedef RAJA::OffsetLayout<2> my_layout;

  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const my_layout layout({-1,-2}, std::array<int, 2>{1,0});

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
  ASSERT_EQ(8, layout(1,0));
}

TEST(LayoutTest, 2D_JI)
{
  typedef RAJA::OffsetLayout<2> my_layout;

  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const my_layout layout({-1,-2}, RAJA::make_permuted_layout<RAJA::PERM_JI>(3,3));

  /*
   * First element, (-1, -2), should have index 0.
   * things.
   */
  ASSERT_EQ(0, layout(-1, -2));

  ASSERT_EQ(1, layout(-0, -2));

  /*
   * Last element, (1, 0), should have index 8.
   */
  ASSERT_EQ(8, layout(1,0));
}

TEST(LayoutTest, View)
{
  int* data = new int[10];

  using layout = RAJA::OffsetLayout<>;

  /*
   * View is constructed by passing in the layout.
   */
  RAJA::View<int, layout> view(data, layout({1}, std::array<int, 1>{10}));

  for (int i = 0; i < 10; i++) {
    data[i] = i;
  }

  ASSERT_EQ(data[0], view(1));
  ASSERT_EQ(data[9], view(10));
}
