//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic view operations
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

TEST(ViewTest, Const)
{
  using layout = RAJA::Layout<1>;

  double data[10];
  RAJA::View<double, layout> view(data, layout(10));

  /*
   * Should be able to construct a non-const View from a non-const View
   */
  RAJA::View<double, layout> view2(view);

  /*
   * Should be able to construct a const View from a non-const View
   */
  RAJA::View<double const, layout> const_view(view);

  /*
   * Should be able to construct a const View from a const View
   */
  RAJA::View<double const, layout> const_view2(const_view);
}
