//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
