//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

TEST(ViewTest, Shift1D)
{

  int N = 10;
  int *a = new int[N];
  int *b = new int[N];

  const int DIM = 1;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0}},{{N-1}});
  RAJA::View<int, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::View<int, RAJA::Layout<DIM>> B(a,N);

  for(int i=0; i<N; ++i) {
    A(i) = i + 1;
  }

  //shift view
  RAJA::View<int, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{10}},{{2*N-1}});
  RAJA::View<int, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{10}},{{2*N-1}});
  for(int i=N; i<2*N; ++i) {
    ASSERT_EQ(Ashift(i),A(i-N));
    ASSERT_EQ(Bshift(i),B(i-N));
  }

  delete[] a;
  delete[] b;
}

TEST(ViewTest, Shift2D)
{

  int N = 10;
  int *a = new int[N*N];
  int *b = new int[N*N];

  const int DIM = 2;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0,0}},{{N-1,N-1}});
  RAJA::View<int, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::View<int, RAJA::Layout<DIM>> B(a,N,N);

  for(int y=0; y<N; ++y) {
    for(int x=0; x<N; ++x) {
      A(y,x) = x + N*y;
    }
  }

  //shift view
  RAJA::View<int, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{N,N}},{{2*N-1,2*N-1}});
  RAJA::View<int, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{N,N}},{{2*N-1,2*N-1}});

  for(int y=N; y<N+N; ++y) {
    for(int x=N; x<N+N; ++x) {
      ASSERT_EQ(Ashift(y,x),A(y-N,x-N));
      ASSERT_EQ(Bshift(y,x),B(y-N,x-N));
    }
  }

  delete[] a;
  delete[] b;
}
