//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

RAJA_INDEX_VALUE(TX, "TX");
RAJA_INDEX_VALUE(TIX, "TIX");
RAJA_INDEX_VALUE(TIY, "TIY");
RAJA_INDEX_VALUE(TIL, "TIL");

template<typename T>
class ViewTest : public ::testing::Test {};

using allTypes = ::testing::Types<RAJA::Index_type,
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
                                  unsigned long long, float , double>;

TYPED_TEST_CASE(ViewTest, allTypes);

TYPED_TEST(ViewTest, Constructors)
{

  using layout = RAJA::Layout<1>;

  TypeParam data[10];
  RAJA::View<TypeParam, layout> view(data, layout(10));

  /*
   * Should be able to construct a non-const View from a non-const View
   */
  RAJA::View<TypeParam, layout> view2(view);

  /*
   * Should be able to construct a const View from a non-const View
   */
  RAJA::View<TypeParam const, layout> const_view(view);

  /*
   * Should be able to construct a const View from a const View
   */
  RAJA::View<TypeParam const, layout> const_view2(const_view);
}

TYPED_TEST(ViewTest, Shift1D)
{

  int N = 10;
  TypeParam *a = new TypeParam[N];
  TypeParam *b = new TypeParam[N];

  const int DIM = 1;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0}},{{N-1}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::View<TypeParam, RAJA::Layout<DIM>> B(a,N);
  RAJA::TypedView<TypeParam, RAJA::Layout<DIM>,TX> C(a,N);

  for(int i=0; i<N; ++i) {
    A(i) = i + 1;
  }

  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{N}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{N}});

  //TypedView
  RAJA::TypedView<TypeParam, RAJA::OffsetLayout<DIM>, TX> Cshift = C.shift({{N}});

  for(int i=N; i<2*N; ++i)
  {
    ASSERT_EQ(Ashift(i),A(i-N));
    ASSERT_EQ(Bshift(i),B(i-N));
  }

  for(TX tx=TX{N}; tx<TX{2*N}; tx++)
  {
    ASSERT_EQ(Cshift(tx),C(tx-N));
  }

  //TypedOffsetLayout + View
  using TLayout = RAJA::TypedLayout<TIL, RAJA::tuple<TIX>>;
  using TOffsetLayout = RAJA::TypedOffsetLayout<TIL, RAJA::tuple<TIX>>;

  TLayout myLayout(10);

  RAJA::View<TypeParam, TLayout> D(a, myLayout);
  RAJA::View<TypeParam, TOffsetLayout> Dshift = D.shift({{N}});

  for(TIX i=TIX{N}; i<TIX{2*N}; ++i)
  {
    ASSERT_EQ(Dshift(i),D(i-N));
  };

  delete[] a;
  delete[] b;

}


TYPED_TEST(ViewTest, Shift2D)
{

  int N = 10;
  TypeParam *a = new TypeParam[N*N];
  TypeParam *b = new TypeParam[N*N];

  const int DIM = 2;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0,0}},{{N-1,N-1}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::View<TypeParam, RAJA::Layout<DIM>> B(a,N,N);

  for(int y=0; y<N; ++y) {
    for(int x=0; x<N; ++x) {
      A(y,x) = x + N*y;
    }
  }

  //shift view
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{N,N}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{N,N}});

  for(int y=N; y<N+N; ++y) {
    for(int x=N; x<N+N; ++x) {
      ASSERT_EQ(Ashift(y,x),A(y-N,x-N));
      ASSERT_EQ(Bshift(y,x),B(y-N,x-N));
    }
  }

  //Permuted layout
  std::array< RAJA::idx_t, 2> perm {{1, 0}};
  RAJA::OffsetLayout<2> playout =
    RAJA::make_permuted_offset_layout<2>( {{0, 0}}, {{N-1, N-1}}, perm );

  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> C(a, playout);
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Cshift = C.shift({{N,N}});

  for(int y=N; y<N+N; ++y) {
    for(int x=N; x<N+N; ++x) {
      ASSERT_EQ(Cshift(y,x),C(y-N,x-N));
    }
  }

  delete[] a;
  delete[] b;
}
