//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

RAJA_INDEX_VALUE(TX, "TX");
RAJA_INDEX_VALUE(TIX, "TIX");
RAJA_INDEX_VALUE(TIY, "TIY");
RAJA_INDEX_VALUE(TIL, "TIL");

template<typename T>
class TypedViewUnitTest : public ::testing::Test {};

template<typename T>
class OffsetLayoutViewUnitTest : public ::testing::Test {};

template<typename T>
class TypedIntegralViewUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(TypedViewUnitTest, UnitIntFloatTypes);
TYPED_TEST_SUITE(OffsetLayoutViewUnitTest, UnitIntFloatTypes);
TYPED_TEST_SUITE(TypedIntegralViewUnitTest, UnitIntFloatTypes);

TYPED_TEST(TypedViewUnitTest, Constructors)
{

  using layout = RAJA::Layout<1>;

  TypeParam   a1[10];
  TypeParam   a2[10];
  TypeParam * data[2];

  data[0] = a1;
  data[1] = a2;

  RAJA::MultiView<TypeParam, layout> view(data, layout(10));

  /*
   * Should be able to construct a non-const View from a non-const View
   */
  RAJA::MultiView<TypeParam, layout> view2(view);

  /*
   * Should be able to construct a const View from a non-const View
   */
  RAJA::MultiView<TypeParam const, layout> const_view(view);

  /*
   * Should be able to construct a const View from a const View
   */
  RAJA::MultiView<TypeParam const, layout> const_view2(const_view);
}

TYPED_TEST(TypedViewUnitTest, Accessor)
{

  const int Nx = 3;
  const int Ny = 5;
  const int Nz = 2;
  const int N  = Nx*Ny*Nz;
  TypeParam *b = new TypeParam[N];
  TypeParam *c = new TypeParam[N];
  TypeParam *a[2];

  a[0] = b;
  a[1] = c;

  int iter{0};
  for(TypeParam i=0; i<TypeParam{N}; ++i)
  {
    a[0][iter] = TypeParam{i};
    a[1][iter] = TypeParam{i};
    ++iter;
  }

  /*
   * 1D Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<1>> view_1D(a,N);
  TypeParam val{0};
  for(int i=0; i<N; ++i) {
    ASSERT_EQ(val, view_1D(0,i));
    val++;
  }

  /*
   * 2D Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<2>> view_2D(a,Ny,Nx);
  val = TypeParam{0};
  for(int j=0; j<Ny; ++j) {
    for(int i=0; i<Nx; ++i) {
      ASSERT_EQ(val, view_2D(0,j,i));
      val++;
    }
  }

  /*
   * 3D Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<3>> view_3D(a,Nz,Ny,Nx);
  val = TypeParam{0};
  for(int k=0; k<Nz; ++k) {
    for(int j=0; j<Ny; ++j) {
      for(int i=0; i<Nx; ++i) {
        ASSERT_EQ(val, view_3D(0,k,j,i));
        val++;
      }
    }
  }

  delete[] b;
  delete[] c;
}
TYPED_TEST(TypedIntegralViewUnitTest, TypedAccessor)
{

  const int Nx = 3;
  const int Ny = 5;
  const int Nz = 2;
  const int N  = Nx*Ny*Nz;
  TypeParam *b = new TypeParam[N];
  TypeParam *c = new TypeParam[N];
  TypeParam *a[2];

  a[0] = b;
  a[1] = c;

  int iter{0};
  for(TypeParam i=0; i<TypeParam{N}; ++i)
  {
    a[0][iter] = TypeParam{i};
    a[1][iter] = TypeParam{i};
    ++iter;
  }

  /*
   * 1D Typed Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<1>> view_1D(a,N);
  TypeParam val{0};
  for(TypeParam i=0; i<N; ++i) {
    ASSERT_EQ(val, view_1D(0,i));
    val++;
  }

  /*
   * 2D Typed Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<2>> view_2D(a,Ny,Nx);
  val = TypeParam{0};
  for(TypeParam j=0; j<Ny; ++j) {
    for(TypeParam i=0; i<Nx; ++i) {
      ASSERT_EQ(val, view_2D(0,j,i));
      val++;
    }
  }

  /*
   * 3D Typed Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<3>> view_3D(a,Nz,Ny,Nx);
  val = TypeParam{0};
  for(TypeParam k=0; k<Nz; ++k) {
    for(TypeParam j=0; j<Ny; ++j) {
      for(TypeParam i=0; i<Nx; ++i) {
        ASSERT_EQ(val, view_3D(0,k,j,i));
        val++;
      }
    }
  }

  delete[] b;
  delete[] c;
}

TYPED_TEST(OffsetLayoutViewUnitTest, View)
{
  TypeParam* d1 = new TypeParam[10];
  TypeParam* d2 = new TypeParam[10];
  TypeParam* data[2];

  data[0] = d1;
  data[1] = d2;

  using layout = RAJA::OffsetLayout<>;

  /*
   * View is constructed by passing in the layout.
   */
  std::array<RAJA::Index_type, 1> lower{{1}};
  std::array<RAJA::Index_type, 1> upper{{10}};
  RAJA::MultiView<TypeParam, layout> view(data, RAJA::make_offset_layout<1>(lower, upper));

  for (int i = 0; i < 10; i++) {
    data[0][i] = static_cast<TypeParam>(i);
    data[1][i] = static_cast<TypeParam>(i);
  }

  ASSERT_EQ(data[0][0], view(0,1));
  ASSERT_EQ(data[1][9], view(0,10));

  delete[] d1;
  delete[] d2;
}
