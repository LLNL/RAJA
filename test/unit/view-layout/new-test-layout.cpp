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
TYPED_TEST_CASE(OffsetLayoutTest,MyTypes);

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

TYPED_TEST(OffsetLayoutTest, Constructors)
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
