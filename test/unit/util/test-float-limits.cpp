//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for floating point numeric limits in 
/// RAJA operators
///

#include "gtest/gtest.h"

#define RAJA_CHECK_LIMITS
#include "RAJA/util/Operators.hpp"

#include <limits>

template <typename T>
class FloatLimitsTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(FloatLimitsTest);

TYPED_TEST_P(FloatLimitsTest, FloatLimits)
{
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::min(),
            -std::numeric_limits<TypeParam>::max());
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::max(),
            std::numeric_limits<TypeParam>::max());
}

REGISTER_TYPED_TEST_CASE_P(FloatLimitsTest, FloatLimits);

using float_types = ::testing::Types<float,
                                     double,
                                     long double>;

INSTANTIATE_TYPED_TEST_CASE_P(FloatLimitsTests,
                              FloatLimitsTest,
                              float_types);
