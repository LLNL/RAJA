//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for numeric limits in RAJA operators
///

#include "gtest/gtest.h"

#define RAJA_CHECK_LIMITS
#include "RAJA/util/Operators.hpp"

#include <limits>

template <typename T>
class IntegralLimitsUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(IntegralLimitsUnitTest);

TYPED_TEST_P(IntegralLimitsUnitTest, IntegralLimits)
{
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::min(),
            std::numeric_limits<TypeParam>::min());
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::max(),
            std::numeric_limits<TypeParam>::max());
}

REGISTER_TYPED_TEST_SUITE_P(IntegralLimitsUnitTest, IntegralLimits);

using integer_types = ::testing::Types<char,
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

INSTANTIATE_TYPED_TEST_SUITE_P(IntegralLimitsUnitTests,
                              IntegralLimitsUnitTest,
                              integer_types);
