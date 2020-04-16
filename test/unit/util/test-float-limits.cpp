//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
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
class FloatLimitsUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(FloatLimitsUnitTest);

TYPED_TEST_P(FloatLimitsUnitTest, FloatLimits)
{
#if !defined(RAJA_ENABLE_TARGET_OPENMP)
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::min(),
            -std::numeric_limits<TypeParam>::max());
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::max(),
            std::numeric_limits<TypeParam>::max());
#endif
}

REGISTER_TYPED_TEST_SUITE_P(FloatLimitsUnitTest, FloatLimits);

using float_types = ::testing::Types<float,
                                     double,
                                     long double>;

INSTANTIATE_TYPED_TEST_SUITE_P(FloatLimitsUnitTests,
                               FloatLimitsUnitTest,
                               float_types);
