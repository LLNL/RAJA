//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for numeric limits in RAJA operators
///

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

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

INSTANTIATE_TYPED_TEST_SUITE_P(IntegralLimitsUnitTests,
                              IntegralLimitsUnitTest,
                              UnitIntegralTypes);
