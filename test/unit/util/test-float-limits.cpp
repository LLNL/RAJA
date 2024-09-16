//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for floating point numeric limits in
/// RAJA operators
///

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

#define RAJA_CHECK_LIMITS
#include "RAJA/util/Operators.hpp"

#include <limits>

template <typename T>
class FloatLimitsUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P(FloatLimitsUnitTest);

TYPED_TEST_P(FloatLimitsUnitTest, FloatLimits)
{
#if !defined(RAJA_ENABLE_TARGET_OPENMP)
  ASSERT_EQ(
      RAJA::operators::limits<TypeParam>::min(),
      -std::numeric_limits<TypeParam>::max());
  ASSERT_EQ(
      RAJA::operators::limits<TypeParam>::max(),
      std::numeric_limits<TypeParam>::max());
#endif
}

REGISTER_TYPED_TEST_SUITE_P(FloatLimitsUnitTest, FloatLimits);

INSTANTIATE_TYPED_TEST_SUITE_P(
    FloatLimitsUnitTests,
    FloatLimitsUnitTest,
    UnitFloatTypes);
