//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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
/// Source file containing tests for numeric limits in RAJA operators
///

#include "gtest/gtest.h"

#define RAJA_CHECK_LIMITS
#include "RAJA/util/Operators.hpp"

#include <limits>

template <typename T>
class IntegralLimitsTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(IntegralLimitsTest);

TYPED_TEST_P(IntegralLimitsTest, IntegralLimits)
{
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::min(),
            std::numeric_limits<TypeParam>::min());
  ASSERT_EQ(RAJA::operators::limits<TypeParam>::max(),
            std::numeric_limits<TypeParam>::max());
}

REGISTER_TYPED_TEST_CASE_P(IntegralLimitsTest, IntegralLimits);

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

INSTANTIATE_TYPED_TEST_CASE_P(IntegralLimitsTests,
                              IntegralLimitsTest,
                              integer_types);
