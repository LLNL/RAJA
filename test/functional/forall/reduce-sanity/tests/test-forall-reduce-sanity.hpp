//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCE_SANITY_HPP__
#define __TEST_FORALL_REDUCE_SANITY_HPP__

#include "gtest/gtest.h"

#include "../../test-forall-utils.hpp"

TYPED_TEST_SUITE_P(ForallReduceSanityTest);
template <typename T>
class ForallReduceSanityTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ForallReduceSanityTestBit);
template <typename T>
class ForallReduceSanityTestBit : public ::testing::Test
{
};

//
// Data types for reduction sanity tests
//
using ReduceSanityDataTypeList = camp::list<int,
                                            float,
                                            double>;

using ReduceSanityDataTypeListBit = camp::list<int>;

#include "test-forall-reduce-sanity-sum.hpp"
#include "test-forall-reduce-sanity-bitor.hpp"
#include "test-forall-reduce-sanity-min.hpp"
#include "test-forall-reduce-sanity-max.hpp"
#include "test-forall-reduce-sanity-minloc.hpp"
#include "test-forall-reduce-sanity-maxloc.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallReduceSanityTest,
                            ReduceSumSanityForall,
                            ReduceMinSanityForall,
                            ReduceMaxSanityForall,
                            ReduceMinLocSanityForall,
                            ReduceMaxLocSanityForall);

REGISTER_TYPED_TEST_SUITE_P(ForallReduceSanityTestBit,
                            ReduceBitOrSanityForall);

#endif  // __TEST_FORALL_REDUCE_SANITY_HPP__
