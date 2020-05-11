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


//
// Data types for reduction sanity tests
//
using ReduceSanityDataTypeList = camp::list<int,
                                            float,
                                            double>;

#include "test-forall-reducesum-sanity.hpp"
#if 0
#include "test-forall-reducemin-sanity.hpp"
#include "test-forall-reducemax-sanity.hpp"
#include "test-forall-reduceminloc-sanity.hpp"
#include "test-forall-reducemaxloc-sanity.hpp"
#endif

REGISTER_TYPED_TEST_SUITE_P(ForallReduceSanityTest,
                            ReduceSumSanityForall);
#if 0
                            ReduceMinSanityForall,
                            ReduceMaxSanityForall,
                            ReduceMinLocSanityForall,
                            ReduceMaxLocSanityForall);
#endif

#endif  // __TEST_FORALL_REDUCE_SANITY_HPP__
