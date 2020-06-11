//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCE_SANITY_HPP__
#define __TEST_FORALL_REDUCE_SANITY_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index-types.hpp"

#include "RAJA_test-forall-data.hpp"
#include "RAJA_test-forall-execpol.hpp"
#include "RAJA_test-reducepol.hpp"

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

#include "tests/test-forall-reduce-sanity-sum.hpp"
#include "tests/test-forall-reduce-sanity-min.hpp"
#include "tests/test-forall-reduce-sanity-max.hpp"
#include "tests/test-forall-reduce-sanity-minloc.hpp"
#include "tests/test-forall-reduce-sanity-maxloc.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallReduceSanityTest,
                            ReduceSumSanityForall,
                            ReduceMinSanityForall,
                            ReduceMaxSanityForall,
                            ReduceMinLocSanityForall,
                            ReduceMaxLocSanityForall);

#endif  // __TEST_FORALL_REDUCE_SANITY_HPP__
