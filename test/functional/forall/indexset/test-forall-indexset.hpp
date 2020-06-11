//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_HPP__
#define __TEST_FORALL_INDEXSET_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index-types.hpp"

#include "RAJA_test-forall-data.hpp"

#include "RAJA_test-indexset-execpol.hpp"
#include "RAJA_test-indexset-build.hpp"

TYPED_TEST_SUITE_P(ForallIndexSetTest);
template <typename T>
class ForallIndexSetTest : public ::testing::Test
{
};

#include "tests/test-forall-basic-indexset.hpp"
#include "tests/test-forall-icount-indexset.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetTest,
                            IndexSetForall,
                            IndexSetForall_Icount);

#endif  // __TEST_FORALL_INDEXSET_HPP__
