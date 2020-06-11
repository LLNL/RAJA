//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_ATOMIC_VIEW_HPP__
#define __TEST_FORALL_ATOMIC_VIEW_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index.hpp"

#include "RAJA_test-atomic-types.hpp"
#include "RAJA_test-atomicpol.hpp"

#include "RAJA_test-forall-execpol.hpp"

TYPED_TEST_SUITE_P(ForallAtomicViewFunctionalTest);
template <typename T>
class ForallAtomicViewFunctionalTest : public ::testing::Test
{
};

#include "tests/test-forall-atomic-view-test.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicViewFunctionalTest,
                            AtomicViewFunctionalForall);

#endif  // __TEST_FORALL_ATOMIC_VIEW_HPP__
