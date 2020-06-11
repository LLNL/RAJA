//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_ATOMIC_BASIC_HPP__
#define __TEST_FORALL_ATOMIC_BASIC_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index.hpp"

#include "RAJA_test-atomic-types.hpp"
#include "RAJA_test-atomicpol.hpp"

#include "RAJA_test-forall-execpol.hpp"

TYPED_TEST_SUITE_P(ForallAtomicBasicFunctionalTest);
template <typename T>
class ForallAtomicBasicFunctionalTest : public ::testing::Test
{
};

#include "tests/test-forall-atomic-basic-test.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicBasicFunctionalTest,
                            AtomicBasicFunctionalForall);

#endif  // __TEST_FORALL_ATOMIC_BASIC_HPP__
