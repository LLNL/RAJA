//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_ATOMIC_REF_HPP__
#define __TEST_FORALL_ATOMIC_REF_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index.hpp"

#include "RAJA_test-atomic-types.hpp"
#include "RAJA_test-atomicpol.hpp"

#include "RAJA_test-forall-execpol.hpp"

TYPED_TEST_SUITE_P(ForallAtomicRefMathFunctionalTest);
template <typename T>
class ForallAtomicRefMathFunctionalTest : public ::testing::Test
{
};

#include "tests/test-forall-atomic-ref-math.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefMathFunctionalTest,
                            AtomicRefMathFunctionalForall);


TYPED_TEST_SUITE_P(ForallAtomicRefOtherFunctionalTest);
template <typename T>
class ForallAtomicRefOtherFunctionalTest : public ::testing::Test
{
};

#include "tests/test-forall-atomic-ref-other.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefOtherFunctionalTest,
                            AtomicRefOtherFunctionalForall);

#endif  // __TEST_FORALL_ATOMIC_REF_HPP__
