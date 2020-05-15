//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_VIEW__HPP__
#define __TEST_FORALL_INDEXSET_VIEW__HPP__

#include "RAJA/RAJA.hpp"

#include "../../test-forall-utils.hpp"

TYPED_TEST_SUITE_P(ForallIndexSetViewTest);
template <typename T>
class ForallIndexSetViewTest : public ::testing::Test
{
};

#include "test-forall-basic-indexset-view.hpp"
#include "test-forall-icount-indexset-view.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallIndexSetViewTest,
                            IndexSetForallView,
                            IndexSetForall_IcountView);

#endif  // __TEST_FORALL_INDEXSET_VIEW__HPP__
