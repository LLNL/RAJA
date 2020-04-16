//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset-view.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "../test-forall-indexset-execpol.hpp"

// Cartesian product of types for Hip tests
using HipForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HipResourceList, 
                                HipForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallIndexSetViewTest,
                               HipForallIndexSetTypes);

#endif
