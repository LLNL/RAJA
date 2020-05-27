//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset.hpp"

#include "../test-forall-indexset-execpol.hpp"

// Cartesian product of types for Sequential tests
using SequentialForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList, 
                                SequentialForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallIndexSetTest,
                               SequentialForallIndexSetTypes);
