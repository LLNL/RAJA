//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment.hpp"

#include "../test-forall-execpol.hpp"

// Cartesian product of types for Sequential tests
using SequentialForallSegmentTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList, 
                                SequentialForallExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallSegmentTest,
                               SequentialForallSegmentTypes);
