//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall.hpp"

// Sequential execution policy types
using SequentialTypes = list< RAJA::seq_exec,
                              RAJA::loop_exec,
                              RAJA::simd_exec >;

// Sequential tests index, resource, and execution policy types
using SequentialForallTypes =
    Test<cartesian_product<IdxTypes, ListHost, SequentialTypes>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallFunctionalSegmentTest,
                               SequentialForallTypes);
