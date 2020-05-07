//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-region.hpp"

#include "../test-forall-execpol.hpp"

using SequentialRegionPols = camp::list< RAJA::seq_region >;

// Cartesian product of types for Sequential tests
using SequentialForallRegionTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList,
                                SequentialRegionPols,
                                SequentialForallExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallRegionTest,
                               SequentialForallRegionTypes);
