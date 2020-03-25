//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment.hpp"

#if defined(RAJA_ENABLE_HIP)

// Hip execution policy types
using HipForallExecPols = list< RAJA::hip_exec<128>,
                                RAJA::hip_exec<256>  >;

// Cartesian product of types for Hip tests
using HipForallSegmentTypes = 
  Test< cartesian_product<IdxTypeList, 
                          HipResourceList, 
                          HipForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallSegmentTest,
                               HipForallSegmentTypes);

#endif
