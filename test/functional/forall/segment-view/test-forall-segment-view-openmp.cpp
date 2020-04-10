//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment-view.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "../test-forall-execpol.hpp"

// Cartesian product of types for OpenMP tests
using OpenMPForallSegmentTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList,
                                OpenMPForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP,
                               ForallSegmentViewTest,
                               OpenMPForallSegmentTypes);

#endif
