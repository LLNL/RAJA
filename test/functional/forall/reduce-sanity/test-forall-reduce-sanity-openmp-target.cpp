//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-reduce-sanity.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "../test-forall-execpol.hpp"
#include "../test-reducepol.hpp"

// Cartesian product of types for OpenMP target tests
using OpenMPTargetForallReduceSanityTypes =
  Test< camp::cartesian_product<ReduceSanityDataTypeList, 
                                OpenMPTargetResourceList, 
                                OpenMPTargetForallExecPols,
                                OpenMPTargetReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTarget,
                               ForallReduceSanityTest,
                               OpenMPTargetForallReduceSanityTypes);

#endif
