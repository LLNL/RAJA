//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-reduce-sanity-tests.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "../test-forall-execpol.hpp"
#include "../test-reducepol.hpp"
#include "../test-forall-utils.hpp"

// Cartesian product of types for OpenMP target tests
using OpenMPTargetForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList, 
                                OpenMPTargetResourceList, 
                                OpenMPTargetForallExecPols,
                                OpenMPTargetReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTarget,
                               ForallReduceMaxLocSanityTest,
                               OpenMPTargetForallReduceSanityTypes);

#endif
