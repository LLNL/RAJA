//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

// OpenMP target execution policy types
using OpenMPTargetForallExecPols = list< RAJA::omp_target_parallel_for_exec<8>,
                                         RAJA::omp_target_parallel_for_exec_nt >;

// Cartesian product of types for OpenMP target tests
using OpenMPTargetForallSegmentTypes =
  Test< cartesian_product<IdxTypeList, 
                          OpenMPTargetResourceList, 
                          OpenMPTargetForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTarget,
                               ForallSegmentTest,
                               OpenMPTargetForallSegmentTypes);
#endif
