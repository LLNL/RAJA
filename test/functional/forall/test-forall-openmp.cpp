//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment.hpp"

#if defined(RAJA_ENABLE_OPENMP)

// OpenMP execution policy types
using OpenMPTypes = list< RAJA::omp_parallel_exec<RAJA::seq_exec>,
                          RAJA::omp_for_nowait_exec,
                          RAJA::omp_for_exec,
                          RAJA::omp_parallel_for_exec >;

// Cartesian product of types for OpenMP tests
using OpenMPForallSegmentTypes =
  Test< cartesian_product<IdxTypeList, 
                          HostResourceList,
                          OpenMPTypes> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP,
                               ForallSegmentTest,
                               OpenMPForallSegmentTypes);

#endif
