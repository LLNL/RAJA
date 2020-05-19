//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-region.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "../test-forall-utils.hpp"

using OpenMPRegionPols = camp::list< RAJA::omp_parallel_region >;

using OpenMPForallExecPols =
  camp::list< RAJA::omp_for_nowait_exec,
              RAJA::omp_for_exec >;

// Cartesian product of types for OpenMP tests
using OpenMPForallRegionTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList,
                                OpenMPRegionPols,
                                OpenMPForallExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP,
                               ForallRegionTest,
                               OpenMPForallRegionTypes);

#endif
