//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset.hpp"

// OpenMP execution policy types
using OpenMPForallIndexSetExecPols = 
  camp::list< RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_segit, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_segit, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_nowait_exec> >;

// Cartesian product of types for OpenMP tests
using OpenMPForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList, 
                                OpenMPForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP,
                               ForallIndexSetTest,
                               OpenMPForallIndexSetTypes);
