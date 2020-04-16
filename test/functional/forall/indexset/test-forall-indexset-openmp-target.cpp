//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset.hpp"

// OpenMP target execution policy types
using OpenMPTargetForallIndexSetExecPols = 
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, 
                               RAJA::omp_target_parallel_for_exec<8>>,
              RAJA::ExecPolicy<RAJA::seq_segit, 
                               RAJA::omp_target_parallel_for_exec_nt> >;

// Cartesian product of types for OpenMP target tests
using OpenMPTargetForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList, 
                                OpenMPTargetForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallIndexSetTest,
                               OpenMPTargetForallIndexSetTypes);
