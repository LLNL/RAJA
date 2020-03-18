//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-rangesegment.hpp"

// Generate OMP Target Type List
#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OMPTargetTypes = list<RAJA::omp_target_parallel_for_exec<8>,
                            RAJA::omp_target_parallel_for_exec_nt>;

using OMPTargetForallTypes =
    Test<cartesian_product<IdxTypes, ListHost, OMPTargetTypes>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TargetOmp,
                               ForallFunctionalTest,
                               OMPTargetForallTypes);
#endif
