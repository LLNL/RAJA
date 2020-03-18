//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-rangesegment.hpp"

// Generate OMP Type List
#if defined(RAJA_ENABLE_OPENMP)
using OMPTypes = list<RAJA::omp_parallel_exec<RAJA::seq_exec>,
                      RAJA::omp_for_nowait_exec,
                      RAJA::omp_for_exec,
                      RAJA::omp_parallel_for_exec>;

using OMPForallTypes =
    Test<cartesian_product<IdxTypes, ListHost, OMPTypes>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Omp,
                               ForallFunctionalTest,
                               OMPForallTypes);
#endif
