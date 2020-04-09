//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-scan-inclusive.hpp"

#if defined(RAJA_ENABLE_OPENMP)

// OpenMP policy types to test
using OpenMPExecTypes = camp::list< RAJA::omp_parallel_for_exec, 
                                    RAJA::omp_for_exec,
                                    RAJA::omp_for_nowait_exec >;

using OpenMPInclusiveScanTypes = 
  Test<cartesian_product< OpenMPExecTypes, ListHostRes, OpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP, 
                               ScanFunctionalTest, 
                               OpenMPInclusiveScanTypes);

#endif
