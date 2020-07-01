//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-kernel-region.hpp"

#if defined(RAJA_ENABLE_OPENMP)

using OpenMPKernelRegionExecPols = 
  camp::list< 

    RAJA::KernelPolicy<
      RAJA::statement::Region<RAJA::omp_parallel_region,
        RAJA::statement::For<0, RAJA::omp_for_nowait_exec,
          RAJA::statement::Lambda<0>
        >,
        RAJA::statement::For<0, RAJA::omp_for_nowait_exec,
          RAJA::statement::Lambda<1>
        >,
        RAJA::statement::For<0, RAJA::omp_for_nowait_exec,
          RAJA::statement::Lambda<2>
        >
      >
    >,

    RAJA::KernelPolicy<
      RAJA::statement::Region<RAJA::omp_parallel_region,
        RAJA::statement::For<0, RAJA::omp_for_exec,
          RAJA::statement::Lambda<0>
        >,
        RAJA::statement::For<0, RAJA::omp_for_exec,
          RAJA::statement::Lambda<1>
        >,
        RAJA::statement::For<0, RAJA::omp_for_exec,
          RAJA::statement::Lambda<2>
        >
      >
    >

  >;


// Cartesian product of types for OpenMP tests
using OpenMPKernelRegionTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList,
                                OpenMPKernelRegionExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP,
                               KernelRegionBasicTest,
                               OpenMPKernelRegionTypes);

#endif
